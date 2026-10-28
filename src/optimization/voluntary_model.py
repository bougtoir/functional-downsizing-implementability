"""Model B: public-service downsizing under voluntary residential choice.

Leader (public authority) chooses facility openings, service-mode shares, and
relocation offers (destination + per-mover subsidy package). Household
clusters choose to stay or accept according to a documented deterministic
choice rule; the planner cannot relocate anyone without an acceptable offer.

Implemented as a single-level MILP (reformulation option B in
docs/MODEL_SPECIFICATION_B.md). Equivalence to the Stackelberg problem under
the stated choice rule is verified on toy instances by enumeration in
tests/test_voluntary_model.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .service_model import (
    LinearModel,
    candidate_edges,
    facility_capacity,
    unit_value,
)

MODEL_VERSION = "2026-10-25.1"

DEFAULT_BEHAVIOUR = {
    "attachment_tau": 0.0,        # per-cluster override via column tau
    "move_burden_kappa": 0.02,    # utility loss per km moved (normalized)
    "access_utility_beta": 0.5,   # converts stay-side provision share to utility
    "baseline_utility_u0": 0.3,   # status-quo housing/amenity value at origin
    "destination_value_h": 0.3,   # housing/amenity value at destination centre
    "participation_rho": 1.0,     # max share of a cluster that can elect to move
    "tie_margin_delta": 1e-4,     # conservative tie-breaking: indifferent -> stay
    "subsidy_package_S": 0.0,     # per-mover transfer offered when relocating
    "public_budget": np.inf,      # E_pub cap; np.inf = unconstrained
    "max_total_offers": np.inf,   # global cap on simultaneous offers (inf = coalitions free)
    # When True: drop offer/acceptance variables entirely and let the planner
    # move any share at the LOWER BOUND of the acceptance threshold,
    #   S_ij^LB = u0 + tau_i + delta - h + kappa*d_ij
    # (the true threshold T_min(i,j) = S_ij^LB + beta*F_stay >= S_ij^LB).
    # If the optimum under this move-favouring pricing still relocates nobody,
    # NO coalition, destination assignment or facility re-optimization can
    # profitably switch under the real thresholds: a single-solve
    # implementability certificate over all 2^|I| coalitions.
    "free_move_at_lb": False,
}


@dataclass
class VoluntarySolution:
    summary: dict
    demand: pd.DataFrame
    facilities: pd.DataFrame
    offers: pd.DataFrame = field(default_factory=pd.DataFrame)
    edges: pd.DataFrame = field(default_factory=pd.DataFrame)


DEFAULT_COOPERATION = {
    "enabled": False,
    "setup_cost": 0.0,          # fixed cost per (origin, destination) municipal agreement
    "per_person_cost": 0.0,     # coordination cost per cross-boundary resident
    "municipal_budgets": None,  # optional dict mun -> budget cap on ITS residents' subsidy share
}


def solve_voluntary_model(
    demand_input: pd.DataFrame,
    facilities_input: pd.DataFrame,
    config: dict,
    behaviour: dict | None = None,
    cooperation: dict | None = None,
    population_decline: float = 0.0,
    *,
    same_municipality_only: bool = False,
    time_limit: float | None = None,
    mip_rel_gap: float | None = None,
) -> VoluntarySolution:
    """Solve model B, optionally with the C extension (intermunicipal
    agreements: fixed setup + per-person coordination cost).

    demand_input may carry a per-cluster "tau" column (attachment threshold);
    otherwise behaviour["attachment_tau"] is used uniformly.
    """
    coop = dict(DEFAULT_COOPERATION)
    if cooperation:
        coop.update(cooperation)
    beh = dict(DEFAULT_BEHAVIOUR)
    if behaviour:
        beh.update(behaviour)

    demand = demand_input.copy().reset_index(drop=True)
    facilities = facilities_input.copy().reset_index(drop=True)
    demand["population"] *= 1 - population_decline
    if "tau" not in demand.columns:
        demand["tau"] = float(beh["attachment_tau"])

    candidates = facilities.index[facilities["is_candidate"]].tolist()
    municipalities = sorted(demand["municipality_code"].astype(str).unique())
    edge_config = config["candidate_edges"]
    edges = candidate_edges(
        demand,
        facilities,
        "B" if same_municipality_only else "D",
        edge_config["same_municipality_limit"],
        edge_config["regional_limit"],
    )
    edge_map: dict[int, list[tuple[int, float]]] = {i: [] for i in demand.index}
    for di, fi, dkm in edges:
        edge_map[di].append((fi, dkm))

    model = LinearModel()
    costs = config["costs"]
    substitution = config["substitution"]
    threshold_minutes = config["mfg"]["healthcare_max_minutes"]
    circuity = config["geography"]["assumed_road_circuity"]
    speed = config["geography"]["assumed_average_speed_kmh"]

    S = float(beh["subsidy_package_S"])
    delta = float(beh["tie_margin_delta"])
    kappa = float(beh["move_burden_kappa"])
    beta = float(beh["access_utility_beta"])
    u0 = float(beh["baseline_utility_u0"])
    h = float(beh["destination_value_h"])
    rho = float(beh["participation_rho"])
    budget = float(beh["public_budget"])
    free_move = bool(beh.get("free_move_at_lb", False))
    # Per-edge big-M for the acceptance implication (o_ij=1 => const - beta*F >= 0).
    # When o_ij=0 the constraint must be vacuous for all F in [0,1]:
    #   -beta*F >= -const - M_ij  for the tightest F (F=1) requires
    #   M_ij >= -const + beta. Using M_ij = beta + max(0, -const) achieves this.
    def edge_big_m(const: float) -> float:
        return beta + max(0.0, -const) + 1e-3

    # --- facility open variables (as M0, incl. transition accounting) ---
    y: dict[int, int] = {}
    existing_candidates = 0
    for fi in candidates:
        fac = facilities.loc[fi]
        existing = bool(fac["is_existing"])
        if existing:
            existing_candidates += 1
        transition_adjustment = -costs["transition_per_changed_asset"] if existing else 0.0
        opening_transition = 0.0 if existing else costs["transition_per_changed_asset"]
        y[fi] = model.variable(
            f"open[{fi}]",
            cost=costs["fixed_facility_open"]
            + facility_capacity(fac, config) * costs["fixed_facility_capacity_unit"]
            + transition_adjustment
            + opening_transition,
            upper=1.0,
            integer=True,
        )

    mobile_hub: dict[str, int] = {}
    for mun in municipalities:
        mobile_hub[mun] = model.variable(
            f"mobile_hub[{mun}]", cost=costs["mobile_hub_open"], upper=1.0, integer=True
        )

    direct: dict[tuple[int, int], int] = {}
    offer: dict[tuple[int, int], int] = {}
    moved: dict[tuple[int, int], int] = {}
    mobile: dict[int, int] = {}
    digital: dict[int, int] = {}
    slack: dict[int, int] = {}
    edge_distance: dict[tuple[int, int], float] = {}
    move_price: dict[tuple[int, int], float] = {}

    for di, node in demand.iterrows():
        pop = float(node["population"])
        for fi, dkm in edge_map[di]:
            edge_distance[(di, fi)] = dkm
            # direct assignment: no fiscal cost (access is private); kept for
            # quality expression only.
            direct[(di, fi)] = model.variable(f"direct[{di},{fi}]", upper=1.0)
            offer[(di, fi)] = model.variable(
                f"offer[{di},{fi}]", upper=1.0, integer=True
            )
            if free_move:
                s_lb = u0 + float(node["tau"]) + delta - h + kappa * dkm
            else:
                s_lb = S
            move_price[(di, fi)] = s_lb
            moved[(di, fi)] = model.variable(
                f"move[{di},{fi}]",
                cost=pop * s_lb,  # fiscal: per-mover subsidy transfer
                upper=rho if free_move else 1.0,
            )
        mobile[di] = model.variable(
            f"mobile[{di}]",
            cost=pop * costs["mobile_service_per_resident"],
            upper=substitution["healthcare_mobile_max_share"],
        )
        digital[di] = model.variable(
            f"digital[{di}]",
            cost=pop * costs["digital_service_per_resident"],
            upper=substitution["healthcare_digital_max_share"],
        )
        slack[di] = model.variable(
            f"mfg_slack[{di}]",
            cost=pop * config["mfg"]["permitted_diagnostic_slack_penalty"],
            upper=1.0,
        )

    theta = config["mfg"]["minimum_effective_provision"]
    for di, node in demand.iterrows():
        # population conservation: staying shares + moved shares = 1
        share = {direct[(di, fi)]: 1.0 for fi, _ in edge_map[di]}
        for fi, _ in edge_map[di]:
            share[moved[(di, fi)]] = 1.0
        share[mobile[di]] = 1.0
        share[digital[di]] = 1.0
        model.constraint(share, lower=1.0, upper=1.0)

        # staying-share quality numerator F_i^stay
        f_stay: dict[int, float] = {}
        for fi, dkm in edge_map[di]:
            minutes = dkm * circuity / speed * 60
            f_stay[direct[(di, fi)]] = min(1.0, threshold_minutes / max(threshold_minutes, minutes))
        f_stay[mobile[di]] = substitution["mobile_quality_effectiveness"]
        f_stay[digital[di]] = substitution["digital_quality_effectiveness"]

        # MFG for stayers: F_i^stay >= theta * staying_share - slack
        mfg = dict(f_stay)
        for fi, _ in edge_map[di]:
            mfg[moved[(di, fi)]] = theta
        mfg[slack[di]] = 1.0
        model.constraint(mfg, lower=theta)

        model.constraint(
            {mobile[di]: 1.0, digital[di]: 1.0},
            upper=substitution["healthcare_combined_max_share"],
        )
        mun = str(node["municipality_code"])
        model.constraint(
            {mobile[di]: 1.0, mobile_hub[mun]: -substitution["healthcare_mobile_max_share"]},
            upper=0.0,
        )

        # at most ONE offered destination per cluster (documented baseline)
        if edge_map[di] and not free_move:
            model.constraint(
                {offer[(di, fi)]: 1.0 for fi, _ in edge_map[di]}, upper=1.0
            )

        tau_i = float(node["tau"])
        for fi, dkm in edge_map[di]:
            if free_move:
                # certificate mode: moves are free at the LB transfer; only
                # the destination facility must be open.
                model.constraint({moved[(di, fi)]: 1.0, y[fi]: -1.0}, upper=0.0)
            else:
                # deterministic choice: an acceptable offer is taken by exactly
                # the participating share rho (moved = rho * offer); dest. open.
                model.constraint({moved[(di, fi)]: 1.0, offer[(di, fi)]: -rho}, lower=0.0)
                model.constraint({moved[(di, fi)]: 1.0, offer[(di, fi)]: -rho}, upper=0.0)
                model.constraint({moved[(di, fi)]: 1.0, y[fi]: -1.0}, upper=0.0)
                # offer only if acceptable: U_move - U_stay >= delta when offer=1
                acc = {fi_var: -beta for fi_var in f_stay}
                const = h + S - kappa * dkm - u0 - tau_i - delta
                m_ij = edge_big_m(const)
                acc[offer[(di, fi)]] = -m_ij
                # const - beta*F_stay >= 0 when offer=1; vacuous when offer=0:
                # -beta*F_stay - M*o >= -const - M
                model.constraint(acc, lower=-const - m_ij)
            # direct assignment requires facility open
            model.constraint({direct[(di, fi)]: 1.0, y[fi]: -1.0}, upper=0.0)

    # facility capacity: direct + relocated population <= capacity
    for fi in candidates:
        coef = {}
        for di, node in demand.iterrows():
            pop = float(node["population"])
            if (di, fi) in direct:
                coef[direct[(di, fi)]] = pop
            if (di, fi) in moved:
                coef[moved[(di, fi)]] = pop
        coef[y[fi]] = -facility_capacity(facilities.loc[fi], config)
        model.constraint(coef, upper=0.0)

    # municipal mobile hub capacity
    for mun in municipalities:
        member = demand["municipality_code"].astype(str) == mun
        coef = {mobile[i]: float(demand.loc[i, "population"]) for i in demand.index[member]}
        mun_pop = float(demand.loc[member, "population"].sum())
        coef[mobile_hub[mun]] = -substitution["mobile_hub_capacity_share"] * mun_pop
        model.constraint(coef, upper=0.0)

    # global cap on simultaneous offers (coalition-size restriction)
    if np.isfinite(float(beh["max_total_offers"])):
        model.constraint(
            {v: 1.0 for v in offer.values()},
            upper=float(beh["max_total_offers"]),
        )

    # --- C extension: intermunicipal agreements with institutional costs ---
    agreement: dict[tuple[str, str], int] = {}
    if coop["enabled"]:
        pairs = set()
        for di, node in demand.iterrows():
            om = str(node["municipality_code"])
            for fi, _ in edge_map[di]:
                fm = str(facilities.loc[fi, "municipality_code"])
                if fm != om:
                    pairs.add((om, fm))
        for om, fm in sorted(pairs):
            agreement[(om, fm)] = model.variable(
                f"agreement[{om},{fm}]",
                cost=float(coop["setup_cost"]),
                upper=1.0,
                integer=True,
            )
        for di, node in demand.iterrows():
            om = str(node["municipality_code"])
            pop = float(node["population"])
            for fi, _ in edge_map[di]:
                fm = str(facilities.loc[fi, "municipality_code"])
                if fm == om:
                    continue
                z = agreement[(om, fm)]
                for var in (direct[(di, fi)], moved[(di, fi)], offer[(di, fi)]):
                    model.constraint({var: 1.0, z: -1.0}, upper=0.0)
                # per-person coordination charge on cross-boundary use
                model.cost[direct[(di, fi)]] += pop * float(coop["per_person_cost"])
                model.cost[moved[(di, fi)]] += pop * float(coop["per_person_cost"])

    # fiscal budget (transfers included; E_pub <= B)
    if np.isfinite(budget):
        budget_row = {i: c for i, c in enumerate(model.cost)}
        model.constraint(budget_row, upper=budget)

    # Lexicographic second stage: given the certified optimum E* (epub_cap),
    # maximize total moved population subject to E_pub <= epub_cap. A zero
    # answer means no even-weakly-optimal plan relocates anyone, so the
    # implementable set is empty up to objective ties, not just strictly.
    if bool(beh.get("lexicographic_movers", False)):
        cap = float(beh.get("epub_cap", np.inf))
        assert np.isfinite(cap), "lexicographic_movers requires epub_cap"
        model.constraint({i: c for i, c in enumerate(model.cost)}, upper=cap)
        model.cost = [0.0] * len(model.cost)
        for (di, fi), var in moved.items():
            model.cost[var] = -float(demand.loc[di, "population"])

    # optional per-municipality budget: subsidy outlay charged to origin mun
    mun_budgets = coop["municipal_budgets"] or {}
    for mun, cap in mun_budgets.items():
        member = demand["municipality_code"].astype(str) == mun
        row = {}
        for di in demand.index[member]:
            pop = float(demand.loc[di, "population"])
            for fi, _ in edge_map[di]:
                row[moved[(di, fi)]] = pop * S
        if row:
            model.constraint(row, upper=float(cap))

    declared_gap = config["solver"]["mip_relative_gap"] if mip_rel_gap is None else mip_rel_gap
    result = model.solve(
        time_limit=config["solver"]["time_limit_seconds"] if time_limit is None else time_limit,
        mip_rel_gap=declared_gap,
    )
    if result.x is None:
        return VoluntarySolution(
            summary={
                "model_version": MODEL_VERSION,
                "feasible": False,
                "solver_status": int(result.status),
                "solver_message": str(result.message),
            },
            demand=pd.DataFrame(),
            facilities=pd.DataFrame(),
        )
    values = result.x

    # ---- decode ----
    records = []
    edge_rows = []
    offers_rows = []
    total_pop = float(demand["population"].sum())
    moved_pop = 0.0
    subsidy_outlay = 0.0
    stayer_mfg_fail = 0.0
    for di, node in demand.iterrows():
        pop = float(node["population"])
        stay_quality = 0.0
        moved_share = 0.0
        best_offer = None
        for fi, dkm in edge_map[di]:
            dv = unit_value(values[direct[(di, fi)]])
            ov = unit_value(values[offer[(di, fi)]])
            mv = unit_value(values[moved[(di, fi)]])
            minutes = dkm * circuity / speed * 60
            stay_quality += dv * min(1.0, threshold_minutes / max(threshold_minutes, minutes))
            moved_share += mv
            moved_pop += pop * mv
            subsidy_outlay += pop * mv * move_price.get((di, fi), S)
            if ov > 0.5 or mv > 1e-6:
                best_offer = (fi, ov, mv)
                offers_rows.append(
                    {
                        "demand_id": node["demand_id"],
                        "facility_id": facilities.loc[fi, "facility_id"],
                        "offered": ov,
                        "moved_share": mv,
                        "subsidy_per_mover": move_price.get((di, fi), S),
                        "tau": float(node["tau"]),
                        "distance_km": dkm,
                        "u_move": h + S - kappa * dkm,
                        "u_stay_components": {"beta_F_stay": beta * stay_quality, "u0_tau": u0 + float(node["tau"])},
                    }
                )
        mshare = unit_value(values[mobile[di]])
        dshare = unit_value(values[digital[di]])
        sshare = unit_value(values[slack[di]])
        stay_share = 1.0 - moved_share
        stay_quality += (
            mshare * substitution["mobile_quality_effectiveness"]
            + dshare * substitution["digital_quality_effectiveness"]
        )
        conditional_stay_quality = stay_quality / stay_share if stay_share > 1e-9 else np.nan
        if np.isfinite(conditional_stay_quality) and conditional_stay_quality + 1e-9 < theta:
            stayer_mfg_fail += pop
        for fi, dkm in edge_map[di]:
            ds = unit_value(values[direct[(di, fi)]])
            if ds > 1e-9:
                edge_rows.append(
                    {
                        "demand_id": node["demand_id"],
                        "facility_id": facilities.loc[fi, "facility_id"],
                        "distance_km": dkm,
                        "direct_share": ds,
                        "moved_share": unit_value(values[moved[(di, fi)]]),
                    }
                )
        records.append(
            {
                "demand_id": node["demand_id"],
                "municipality_code": str(node["municipality_code"]),
                "population": pop,
                "tau": float(node["tau"]),
                "stay_share": stay_share,
                "moved_share": moved_share,
                "mobile_share": mshare,
                "digital_share": dshare,
                "mfg_slack": sshare,
                "stayer_conditional_quality": conditional_stay_quality,
                "stayer_mfg_met": bool(
                    not np.isfinite(conditional_stay_quality)
                    or conditional_stay_quality + 1e-9 >= theta
                ),
            }
        )

    fac_rows = []
    for fi in candidates:
        load = 0.0
        for di, node in demand.iterrows():
            pop = float(node["population"])
            if (di, fi) in direct:
                load += pop * unit_value(values[direct[(di, fi)]])
            if (di, fi) in moved:
                load += pop * unit_value(values[moved[(di, fi)]])
        cap = facility_capacity(facilities.loc[fi], config)
        fac_rows.append(
            {
                "facility_id": facilities.loc[fi, "facility_id"],
                "facility_class": facilities.loc[fi, "facility_class"],
                "municipality_code": str(facilities.loc[fi, "municipality_code"]),
                "is_existing": bool(facilities.loc[fi, "is_existing"]),
                "open": unit_value(values[y[fi]]),
                "capacity": cap,
                "load": load,
            }
        )

    summary = {
        "model_version": MODEL_VERSION,
        "feasible": bool(result.success),
        "solver_status": int(result.status),
        "solver_message": str(result.message),
        "solver_mip_gap": float(result.get("mip_gap", np.nan)),
        "E_pub": float(result.fun),
        "subsidy_outlay": subsidy_outlay,
        "moved_population": moved_pop,
        "moved_share_population": moved_pop / total_pop if total_pop else 0.0,
        "offers_made": len(offers_rows),
        "stayer_mfg_fail_population": stayer_mfg_fail,
        "slack_population": float(
            sum(
                unit_value(values[slack[i]]) * float(demand.loc[i, "population"])
                for i in demand.index
            )
        ),
        "n_open_facilities": sum(1 for fi in candidates if values[y[fi]] > 0.5),
        "agreements_active": sum(
            1 for z in agreement.values() if values[z] > 0.5
        ),
        "agreement_pairs": [
            pair for pair, z in agreement.items() if values[z] > 0.5
        ],
        "cooperation": coop,
        "behaviour": beh,
    }
    return VoluntarySolution(
        summary=summary,
        demand=pd.DataFrame(records),
        facilities=pd.DataFrame(fac_rows),
        offers=pd.DataFrame(offers_rows),
        edges=pd.DataFrame(edge_rows),
    )


def enumerate_voluntary_bilevel(
    demand_input: pd.DataFrame,
    facilities_input: pd.DataFrame,
    config: dict,
    behaviour: dict | None = None,
) -> dict:
    """Brute-force bilevel solver for toy instances.

    Enumerates every (facility subset x offers x staying-mode shares on a
    coarse grid); households respond deterministically per the documented
    choice rule; returns the best feasible plan and its E_pub. Used only to
    verify the MILP reformulation on small instances.
    """
    from itertools import combinations, product

    beh = dict(DEFAULT_BEHAVIOUR)
    if behaviour:
        beh.update(behaviour)
    demand = demand_input.reset_index(drop=True)
    facilities = facilities_input.reset_index(drop=True)
    cand = facilities.index[facilities["is_candidate"]].tolist()
    costs = config["costs"]
    substitution = config["substitution"]
    theta = config["mfg"]["minimum_effective_provision"]
    threshold_minutes = config["mfg"]["healthcare_max_minutes"]
    circuity = config["geography"]["assumed_road_circuity"]
    speed = config["geography"]["assumed_average_speed_kmh"]

    S, delta = beh["subsidy_package_S"], beh["tie_margin_delta"]
    kappa, beta, u0, h, rho = (
        beh["move_burden_kappa"],
        beh["access_utility_beta"],
        beh["baseline_utility_u0"],
        beh["destination_value_h"],
        beh["participation_rho"],
    )
    budget = float(beh["public_budget"])
    if "tau" not in demand.columns:
        demand["tau"] = float(beh["attachment_tau"])

    def dist(di, fi):
        return float(np.hypot(facilities.loc[fi, "x"] - demand.loc[di, "x"],
                              facilities.loc[fi, "y"] - demand.loc[di, "y"]) / 1000)

    # candidate edges: all facilities within regional_limit nearest per cluster
    edge_map = {}
    for di in demand.index:
        ds = sorted(((fi, dist(di, fi)) for fi in cand), key=lambda t: t[1])
        edge_map[di] = ds[: config["candidate_edges"]["regional_limit"]]

    mobile_cap = substitution["healthcare_mobile_max_share"]
    digital_cap = substitution["healthcare_digital_max_share"]
    comb_cap = substitution["healthcare_combined_max_share"]
    mq, dq = substitution["mobile_quality_effectiveness"], substitution["digital_quality_effectiveness"]

    best = None
    n_eval = 0
    for k in range(len(cand) + 1):
        for open_set in combinations(cand, k):
            open_set = set(open_set)
            fixed_cost = sum(
                costs["fixed_facility_open"]
                + facility_capacity(facilities.loc[fi], config) * costs["fixed_facility_capacity_unit"]
                for fi in open_set
            )
            # offer choices: for each cluster, no offer or one offered edge
            offer_choices = []
            for di in demand.index:
                opts = [None] + [(di, fi, dkm) for fi, dkm in edge_map[di] if fi in open_set]
                offer_choices.append(opts)
            for offer_tuple in product(*offer_choices):
                offers = {di: opt for di, opt in zip(demand.index, offer_tuple) if opt}
                # household response: accept iff condition holds; then move rho
                moves = {}
                for di, opt in offers.items():
                    _, fi, dkm = opt
                    moves[di] = (fi, dkm)
                # compute stay shares -> F_stay needed for accept condition;
                # solve small LP analytically: given moves, allocate staying
                # share to cheapest open facility to satisfy MFG if possible.
                # Then check acceptance uses achieved F_stay.
                # iterate: compute household response under achieved F_stay
                sub_used: dict[int, bool] = {}
                for _ in range(3):
                    F_stay = {}
                    feasible_i = {}
                    for di in demand.index:
                        pop = float(demand.loc[di, "population"])
                        stay_share = 1.0 - rho if di in moves else 1.0
                        best_q = 0.0
                        for fi, dkm in edge_map[di]:
                            if fi in open_set:
                                minutes = dkm * circuity / speed * 60
                                best_q = max(best_q, min(1.0, threshold_minutes / max(threshold_minutes, minutes)))
                        # staying share T is split between direct (at best
                        # quality) and substitution shares (which DISPLACE
                        # direct share: x + m + d = T, m+d <= comb_cap).
                        # F(T) = max over t in [0, comb_cap] of
                        #   (T - t)*best_q + t*eff   with eff = best mode mix.
                        q_need = theta * stay_share
                        eff = max(mq, dq) if (mobile_cap > 0 or digital_cap > 0) else 0.0
                        t_max = min(comb_cap, mobile_cap + digital_cap, stay_share)
                        if eff > best_q:
                            t_star = min(t_max, max(0.0, (q_need - best_q * stay_share) / (eff - best_q)) if q_need > best_q * stay_share else 0.0)
                        else:
                            t_star = 0.0
                        F = (stay_share - t_star) * best_q + t_star * eff
                        feasible = F + 1e-9 >= q_need
                        if not feasible and eff > best_q:
                            # even maximal substitution can't reach the floor
                            t_star = t_max
                            F = (stay_share - t_star) * best_q + t_star * eff
                        F_stay[di] = F
                        feasible_i[di] = feasible
                        sub_used[di] = t_star > 1e-9
                    # recompute moves under F_stay
                    new_moves = {}
                    for di, opt in offers.items():
                        _, fi, dkm = opt
                        u_move = h + S - kappa * dkm
                        u_stay = beta * F_stay[di] + u0 + float(demand.loc[di, "tau"])
                        if u_move >= u_stay + delta:
                            new_moves[di] = (fi, dkm)
                    if new_moves == moves:
                        break
                    moves = new_moves
                n_eval += 1
                if not all(feasible_i.values()):
                    continue
                # capacity check: staying population may split across open
                # facilities; greedily fill best-quality facility first.
                loads = {fi: 0.0 for fi in open_set}
                cap_ok = True
                for di in demand.index:
                    pop = float(demand.loc[di, "population"])
                    stay_share = 1.0 - rho if di in moves else 1.0
                    if di in moves:
                        loads[moves[di][0]] += pop * rho
                    ranked = sorted(
                        (
                            (
                                fi,
                                min(1.0, threshold_minutes / max(1e-9, dist(di, fi) * circuity / speed * 60)),
                            )
                            for fi, _ in edge_map[di]
                            if fi in open_set
                        ),
                        key=lambda t: -t[1],
                    )
                    remaining = pop * stay_share
                    for fi, _q in ranked:
                        room = facility_capacity(facilities.loc[fi], config) - loads[fi]
                        take = min(room, remaining)
                        loads[fi] += take
                        remaining -= take
                        if remaining <= 1e-9:
                            break
                    if remaining > 1e-6:
                        cap_ok = False
                for fi, ld in loads.items():
                    if ld > facility_capacity(facilities.loc[fi], config) + 1e-6:
                        cap_ok = False
                if not cap_ok:
                    continue
                subsidy_out = sum(
                    float(demand.loc[di, "population"]) * rho * S for di in moves
                )
                mobile_used = any(sub_used.values())
                service_cost = 0.0
                if mobile_used:
                    service_cost += costs["mobile_hub_open"]
                E_pub = fixed_cost + subsidy_out + service_cost
                if E_pub > budget + 1e-9:
                    continue
                if best is None or E_pub < best["E_pub"]:
                    best = {
                        "E_pub": E_pub,
                        "open_set": sorted(open_set),
                        "offers": {int(k): v for k, v in offers.items() if k in moves},
                        "moves": {int(k): v for k, v in moves.items()},
                    }
    return {"feasible": best is not None, "best": best, "n_evaluated": n_eval}
