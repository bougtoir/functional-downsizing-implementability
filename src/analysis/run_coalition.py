"""Coalition relocation gate: toy counterexamples + real-geography screen.

Run: .venv/bin/python -m src.analysis.run_coalition
Outputs: outputs/coalition/*.csv, canonical registry JSON.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..common import load_config
from ..optimization.service_model import facility_capacity
from ..optimization.voluntary_model import solve_voluntary_model

OUT = Path("outputs/coalition")
NO_MOVE_S = -1e6


def toy_coalition(case: str):
    """5 small clusters sharing one existing clinic; a distant existing
    clinic with spare capacity is the candidate destination."""
    k = 5
    demand = pd.DataFrame(
        [
            {
                "demand_id": f"A{n}",
                "municipality_code": "M1",
                "pref_code": "31",
                "population": 20.0,
                "x": 200.0 * n,
                "y": 0.0,
                "tau": 0.0,
            }
            for n in range(k)
        ]
        + [
            {
                "demand_id": "D",
                "municipality_code": "M2",
                "pref_code": "32",
                "population": 500.0,
                "x": 40000.0,
                "y": 0.0,
                "tau": 0.0,
            }
        ]
    )
    facilities = pd.DataFrame(
        [
            {"facility_id": "F_s", "facility_class": "clinic", "beds": 0,
             "municipality_code": "M1", "x": 400.0, "y": 0.0,
             "is_candidate": True, "is_existing": True},
            {"facility_id": "F_d", "facility_class": "clinic", "beds": 50,
             "municipality_code": "M2", "x": 40000.0, "y": 0.0,
             "is_candidate": True, "is_existing": True},
        ]
    )
    if case == "refused":
        demand.loc[demand["demand_id"].str.startswith("A"), "tau"] = 100.0
    return demand, facilities


def beh_for(case: str) -> dict:
    b = {"move_burden_kappa": 0.005, "attachment_tau": 0.0}
    if case == "accepted_preferred":
        # S just above T_min (~0.2001) so outlay 21 < freed ~23.64
        b.update(destination_value_h=0.8, subsidy_package_S=0.21)
    elif case == "accepted_not_preferred":
        b.update(destination_value_h=0.5, subsidy_package_S=0.6)
    elif case == "refused":
        b.update(destination_value_h=0.8, subsidy_package_S=0.25)
    return b


def single_pair_costs(demand, facilities, config, beh):
    """Paired M4 solves for C_avoid({i}) and C_avoid(all A)."""
    a_ids = [d for d in demand["demand_id"] if d.startswith("A")]
    def epub(dd):
        b = dict(beh); b["subsidy_package_S"] = NO_MOVE_S
        return solve_voluntary_model(dd, facilities, config, behaviour=b).summary["E_pub"]
    full = epub(demand)
    singles = {i: full - epub(demand[demand["demand_id"] != i]) for i in a_ids}
    all_a = full - epub(demand[~demand["demand_id"].isin(a_ids)])
    return full, singles, all_a


def toy_battery(config):
    rows = []
    for case in ["accepted_preferred", "accepted_not_preferred", "refused"]:
        d, f = toy_coalition(case)
        beh = beh_for(case)
        full, singles, all_a = single_pair_costs(d, f, config, beh)
        for cap in [1, 5]:
            sol = solve_voluntary_model(
                d, f, config,
                behaviour={**beh, "max_total_offers": float(cap)},
            )
            s = sol.summary
            rows.append({
                "case": case, "offer_cap": cap,
                "feasible": s["feasible"], "E_pub": s["E_pub"],
                "offers_made": s["offers_made"],
                "moved_population": s["moved_population"],
                "subsidy_outlay": s["subsidy_outlay"],
                "E_pub_M4": full,
                "delta_vs_M4": s["E_pub"] - full,
                "C_avoid_singles_max": max(singles.values()),
                "C_avoid_singles_sum": sum(singles.values()),
                "C_avoid_coalition": all_a,
                "superadditive": all_a > sum(singles.values()),
                "mip_gap": s.get("solver_mip_gap"),
            })
    return pd.DataFrame(rows)


def real_screen(config, demand, facilities):
    """Facility-user and municipality coalition families, no solves needed
    for the first-pass screen; paired solve for top candidates."""
    beh = {"attachment_tau": 0.0}
    sol_m4 = solve_voluntary_model(
        demand, facilities, config,
        behaviour={**beh, "subsidy_package_S": NO_MOVE_S},
    )
    e_full = sol_m4.summary["E_pub"]
    edges = sol_m4.edges  # actual M4 direct shares
    tmin = pd.read_csv("outputs/frontier/frontier_tmin.csv")
    tmin_min = tmin.groupby("demand_id")["T_min"].min().to_dict()
    pop = demand.set_index("demand_id")["population"].to_dict()
    costs = config["costs"]
    unit_srv = (costs["mobile_service_per_resident"]
                + costs["digital_service_per_resident"])
    open_rows = sol_m4.facilities[sol_m4.facilities["open"] > 0.5]
    fc_cost = {}
    for _, fr in facilities[facilities["is_candidate"]].iterrows():
        fc_cost[fr["facility_id"]] = (
            costs["fixed_facility_open"]
            + facility_capacity(fr, config)
            * costs["fixed_facility_capacity_unit"]
        )
    # family (a): each open facility's actual user coalition
    fam_rows = []
    for _, fr in open_rows.iterrows():
        fid = fr["facility_id"]
        users = edges.loc[edges["facility_id"] == fid, "demand_id"].unique()
        # load share of f attributable to each user
        need = sum(pop[u] * tmin_min.get(u, np.inf) for u in users)
        ub = fc_cost.get(fid, 0.0) + sum(pop[u] * unit_srv for u in users)
        fam_rows.append({
            "family": "facility_users", "member": fid,
            "n_clusters": len(users),
            "transfer_LB": need, "C_avoid_UB": ub,
            "violation_possible": ub > need,
            "margin": ub - need,
        })
    # family (c): municipalities
    for mun, grp in demand.groupby(
        demand["municipality_code"].astype(str)
    ):
        ids = grp["demand_id"].tolist()
        need = sum(pop[u] * tmin_min.get(u, np.inf) for u in ids)
        freed = sum(
            fc_cost.get(fid, 0.0)
            for fid in edges.loc[
                edges["demand_id"].isin(ids), "facility_id"
            ].unique()
            if set(
                edges.loc[edges["facility_id"] == fid, "demand_id"]
            ).issubset(set(ids))
        )
        ub = freed + sum(pop[u] * unit_srv for u in ids)
        fam_rows.append({
            "family": "municipality", "member": mun,
            "n_clusters": len(ids),
            "transfer_LB": need, "C_avoid_UB": ub,
            "violation_possible": ub > need,
            "margin": ub - need,
        })
    fam = pd.DataFrame(fam_rows)
    return fam, sol_m4


def paired_check(config, demand, facilities, sol_m4, fam, top=6):
    """Paired re-solves for the most promising coalitions: remove the
    coalition's demand, reoptimize, compare E_pub vs transfer_LB."""
    e_full = sol_m4.summary["E_pub"]
    edges = sol_m4.edges
    tmin = pd.read_csv("outputs/frontier/frontier_tmin.csv")
    tmin_min = tmin.groupby("demand_id")["T_min"].min().to_dict()
    pop = demand.set_index("demand_id")["population"].to_dict()
    cands = fam.sort_values("margin", ascending=False).head(top)
    rows = []
    for _, r in cands.iterrows():
        if r["family"] == "facility_users":
            ids = edges.loc[
                edges["facility_id"] == r["member"], "demand_id"
            ].unique().tolist()
        else:
            ids = demand.loc[
                demand["municipality_code"].astype(str) == r["member"],
                "demand_id",
            ].tolist()
        sol = solve_voluntary_model(
            demand[~demand["demand_id"].isin(ids)].copy(),
            facilities, config,
            behaviour={"subsidy_package_S": NO_MOVE_S},
        )
        cav = e_full - sol.summary["E_pub"]
        need = sum(pop[i] * tmin_min.get(i, np.inf) for i in ids)
        rows.append({
            "family": r["family"], "member": r["member"],
            "n_clusters": len(ids), "C_avoid_exact": cav,
            "transfer_LB": need, "switch_possible": cav > need,
        })
    return pd.DataFrame(rows)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    config = load_config()
    demand = pd.read_csv("data/processed/demand.csv")
    facilities = pd.read_csv("data/processed/facilities.csv")

    toy = toy_battery(config)
    toy.to_csv(OUT / "toy_coalition.csv", index=False)
    print(toy.to_string())

    fam, sol_m4 = real_screen(config, demand, facilities)
    fam.to_csv(OUT / "coalition_family_screen.csv", index=False)
    print(fam.sort_values("margin", ascending=False).head(15).to_string())

    pairs = paired_check(config, demand, facilities, sol_m4, fam)
    pairs.to_csv(OUT / "coalition_paired_solves.csv", index=False)
    print(pairs.to_string())

    registry = {
        "toy": toy.to_dict("records"),
        "family_screen": fam.to_dict("records"),
        "paired": pairs.to_dict("records"),
    }
    (OUT / "CANONICAL_COALITION_RESULTS.json").write_text(
        json.dumps(registry, indent=2)
    )


if __name__ == "__main__":
    main()
