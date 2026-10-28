"""Policy-switching frontier: T_min, C_avoid, dual-ledger regime scan.

Implements docs/FRONTIER_ANALYSIS_PLAN.md. Run:
    .venv/bin/python -m src.analysis.run_frontier
Outputs: outputs/frontier/*.csv
"""
from __future__ import annotations

import copy
import itertools
import math
from pathlib import Path

import numpy as np
import pandas as pd

from ..common import load_config
from ..optimization.service_model import candidate_edges
from ..optimization.voluntary_model import (
    DEFAULT_BEHAVIOUR,
    solve_voluntary_model,
)

OUT = Path("outputs/frontier")
NO_MOVE_S = -1e6  # packages that make every offer infeasible => M4 baseline


def solve_m2(demand, facilities, config, behaviour):
    return solve_voluntary_model(demand, facilities, config, behaviour=behaviour)


def solve_m4(demand, facilities, config, behaviour):
    """Fixed-demand baseline: relocation technically impossible."""
    b = dict(behaviour)
    b["subsidy_package_S"] = NO_MOVE_S
    return solve_voluntary_model(demand, facilities, config, behaviour=b)


def f_stay_map(sol_m4, demand, facilities, config):
    """Per-cluster stay quality F in the no-relocation regime.

    In M4 every stay_share == 1, so stayer_conditional_quality == F_stay.
    """
    return sol_m4.demand.set_index("demand_id")[
        "stayer_conditional_quality"
    ].to_dict()


def t_min_table(demand, facilities, config, behaviour, f_stay):
    """Minimum acceptable package per (cluster, candidate facility) edge."""
    edges = candidate_edges(
        demand,
        facilities,
        "D",
        config["candidate_edges"]["same_municipality_limit"],
        config["candidate_edges"]["regional_limit"],
    )
    beh = {**DEFAULT_BEHAVIOUR, **behaviour}
    rows = []
    dist = {}
    for di, fi, dkm in edges:
        did = demand.loc[di, "demand_id"]
        fid = facilities.loc[fi, "facility_id"]
        dist[(did, fid)] = dkm
    for di, node in demand.iterrows():
        did = node["demand_id"]
        F = f_stay.get(did, np.nan)
        for fid, dkm in [
            (fid2, d2) for (did2, fid2), d2 in dist.items() if did2 == did
        ]:
            tau_i = float(node["tau"]) if "tau" in demand.columns else float(
                beh["attachment_tau"]
            )
            tmin = (
                beh["access_utility_beta"] * F
                + beh["baseline_utility_u0"]
                + tau_i
                + beh["tie_margin_delta"]
                - beh["destination_value_h"]
                + beh["move_burden_kappa"] * dkm
            )
            rows.append(
                {
                    "demand_id": did,
                    "facility_id": fid,
                    "distance_km": dkm,
                    "population": float(node["population"]),
                    "tau": tau_i,
                    "F_stay_M4": F,
                    "T_min": max(0.0, tmin),
                }
            )
    return pd.DataFrame(rows)


def social_ledger(sol):
    """E_pub minus transfers (subsidies cancel in social accounting)."""
    return sol.summary["E_pub"] - sol.summary["subsidy_outlay"]


def regime_row(tag, lever_vals, sol_m2, sol_m4):
    s2 = sol_m2.summary
    s4 = sol_m4.summary
    if not s2["feasible"]:
        return {
            "case": tag,
            **lever_vals,
            "regime": "INFEASIBLE",
            "feasible": False,
            "solver_message": s2.get("solver_message"),
        }
    return {
        "case": tag,
        **lever_vals,
        "regime": "V" if s2["moved_population"] > 0.5 else "R",
        "feasible": s2["feasible"],
        "mip_gap": s2.get("solver_mip_gap"),
        "E_pub_V": s2["E_pub"],
        "E_social_V": social_ledger(sol_m2),
        "subsidy_outlay": s2["subsidy_outlay"],
        "moved_population": s2["moved_population"],
        "offers_made": s2["offers_made"],
        "slack_population": s2["slack_population"],
        "stayer_mfg_fail": s2["stayer_mfg_fail_population"],
        "open_facilities_V": s2["n_open_facilities"],
        "E_pub_M4": s4.get("E_pub", np.nan),
        "open_facilities_M4": s4.get("n_open_facilities", np.nan),
        "delta_E_pub_vs_M4": s2["E_pub"] - s4.get("E_pub", np.nan),
        "delta_social_vs_M4": social_ledger(sol_m2)
        - (s4["E_pub"] - s4.get("subsidy_outlay", 0.0)
           if "E_pub" in s4 else np.nan),
    }


def c_avoid_scan(demand, facilities, config, behaviour, tmin, k=8):
    """Paired-solve incremental avoidable cost for the k most promising
    clusters (smallest min T_min)."""
    cand = (
        tmin.groupby("demand_id")["T_min"].min().nsmallest(k).index.tolist()
    )
    sol_full = solve_m4(demand, facilities, config, behaviour)
    e_full = sol_full.summary["E_pub"]
    rows = []
    for did in cand:
        reduced = demand[demand["demand_id"] != did].copy()
        sol_red = solve_m4(reduced, facilities, config, behaviour)
        rows.append(
            {
                "demand_id": did,
                "population": float(
                    demand.loc[demand["demand_id"] == did, "population"].iloc[0]
                ),
                "min_T_min": float(
                    tmin.loc[tmin["demand_id"] == did, "T_min"].min()
                ),
                "E_pub_with": e_full,
                "E_pub_without": sol_red.summary["E_pub"],
                "C_avoid": e_full - sol_red.summary["E_pub"],
            }
        )
    return pd.DataFrame(rows)


def scale_config(config, cost_factor=None, q_shift=None, share_cap=None):
    c = copy.deepcopy(config)
    if cost_factor is not None:
        c["costs"]["fixed_facility_open"] *= cost_factor
        c["costs"]["fixed_facility_capacity_unit"] *= cost_factor
    if q_shift is not None:
        c["substitution"]["mobile_quality_effectiveness"] = min(
            1.0,
            max(0.0, c["substitution"]["mobile_quality_effectiveness"] + q_shift),
        )
        c["substitution"]["digital_quality_effectiveness"] = min(
            1.0,
            max(0.0, c["substitution"]["digital_quality_effectiveness"] + q_shift),
        )
    if share_cap is not None:
        c["substitution"]["healthcare_combined_max_share"] = share_cap
    return c


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    config = load_config()
    demand = pd.read_csv("data/processed/demand.csv")
    facilities = pd.read_csv("data/processed/facilities.csv")
    base_beh = {"attachment_tau": 0.0}

    # ---- T_min under the M4 (stayer-compatible) regime ----
    sol_m4_base = solve_m4(demand, facilities, config, base_beh)
    fmap = f_stay_map(sol_m4_base, demand, facilities, config)
    tmin = t_min_table(demand, facilities, config, base_beh, fmap)
    tmin.to_csv(OUT / "frontier_tmin.csv", index=False)

    # ---- parametric regime scan ----
    rows = []
    # lever 1: S x kappa x tau shift
    for S, kappa, tau in itertools.product(
        [0.0, 3.0, 30.0], [0.005, 0.05], [0.0, 0.5, 2.0]
    ):
        beh = {
            "attachment_tau": 0.0,
            "subsidy_package_S": S,
            "move_burden_kappa": kappa,
        }
        d = demand.copy()
        d["tau"] = tau
        sol_m2 = solve_m2(d, facilities, config, beh)
        sol_m4 = solve_m4(d, facilities, config, beh)
        rows.append(
            regime_row(
                "grid_S_kappa_tau",
                {"S": S, "kappa": kappa, "tau": tau},
                sol_m2,
                sol_m4,
            )
        )
    # lever 2: facility fixed cost factor (relocation-favourable corner)
    for cf, S in itertools.product([0.5, 2.0], [3.0, 30.0]):
        c = scale_config(config, cost_factor=cf)
        beh = {"attachment_tau": 0.0, "subsidy_package_S": S,
               "move_burden_kappa": 0.005}
        sol_m2 = solve_m2(demand, facilities, c, beh)
        sol_m4 = solve_m4(demand, facilities, c, beh)
        rows.append(
            regime_row("grid_costfac", {"cost_factor": cf, "S": S},
                       sol_m2, sol_m4)
        )
    # lever 3: substitution quality/cap (retention-favourable corner)
    for qs, cap in itertools.product([0.0, 0.15], [0.25, 0.5]):
        c = scale_config(config, q_shift=qs, share_cap=cap)
        beh = {"attachment_tau": 0.0, "subsidy_package_S": 30.0,
               "move_burden_kappa": 0.005}
        sol_m2 = solve_m2(demand, facilities, c, beh)
        sol_m4 = solve_m4(demand, facilities, c, beh)
        rows.append(
            regime_row("grid_subst", {"q_shift": qs, "share_cap": cap},
                       sol_m2, sol_m4)
        )
    # lever 4: public budget around the M4 optimum
    e_star = sol_m4_base.summary["E_pub"]
    for bfac in [1.1, 1.0, 0.8]:
        beh = {"attachment_tau": 0.0, "subsidy_package_S": 30.0,
               "move_burden_kappa": 0.005,
               "public_budget": bfac * e_star}
        sol_m2 = solve_m2(demand, facilities, config, beh)
        sol_m4 = solve_m4(demand, facilities, config, beh)
        rows.append(
            regime_row("grid_budget", {"budget_factor": bfac, "S": 30.0},
                       sol_m2, sol_m4)
        )
    grid = pd.DataFrame(rows)
    grid.to_csv(OUT / "frontier_grid.csv", index=False)

    # ---- C_avoid paired-solve scan (top-8 by min T_min) ----
    cav = c_avoid_scan(demand, facilities, config, base_beh, tmin, k=8)
    cav.to_csv(OUT / "frontier_cavoid.csv", index=False)

    # ---- provenance ----
    prov = []
    for _, r in grid.iterrows():
        prov.append({"run": r["case"], "params": str(
            {k: v for k, v in r.items() if k in
             ("S", "kappa", "tau", "cost_factor", "q_shift",
              "share_cap", "budget_factor")})})
    pd.DataFrame(prov).to_csv(OUT / "frontier_provenance.csv", index=False)
    print(grid.to_string())
    print(cav.to_string())


if __name__ == "__main__":
    main()
