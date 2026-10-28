"""All-coalitions implementability certificate (single-solve, class B).

Idea: let the planner move any share of any cluster to any candidate edge at
the LOWER BOUND of the voluntary-acceptance threshold,

    S^LB(i,j) = u0 + tau_i + delta - h + kappa * d_ij

(the true minimum acceptable transfer is T_min(i,j) = S^LB + beta*F_stay >=
S^LB because F_stay >= 0). Consent constraints are dropped entirely — the most
favourable possible regime for relocation. If the optimal plan under this
move-favouring pricing still relocates nobody, then under the real
(higher-or-equal) thresholds and real consent constraints NO coalition
S ⊆ I, no destination assignment and no facility re-optimization can reduce
E_pub: the implementable set of voluntary-relocation switches is empty.

This upgrades the family-restricted class-C screen to a class-B certificate
covering all 2^|I| coalitions, subject to the declared model class and the
solver's certified optimality bound.

Run: python -m src.analysis.run_all_coalitions
Outputs: outputs/all_coalitions/*.csv, CANONICAL_ALL_COALITIONS.json
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from ..common import load_config
from ..optimization.voluntary_model import solve_voluntary_model

OUT = Path("outputs/all_coalitions")
NO_MOVE_S = -1e6


def solve_baseline(config, demand, facilities):
    sol = solve_voluntary_model(
        demand, facilities, config,
        behaviour={"subsidy_package_S": NO_MOVE_S},
        time_limit=900, mip_rel_gap=0.0,
    )
    return sol.summary["E_pub"], sol


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    config = load_config()
    demand = pd.read_csv("data/processed/demand.csv")
    facilities = pd.read_csv("data/processed/facilities.csv")

    e_m4, sol_m4 = solve_baseline(config, demand, facilities)
    print(f"E_pub(M4 baseline, no offers) = {e_m4}")

    # Sanity: the toy counterexample MUST move under free_move_at_lb, else the
    # certificate is vacuous (mechanism silent in both directions).
    from .run_coalition import toy_coalition, beh_for

    d_toy, f_toy = toy_coalition("accepted_preferred")
    beh = {**beh_for("accepted_preferred"), "destination_value_h": 0.3}
    e_toy_m4, _ = (lambda s: (s.summary["E_pub"], s))(
        solve_voluntary_model(
            d_toy, f_toy, config,
            behaviour={**beh, "subsidy_package_S": NO_MOVE_S},
        )
    )
    toy_cert = solve_voluntary_model(
        d_toy, f_toy, config,
        behaviour={**beh, "free_move_at_lb": True},
    )
    ts = toy_cert.summary
    print(
        "toy free-move: E=%.4f vs M4 %.4f moved=%.1f gap=%.3g"
        % (ts["E_pub"], e_toy_m4, ts["moved_population"], ts.get("solver_mip_gap", np.nan))
    )

    # Real geography: baseline and sensitivity settings (kappa, tau, h).
    grid = []
    for kappa in [0.005, 0.02, 0.05]:
        for tau in [0.0, 0.5, 2.0]:
            for h in [0.3, 0.8]:
                grid.append({"move_burden_kappa": kappa,
                             "attachment_tau": tau,
                             "destination_value_h": h})
    # ensure the canonical (kappa=0.02, tau=0, h=0.3) appears first
    grid = sorted(grid, key=lambda b: (
        b["move_burden_kappa"] != 0.02 or b["attachment_tau"] != 0.0
        or b["destination_value_h"] != 0.3
    ))

    rows = []
    for b in grid:
        t0 = time.time()
        sol = solve_voluntary_model(
            demand, facilities, config,
            behaviour={**b, "free_move_at_lb": True},
            time_limit=900, mip_rel_gap=0.0,
        )
        s = sol.summary
        gap = s.get("solver_mip_gap", np.nan)
        obj = s["E_pub"] if s["feasible"] else np.nan
        # dual bound: obj*(1-gap) is HiGHS' optimality gap convention;
        # certified lower bound on the true optimum.
        lb = obj * (1.0 - gap) if np.isfinite(gap) else np.nan
        rows.append({
            **b,
            "feasible": s["feasible"],
            "solver_status": s.get("solver_status"),
            "E_pub_M4": e_m4,
            "E_pub_free_move": obj,
            "mip_gap": gap,
            "certified_lb": lb,
            "moved_population": s.get("moved_population", np.nan),
            "moved_share_population": s.get("moved_share_population", np.nan),
            "subsidy_outlay": s.get("subsidy_outlay", np.nan),
            "n_open_facilities": s.get("n_open_facilities"),
            "stayer_mfg_fail_population": s.get("stayer_mfg_fail_population"),
            "no_switch_certified": bool(
                s["feasible"] and np.isfinite(gap) and gap < 1e-9
                and s.get("moved_population", 1) < 1e-6
            ),
            "runtime_seconds": round(time.time() - t0, 3),
        })
        print(rows[-1])

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "all_coalitions_certificate.csv", index=False)

    registry = {
        "schema": "all_coalitions_certificate/v1",
        "E_pub_M4_baseline": e_m4,
        "toy_vacuity_check": {
            "E_pub_M4": e_toy_m4,
            "E_pub_free_move": ts["E_pub"],
            "moved_population": ts["moved_population"],
            "moves_detected": bool(ts["moved_population"] > 1e-6),
        },
        "grid": df.to_dict("records"),
        "verdict": (
            "CERTIFIED_NO_SWITCH_ALL_COALITIONS"
            if bool(df["no_switch_certified"].all())
            else "NOT_CERTIFIED"
        ),
    }
    (OUT / "CANONICAL_ALL_COALITIONS.json").write_text(json.dumps(registry, indent=2))
    print(json.dumps({k: v for k, v in registry.items() if k != "grid"}, indent=2))


if __name__ == "__main__":
    main()
