"""Lexicographic tie-break audit for the all-coalitions certificate.

For each certified grid cell of the threshold-LB certificate we re-solve
twice: (1) the free_move_at_lb optimum E*; (2) max moved population subject
to E_pub <= E* + tol. max movers = 0 => no even objective-tied plan moves
anyone; the empty-implementable-set certificate excludes ties, not only
strict improvements. We also report whether E* is strictly below the
maintain-optimum by epsilon (strict cost-improving switch test).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.common import ROOT, load_config
from src.optimization.voluntary_model import solve_voluntary_model

OUT = ROOT / "outputs" / "all_coalitions"
EPS_STRICT = 0.01
TIE_TOL = 1e-6


def main() -> None:
    config = load_config()
    demand = pd.read_csv(ROOT / "data/processed/demand.csv")
    facilities = pd.read_csv(ROOT / "data/processed/facilities.csv")
    can = json.loads((OUT / "CANONICAL_ALL_COALITIONS.json").read_text())
    e_m4 = float(can["E_pub_M4_baseline"])
    grid = pd.read_csv(OUT / "all_coalitions_certificate.csv")

    rows = []
    for _, g in grid.iterrows():
        beh = {
            "attachment_tau": float(g.attachment_tau),
            "move_burden_kappa": float(g.move_burden_kappa),
            "destination_value_h": float(g.destination_value_h),
            "free_move_at_lb": True,
        }
        first = solve_voluntary_model(
            demand, facilities, config, behaviour=beh,
            time_limit=900, mip_rel_gap=0.0)
        e_star = float(first.summary.get("E_pub", np.nan))
        beh["lexicographic_movers"] = True
        beh["epub_cap"] = e_star + TIE_TOL
        second = solve_voluntary_model(
            demand, facilities, config, behaviour=beh,
            time_limit=900, mip_rel_gap=0.0)
        max_movers = float(second.summary.get("moved_population", np.nan))
        rows.append({
            **{k: float(g[k]) for k in ("move_burden_kappa", "attachment_tau", "destination_value_h")},
            "E_star": e_star,
            "E_M4": e_m4,
            "strict_improvement_possible": bool(e_star < e_m4 - EPS_STRICT),
            "max_movers_at_optimum": max_movers,
            "ties_excluded": bool(max_movers < 1e-6),
            "feasible": bool(second.summary.get("feasible")),
        })
        print(rows[-1], flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "lexicographic_certificate.csv", index=False)
    print(json.dumps({
        "schema": "lexicographic_certificate/v1",
        "eps_strict": EPS_STRICT,
        "tie_tol": TIE_TOL,
        "all_ties_excluded": bool((df.max_movers_at_optimum < 1e-6).all()),
        "strict_improvement_cells": int(df.strict_improvement_possible.sum()),
    }, indent=1))


if __name__ == "__main__":
    main()
