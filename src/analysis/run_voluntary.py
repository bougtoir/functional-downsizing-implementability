"""Matched-baseline experiments: M0 (frozen planner-RF) vs M3 (voluntary
residential choice) vs M5/M6 (cross-boundary cooperation w/o and w/ costs).

Runs (a) prespecified toy stress cases and (b) the real Chugoku inputs,
recording implementability gaps: where the planner-RF optimum prescribes
relocations that residents would refuse, and how the implementable optimum
differs in facility selection, guarantees and expenditure.

Outputs land in outputs/voluntary/.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..common import load_config
from ..optimization.service_model import solve_service_model
from ..optimization.voluntary_model import solve_voluntary_model

OUT = Path("outputs/voluntary")

RF_GRID = [0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0]


def implementability_audit(m0_solution, demand, facilities, config, behaviour):
    """Check an M0 planner-RF solution's relocations under choice constraints.

    For each demand unit where M0 relocated share r>eps to facility j, ask
    whether the move would be individually rational at the modelled subsidy:
    U_move >= U_stay + tau + delta using the *achieved* stayer quality under
    M0's facility set. Returns per-cluster audit table + summary.
    """
    beh = dict(behaviour)
    threshold_minutes = config["mfg"]["healthcare_max_minutes"]
    circuity = config["geography"]["assumed_road_circuity"]
    speed = config["geography"]["assumed_average_speed_kmh"]
    theta = config["mfg"]["minimum_effective_provision"]
    dsub = m0_solution.demand.set_index("demand_id")
    rows = []
    for _, node in demand.iterrows():
        did = node["demand_id"]
        rec = dsub.loc[did] if did in dsub.index else None
        relocated = float(rec["relocated_share"]) if rec is not None else 0.0
        stay_q = float(rec["expected_access_minutes"]) if rec is not None else np.nan
        dest = rec["dominant_facility_id"] if rec is not None else ""
        # distance to dominant destination
        if dest and relocated > 1e-6:
            fx = facilities.set_index("facility_id").loc[dest]
            dkm = float(np.hypot(fx["x"] - node["x"], fx["y"] - node["y"]) / 1000)
        else:
            dkm = np.nan
        u_move = (
            beh.get("destination_value_h", 0.3)
            + beh.get("subsidy_package_S", 0.0)
            - beh.get("move_burden_kappa", 0.02) * (dkm if np.isfinite(dkm) else 0.0)
        )
        # stayer utility proxy: access quality capped at 1 for stayers
        u_stay = (
            beh.get("access_utility_beta", 0.5) * min(1.0, theta)
            + beh.get("baseline_utility_u0", 0.3)
            + float(node.get("tau", beh.get("attachment_tau", 0.0)))
        )
        acceptable = bool(
            relocated > 1e-6
            and np.isfinite(dkm)
            and u_move >= u_stay + beh.get("tie_margin_delta", 1e-4)
        )
        rows.append(
            {
                "demand_id": did,
                "population": float(node["population"]),
                "relocated_share_m0": relocated,
                "destination": dest,
                "distance_km": dkm,
                "u_move": u_move,
                "u_stay_threshold": u_stay,
                "acceptable": acceptable,
            }
        )
    audit = pd.DataFrame(rows)
    moved_pop = audit["population"] * audit["relocated_share_m0"]
    acc_pop = audit.loc[audit["acceptable"], "population"] * audit.loc[
        audit["acceptable"], "relocated_share_m0"
    ]
    return audit, {
        "m0_relocated_population": float(moved_pop.sum()),
        "m0_relocated_acceptable": float(acc_pop.sum()),
        "share_of_m0_relocation_acceptable": float(
            acc_pop.sum() / moved_pop.sum()
        )
        if moved_pop.sum() > 0
        else np.nan,
    }


def run_matched(demand, facilities, config, behaviour, tag, rf_grid=RF_GRID):
    """Run M0-RF sweep vs voluntary model on identical inputs."""
    rows = []
    for rf in rf_grid:
        m0 = solve_service_model(
            demand, facilities, config, "D",
            relocation_friction=rf, population_decline=0.0,
        )
        audit, audit_sum = implementability_audit(
            m0, demand, facilities, config, behaviour
        )
        rows.append(
            {
                "case": tag,
                "model": "M0_plannerRF",
                "rf": rf,
                "objective": m0.summary["objective_total"],
                "feasible": True,
                "relocated_population": audit_sum["m0_relocated_population"],
                "relocated_acceptable": audit_sum["m0_relocated_acceptable"],
                "open_facilities": int(
                    (m0.facilities["open"] > 0.5).sum()
                ),
            }
        )
        audit.to_csv(OUT / f"{tag}_m0_rf{rf}_implementability.csv", index=False)
    sol = solve_voluntary_model(demand, facilities, config, behaviour=behaviour)
    rows.append(
        {
            "case": tag,
            "model": "M3_voluntary",
            "feasible": bool(sol.summary["feasible"]),
            "E_pub": sol.summary.get("E_pub", np.nan),
            "relocated_population": sol.summary.get("moved_population", np.nan),
            "offers_made": sol.summary.get("offers_made", np.nan),
            "open_facilities": sol.summary.get("n_open_facilities", np.nan),
            "slack_population": sol.summary.get("slack_population", np.nan),
            "stayer_mfg_fail": sol.summary.get("stayer_mfg_fail_population", np.nan),
        }
    )
    return rows, sol


def toy(name):
    """Prespecified toy stress instances."""
    base_demand = pd.DataFrame(
        [
            {"demand_id": "A", "municipality_code": "M1", "pref_code": "31",
             "population": 100.0, "x": 0.0, "y": 0.0},
            {"demand_id": "B", "municipality_code": "M2", "pref_code": "32",
             "population": 60.0, "x": 80000.0, "y": 0.0},
        ]
    )
    base_fac = pd.DataFrame(
        [
            {"facility_id": "F0", "facility_class": "clinic", "beds": 0,
             "municipality_code": "M1", "x": 3000.0, "y": 0.0,
             "is_candidate": True, "is_existing": True},
            {"facility_id": "F1", "facility_class": "clinic", "beds": 0,
             "municipality_code": "M2", "x": 40000.0, "y": 3000.0,
             "is_candidate": True, "is_existing": True},
        ]
    )
    if name == "all_willing":
        base_demand["tau"] = 0.0
    elif name == "none_willing":
        base_demand["tau"] = 100.0
    elif name == "hetero":
        base_demand["tau"] = [0.0, 2.0]
    return base_demand, base_fac


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    config = load_config()
    demand = pd.read_csv("data/processed/demand.csv")
    facilities = pd.read_csv("data/processed/facilities.csv")

    all_rows = []

    # ---- toy battery ----
    for case in ["all_willing", "none_willing", "hetero"]:
        d, f = toy(case)
        beh = {"subsidy_package_S": 0.3, "move_burden_kappa": 0.01,
               "destination_value_h": 0.6, "access_utility_beta": 0.5}
        rows, _ = run_matched(d, f, config, beh, f"toy_{case}",
                              rf_grid=[0.0, 1.0, 4.0])
        all_rows += rows

    # ---- real inputs ----
    for label, beh in [
        ("real_tau0", {"attachment_tau": 0.0, "subsidy_package_S": 0.3}),
        ("real_tau05", {"attachment_tau": 0.5, "subsidy_package_S": 0.3}),
        ("real_tau2", {"attachment_tau": 2.0, "subsidy_package_S": 0.3}),
    ]:
        rows, _ = run_matched(demand, facilities, config, beh, label,
                              rf_grid=[0.0, 1.0, 4.0])
        all_rows += rows

    # ---- subsidy frontier on real input (tau=0.5) ----
    for S in [0.0, 0.3, 1.0, 3.0]:
        beh = {"attachment_tau": 0.5, "subsidy_package_S": S}
        sol = solve_voluntary_model(demand, facilities, config, behaviour=beh)
        all_rows.append(
            {
                "case": "real_sweep",
                "model": "M3_voluntary",
                "S": S,
                "feasible": bool(sol.summary["feasible"]),
                "E_pub": sol.summary.get("E_pub", np.nan),
                "relocated_population": sol.summary.get(
                    "moved_population", np.nan
                ),
                "offers_made": sol.summary.get("offers_made", np.nan),
                "open_facilities": sol.summary.get(
                    "n_open_facilities", np.nan
                ),
                "slack_population": sol.summary.get(
                    "slack_population", np.nan
                ),
                "stayer_mfg_fail": sol.summary.get(
                    "stayer_mfg_fail_population", np.nan
                ),
            }
        )

    # ---- cooperation extension (Model C) on real input ----
    for setup in [0.0, 5.0, 50.0]:
        coop = {"enabled": True, "setup_cost": setup,
                "per_person_cost": 0.05}
        sol = solve_voluntary_model(
            demand, facilities, config,
            behaviour={"attachment_tau": 0.5, "subsidy_package_S": 0.3},
            cooperation=coop,
        )
        all_rows.append(
            {
                "case": "real_coop",
                "model": "M5_voluntary_coop",
                "S": 0.3,
                "setup_cost": setup,
                "feasible": bool(sol.summary["feasible"]),
                "E_pub": sol.summary.get("E_pub", np.nan),
                "relocated_population": sol.summary.get(
                    "moved_population", np.nan
                ),
                "offers_made": sol.summary.get("offers_made", np.nan),
                "open_facilities": sol.summary.get(
                    "n_open_facilities", np.nan
                ),
                "slack_population": sol.summary.get(
                    "slack_population", np.nan
                ),
                "agreements": sol.summary.get("agreements_active", np.nan),
            }
        )

    pd.DataFrame(all_rows).to_csv(OUT / "matched_baselines.csv", index=False)
    print(pd.DataFrame(all_rows).to_string())


if __name__ == "__main__":
    main()
