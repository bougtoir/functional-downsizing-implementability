from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from src.common import ROOT, load_config, reporting_parameters, write_json
from src.optimization.service_model import candidate_edges, solve_service_model


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_raw_ledger() -> list[dict]:
    ledger = pd.read_csv(ROOT / "data" / "raw" / "acquisition_ledger.csv")
    records = []
    for row in ledger.itertuples():
        path = ROOT / row.storage_path
        actual_size = path.stat().st_size
        actual_hash = sha256(path)
        records.append(
            {
                "dataset_id": row.dataset_id,
                "exists": path.exists(),
                "size_matches": actual_size == row.file_size_bytes,
                "sha256_matches": actual_hash == row.sha256,
            }
        )
    return records


def results_within_tolerance(frame: pd.DataFrame, maximum_mip_gap: float) -> bool:
    mip_gaps = pd.to_numeric(frame["solver_mip_gap"], errors="coerce")
    return bool(
        (
            frame["solver_success"].astype(bool)
            & mip_gaps.notna()
            & mip_gaps.le(maximum_mip_gap + 1e-9)
        ).all()
    )


def main() -> None:
    dtype = {"municipality_code": str, "pref_code": str}
    demand = pd.read_csv(ROOT / "data" / "processed" / "demand.csv", dtype=dtype)
    facilities = pd.read_csv(ROOT / "data" / "processed" / "facilities.csv", dtype=dtype)
    config = load_config()
    results = ROOT / "outputs" / "results"
    primary_results = pd.read_csv(results / "primary_results.csv")
    sensitivity_results = pd.read_csv(results / "sensitivity_results.csv")
    refinement_results = pd.read_csv(results / "transition_refinement.csv")
    reporting = reporting_parameters(config)
    decline_values = sorted(config["experiment"]["population_decline"])
    validation_decline = decline_values[len(decline_values) // 2]

    low_rf = solve_service_model(
        demand,
        facilities,
        config,
        "D",
        reporting["minimum_rf"],
        validation_decline,
    )
    high_rf = solve_service_model(
        demand,
        facilities,
        config,
        "D",
        reporting["maximum_rf"],
        validation_decline,
    )
    cheap_substitution = solve_service_model(
        demand,
        facilities,
        config,
        "D",
        reporting["maximum_rf"],
        validation_decline,
        mobile_cost_multiplier=0.01,
        digital_cost_multiplier=0.01,
    )
    expensive_substitution = solve_service_model(
        demand,
        facilities,
        config,
        "D",
        reporting["maximum_rf"],
        validation_decline,
        mobile_cost_multiplier=100.0,
        digital_cost_multiplier=100.0,
    )
    comparison_arguments = {
        "relocation_friction": reporting["central_rf"],
        "population_decline": validation_decline,
        "time_limit": config["solver"]["retry_time_limit_seconds"],
    }
    constrained = solve_service_model(
        demand,
        facilities,
        config,
        "C",
        **comparison_arguments,
    )
    relaxed = solve_service_model(
        demand,
        facilities,
        config,
        "D",
        **comparison_arguments,
    )

    cheap_share = (
        cheap_substitution.summary["mobile_share"]
        + cheap_substitution.summary["digital_share"]
    )
    expensive_share = (
        expensive_substitution.summary["mobile_share"]
        + expensive_substitution.summary["digital_share"]
    )
    edge_config = config["candidate_edges"]
    constrained_edges = {
        (demand_index, facility_index)
        for demand_index, facility_index, _ in candidate_edges(
            demand,
            facilities,
            "C",
            edge_config["same_municipality_limit"],
            edge_config["regional_limit"],
        )
    }
    relaxed_edges = {
        (demand_index, facility_index)
        for demand_index, facility_index, _ in candidate_edges(
            demand,
            facilities,
            "D",
            edge_config["same_municipality_limit"],
            edge_config["regional_limit"],
        )
    }
    numerical_tolerance = (
        abs(constrained.summary["objective_total"])
        * config["solver"]["mip_relative_gap"]
        / (1 - config["solver"]["mip_relative_gap"])
        + 1e-5
    )
    checks = {
        "rf_zero_allows_more_relocation": (
            low_rf.summary["relocated_share"] > high_rf.summary["relocated_share"]
        ),
        "high_rf_suppresses_relocation": high_rf.summary["relocated_share"] < 0.05,
        "near_zero_substitution_cost_increases_substitution": cheap_share
        > expensive_share,
        "huge_substitution_cost_returns_to_physical_or_relocation": expensive_share
        < 0.01,
        "boundary_relaxation_preserves_all_constrained_edges": (
            constrained_edges <= relaxed_edges
        ),
        "boundary_relaxation_weakly_improves_within_numerical_tolerance": (
            relaxed.summary["objective_total"]
            <= constrained.summary["objective_total"] + numerical_tolerance
        ),
        "population_conserved": abs(
            demand["population"].sum()
            - json.loads(
                (ROOT / "data" / "processed" / "preprocessing_summary.json").read_text()
            )["population_from_cells"]
        )
        < 1e-6,
        "primary_results_within_solver_tolerance": results_within_tolerance(
            primary_results,
            config["solver"]["mip_relative_gap"],
        ),
        "sensitivity_results_within_solver_tolerance": results_within_tolerance(
            sensitivity_results,
            config["solver"]["mip_relative_gap"],
        ),
        "transition_results_within_solver_tolerance": results_within_tolerance(
            refinement_results,
            config["solver"]["transition_mip_relative_gap"],
        ),
    }
    raw_checks = verify_raw_ledger()
    checks["all_raw_files_match_ledger"] = all(
        item["exists"] and item["size_matches"] and item["sha256_matches"]
        for item in raw_checks
    )
    checks["all_solvers_returned_solutions"] = all(
        solution.summary["solver_success"]
        for solution in (
            low_rf,
            high_rf,
            cheap_substitution,
            expensive_substitution,
            constrained,
            relaxed,
        )
    )
    checks["objective_components_match_solver"] = all(
        abs(solution.summary["objective_solver_difference"]) < 1e-5
        for solution in (
            low_rf,
            high_rf,
            cheap_substitution,
            expensive_substitution,
            constrained,
            relaxed,
        )
    )
    checks = {key: bool(value) for key, value in checks.items()}
    if not all(checks.values()):
        raise RuntimeError(f"Validation failed: {checks}")

    report = {
        "checks": checks,
        "raw_files": raw_checks,
        "diagnostics": {
            "relocated_share_minimum_rf": low_rf.summary["relocated_share"],
            "relocated_share_maximum_rf": high_rf.summary["relocated_share"],
            "substitution_share_cheap": cheap_share,
            "substitution_share_expensive": expensive_share,
            "constrained_objective": constrained.summary["objective_total"],
            "relaxed_objective": relaxed.summary["objective_total"],
        },
    }
    write_json(ROOT / "outputs" / "diagnostics" / "validation_report.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
