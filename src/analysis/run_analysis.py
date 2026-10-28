from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import qmc

from src.common import ROOT, detail_stem, load_config, reporting_parameters, write_json
from src.optimization.service_model import (
    MODEL_VERSION,
    ModelSolution,
    solve_service_model,
)

PROCESSED = ROOT / "data" / "processed"
RESULTS = ROOT / "outputs" / "results"
CACHE = RESULTS / "cache"
DETAILS = RESULTS / "details"


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    string_columns = {"municipality_code": str, "pref_code": str}
    demand = pd.read_csv(PROCESSED / "demand.csv", dtype=string_columns)
    facilities = pd.read_csv(PROCESSED / "facilities.csv", dtype=string_columns)
    return demand, facilities


def cache_key(parameters: dict, config: dict) -> str:
    cache_config = json.loads(json.dumps(config))
    cache_config.pop("network_validation", None)
    cache_config.pop("robustness", None)
    cache_config["solver"].pop("extended_retry_multiplier", None)
    cache_config["solver"].pop("transition_mip_relative_gap", None)
    cache_config["sensitivity"].pop("parameter_decimal_places", None)
    serialized = json.dumps(
        {
            "model_version": MODEL_VERSION,
            "parameters": parameters,
            "config": cache_config,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(serialized.encode()).hexdigest()[:20]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def rounded_sensitivity_parameter(value: float, config: dict) -> float:
    decimal_places = int(config["sensitivity"]["parameter_decimal_places"])
    return round(float(value), decimal_places)


def require_solver_results(
    frame: pd.DataFrame,
    label: str,
    maximum_mip_gap: float,
) -> None:
    mip_gaps = pd.to_numeric(frame["solver_mip_gap"], errors="coerce")
    usable = (
        frame["solver_success"].astype(bool)
        & mip_gaps.notna()
        & mip_gaps.le(maximum_mip_gap + 1e-9)
    )
    if usable.all():
        return
    identifier_columns = [
        column
        for column in (
            "draw",
            "scenario",
            "relocation_friction",
            "population_decline",
            "solver_status",
            "solver_mip_gap",
        )
        if column in frame
    ]
    failures = frame.loc[~usable, identifier_columns].to_dict(orient="records")
    raise RuntimeError(f"{label} solver tolerance failure: {failures}")


def run_cached(
    demand: pd.DataFrame,
    facilities: pd.DataFrame,
    config: dict,
    parameters: dict,
    *,
    save_details: str | None = None,
) -> dict:
    key = cache_key(parameters, config)
    path = CACHE / f"{key}.json"
    cached = (
        json.loads(path.read_text(encoding="utf-8"))
        if path.exists()
        else None
    )
    details_exist = (
        save_details is None
        or (
            (DETAILS / f"{save_details}_demand.csv").exists()
            and (DETAILS / f"{save_details}_facilities.csv").exists()
        )
    )
    if cached is not None and cached["solver_success"] and details_exist:
        return cached
    solve_parameters = dict(parameters)
    if cached is not None and not cached["solver_success"]:
        solve_parameters["time_limit"] = (
            config["solver"]["retry_time_limit_seconds"]
            * config["solver"]["extended_retry_multiplier"]
        )
    solution: ModelSolution = solve_service_model(
        demand,
        facilities,
        config,
        **solve_parameters,
    )
    if (
        not solution.summary["solver_success"]
        and "time_limit" not in solve_parameters
    ):
        solve_parameters["time_limit"] = config["solver"][
            "retry_time_limit_seconds"
        ]
        solution = solve_service_model(
            demand,
            facilities,
            config,
            **solve_parameters,
        )
    if not solution.summary["solver_success"]:
        extended_limit = (
            config["solver"]["retry_time_limit_seconds"]
            * config["solver"]["extended_retry_multiplier"]
        )
        if solve_parameters.get("time_limit", 0.0) < extended_limit:
            solve_parameters["time_limit"] = extended_limit
            solution = solve_service_model(
                demand,
                facilities,
                config,
                **solve_parameters,
            )
    summary = dict(solution.summary)
    summary.update(
        {
            key: value
            for key, value in parameters.items()
            if key
            not in {
                "scenario",
                "relocation_friction",
                "population_decline",
            }
        }
    )
    write_json(path, summary)
    if save_details is not None:
        solution.demand.to_csv(DETAILS / f"{save_details}_demand.csv", index=False)
        solution.facilities.to_csv(DETAILS / f"{save_details}_facilities.csv", index=False)
    return summary


def primary_analysis(
    demand: pd.DataFrame, facilities: pd.DataFrame, config: dict
) -> pd.DataFrame:
    records = []
    rf_values = config["experiment"]["relocation_friction"]
    declines = config["experiment"]["population_decline"]
    scenarios = config["experiment"]["scenarios"]
    reporting = reporting_parameters(config)
    detail_rf_values = {
        reporting["minimum_rf"],
        reporting["central_rf"],
        reporting["maximum_rf"],
    }
    for decline in declines:
        status_quo = run_cached(
            demand,
            facilities,
            config,
            {
                "scenario": "A",
                "relocation_friction": 0.0,
                "population_decline": decline,
            },
            save_details=(
                detail_stem("A", 0.0, decline)
                if decline == reporting["baseline_decline"]
                else None
            ),
        )
        for rf in rf_values:
            record = dict(status_quo)
            record["relocation_friction"] = rf
            records.append(record)
        for scenario in [item for item in scenarios if item != "A"]:
            for rf in rf_values:
                detail_name = None
                if (
                    decline == reporting["baseline_decline"]
                    and rf in detail_rf_values
                ):
                    detail_name = detail_stem(scenario, rf, decline)
                record = run_cached(
                    demand,
                    facilities,
                    config,
                    {
                        "scenario": scenario,
                        "relocation_friction": rf,
                        "population_decline": decline,
                    },
                    save_details=detail_name,
                )
                records.append(record)
                print(
                    f"primary {scenario} RF={rf:g} decline={decline:.0%}: "
                    f"cost={record['cost_per_resident']:.6f}, "
                    f"move={record['relocated_share']:.3f}"
                )
    frame = pd.DataFrame(records).sort_values(
        ["population_decline", "scenario", "relocation_friction"]
    )
    frame.to_csv(RESULTS / "primary_results.csv", index=False)
    return frame


def transition_refinement(
    primary: pd.DataFrame,
    demand: pd.DataFrame,
    facilities: pd.DataFrame,
    config: dict,
) -> pd.DataFrame:
    records = []
    for decline in config["experiment"]["population_decline"]:
        for scenario in ("B", "C", "D"):
            subset = primary[
                (primary["population_decline"] == decline)
                & (primary["scenario"] == scenario)
            ].sort_values("relocation_friction")
            change = subset["relocated_share"].diff().abs()
            if change.iloc[1:].max() <= 1e-4:
                continue
            index = change.idxmax()
            position = subset.index.get_loc(index)
            low = float(subset.iloc[position - 1]["relocation_friction"])
            high = float(subset.iloc[position]["relocation_friction"])
            for fraction in (0.25, 0.5, 0.75):
                rf = low + (high - low) * fraction
                record = run_cached(
                    demand,
                    facilities,
                    config,
                    {
                        "scenario": scenario,
                        "relocation_friction": rf,
                        "population_decline": decline,
                        "mip_rel_gap": config["solver"][
                            "transition_mip_relative_gap"
                        ],
                    },
                )
                records.append(record)
                print(
                    f"refine {scenario} decline={decline:.0%} RF={rf:g}: "
                    f"move={record['relocated_share']:.3f}"
                )
    refined = pd.DataFrame(records)
    refined.to_csv(RESULTS / "transition_refinement.csv", index=False)
    return refined


def sensitivity_analysis(
    demand: pd.DataFrame, facilities: pd.DataFrame, config: dict
) -> pd.DataFrame:
    draws = int(config["experiment"]["sensitivity_draws"])
    sample = qmc.LatinHypercube(d=7, seed=config["project"]["seed"]).random(draws)
    records = []
    ranges = config["sensitivity"]
    for index, draw in enumerate(sample):
        rf_bounds = ranges["relocation_friction"]
        rf = rounded_sensitivity_parameter(
            np.exp(
                np.log(rf_bounds[0])
                + draw[0] * (np.log(rf_bounds[1]) - np.log(rf_bounds[0]))
            ),
            config,
        )
        mobile_bounds = ranges["mobile_cost_multiplier"]
        mobile_multiplier = rounded_sensitivity_parameter(
            np.exp(
                np.log(mobile_bounds[0])
                + draw[1]
                * (np.log(mobile_bounds[1]) - np.log(mobile_bounds[0]))
            ),
            config,
        )
        digital_bounds = ranges["digital_cost_multiplier"]
        digital_multiplier = rounded_sensitivity_parameter(
            np.exp(
                np.log(digital_bounds[0])
                + draw[2]
                * (np.log(digital_bounds[1]) - np.log(digital_bounds[0]))
            ),
            config,
        )
        impedance_bounds = ranges["impedance_multiplier"]
        impedance = rounded_sensitivity_parameter(
            impedance_bounds[0]
            + draw[3] * (impedance_bounds[1] - impedance_bounds[0]),
            config,
        )
        capacity_bounds = ranges["capacity_multiplier"]
        capacity = rounded_sensitivity_parameter(
            capacity_bounds[0]
            + draw[4] * (capacity_bounds[1] - capacity_bounds[0]),
            config,
        )
        mfg_bounds = ranges["mfg_multiplier"]
        mfg = rounded_sensitivity_parameter(
            mfg_bounds[0] + draw[5] * (mfg_bounds[1] - mfg_bounds[0]),
            config,
        )
        decline_levels = config["experiment"]["population_decline"]
        decline = decline_levels[
            min(len(decline_levels) - 1, int(draw[6] * len(decline_levels)))
        ]
        parameters = {
            "scenario": "D",
            "relocation_friction": rf,
            "population_decline": decline,
            "mobile_cost_multiplier": mobile_multiplier,
            "digital_cost_multiplier": digital_multiplier,
            "impedance_multiplier": impedance,
            "capacity_multiplier": capacity,
            "mfg_multiplier": mfg,
        }
        record = run_cached(demand, facilities, config, parameters)
        record["draw"] = index
        records.append(record)
        print(
            f"sensitivity {index + 1}/{draws}: RF={rf:.3f}, "
            f"move={record['relocated_share']:.3f}, "
            f"substitution={record['mobile_share'] + record['digital_share']:.3f}"
        )
    frame = pd.DataFrame(records).sort_values("draw")
    frame.to_csv(RESULTS / "sensitivity_results.csv", index=False)
    return frame


def main() -> None:
    CACHE.mkdir(parents=True, exist_ok=True)
    DETAILS.mkdir(parents=True, exist_ok=True)
    demand, facilities = load_inputs()
    config = load_config()
    primary = primary_analysis(demand, facilities, config)
    require_solver_results(
        primary,
        "Primary analysis",
        config["solver"]["mip_relative_gap"],
    )
    refinement = transition_refinement(primary, demand, facilities, config)
    require_solver_results(
        refinement,
        "Transition refinement",
        config["solver"]["transition_mip_relative_gap"],
    )
    sensitivity = sensitivity_analysis(demand, facilities, config)
    require_solver_results(
        sensitivity,
        "Sensitivity analysis",
        config["solver"]["mip_relative_gap"],
    )
    result_files = (
        RESULTS / "primary_results.csv",
        RESULTS / "transition_refinement.csv",
        RESULTS / "sensitivity_results.csv",
    )
    write_json(
        RESULTS / "analysis_manifest.json",
        {
            "model_version": MODEL_VERSION,
            "primary_runs": int(len(primary)),
            "transition_refinement_runs": int(len(refinement)),
            "sensitivity_runs": int(len(sensitivity)),
            "demand_nodes": int(len(demand)),
            "candidate_assets": int(facilities["is_candidate"].sum()),
            "seed": config["project"]["seed"],
            "config_sha256": hashlib.sha256(
                json.dumps(config, sort_keys=True).encode()
            ).hexdigest(),
            "result_sha256": {
                str(path.relative_to(ROOT)): sha256_file(path)
                for path in result_files
            },
        },
    )


if __name__ == "__main__":
    main()
