from __future__ import annotations

import numpy as np
import pandas as pd

from src.common import ROOT, detail_stem, load_config, reporting_parameters

PROCESSED = ROOT / "data" / "processed"
DETAILS = ROOT / "outputs" / "results" / "details"
RESULTS = ROOT / "outputs" / "results"


def central_detail_stem() -> str:
    reporting = reporting_parameters(load_config())
    return detail_stem(
        "D",
        reporting["central_rf"],
        reporting["baseline_decline"],
    )


def weighted_mean(frame: pd.DataFrame, column: str) -> float:
    return float(np.average(frame[column], weights=frame["population"]))


def density_diagnostics() -> pd.DataFrame:
    demand = pd.read_csv(
        PROCESSED / "demand.csv",
        dtype={"municipality_code": str, "pref_code": str},
    )
    outcome = pd.read_csv(
        DETAILS / f"{central_detail_stem()}_demand.csv",
        dtype={"municipality_code": str, "pref_code": str},
    )
    frame = outcome.merge(
        demand[["demand_id", "source_cells"]],
        on="demand_id",
        validate="one_to_one",
    )
    frame["populated_cell_density_proxy"] = frame["population"] / frame["source_cells"]
    frame["density_tercile"] = pd.qcut(
        frame["populated_cell_density_proxy"],
        q=3,
        labels=["low", "middle", "high"],
    )
    records = []
    for density_tercile, subset in frame.groupby("density_tercile", observed=True):
        records.append(
            {
                "density_tercile": str(density_tercile),
                "demand_clusters": len(subset),
                "population": subset["population"].sum(),
                "mean_access_minutes": weighted_mean(
                    subset,
                    "expected_access_minutes",
                ),
                "relocated_share": weighted_mean(subset, "relocated_share"),
                "mobile_share": weighted_mean(subset, "mobile_share"),
                "digital_share": weighted_mean(subset, "digital_share"),
                "cross_boundary_share": weighted_mean(
                    subset,
                    "cross_boundary_share",
                ),
                "mfg_violation_share": float(
                    subset.loc[subset["mfg_slack"] > 1e-7, "population"].sum()
                    / subset["population"].sum()
                ),
            }
        )
    return pd.DataFrame(records)


def facility_diagnostics() -> pd.DataFrame:
    frame = pd.read_csv(
        DETAILS / f"{central_detail_stem()}_facilities.csv",
        dtype={"municipality_code": str},
    )
    records = []
    for facility_class, subset in frame.groupby("facility_class"):
        open_subset = subset[subset["open"] > 0.5]
        records.append(
            {
                "facility_class": facility_class,
                "candidate_sites": len(subset),
                "open_sites": len(open_subset),
                "direct_load": open_subset["direct_load"].sum(),
                "relocated_load": open_subset["relocated_load"].sum(),
                "cross_boundary_load": open_subset["cross_boundary_load"].sum(),
                "mean_capacity_utilization_open": (
                    open_subset["capacity_utilization"].mean()
                    if len(open_subset)
                    else 0.0
                ),
                "maximum_capacity_utilization_open": (
                    open_subset["capacity_utilization"].max()
                    if len(open_subset)
                    else 0.0
                ),
            }
        )
    return pd.DataFrame(records)


def municipality_diagnostics() -> pd.DataFrame:
    frame = pd.read_csv(
        DETAILS / f"{central_detail_stem()}_demand.csv",
        dtype={"municipality_code": str, "pref_code": str},
    )
    records = []
    for municipality_code, subset in frame.groupby("municipality_code"):
        records.append(
            {
                "municipality_code": municipality_code,
                "population": subset["population"].sum(),
                "mean_access_minutes": weighted_mean(
                    subset,
                    "expected_access_minutes",
                ),
                "cross_boundary_share": weighted_mean(
                    subset,
                    "cross_boundary_share",
                ),
                "digital_share": weighted_mean(subset, "digital_share"),
                "relocated_share": weighted_mean(subset, "relocated_share"),
            }
        )
    return pd.DataFrame(records)


def main() -> None:
    density_diagnostics().to_csv(
        RESULTS / "density_stratified_results.csv",
        index=False,
    )
    facility_diagnostics().to_csv(
        RESULTS / "facility_diagnostics.csv",
        index=False,
    )
    municipality_diagnostics().to_csv(
        RESULTS / "municipality_diagnostics.csv",
        index=False,
    )


if __name__ == "__main__":
    main()
