from __future__ import annotations

import pandas as pd

from src.analysis.robustness_diagnostics import (
    capacity_nonmonotonicity_diagnostic,
    rf_equivalence_spread,
    sensitivity_stability_summary,
    transition_intervals,
)


def test_transition_intervals_rescale_nominal_rf() -> None:
    frame = pd.DataFrame(
        {
            "relocation_cost_coefficient_multiplier": [0.5, 0.5, 1.0, 1.0],
            "nominal_relocation_friction": [2.0, 4.0, 1.0, 2.0],
            "effective_relocation_friction": [1.0, 2.0, 1.0, 2.0],
            "relocated_share": [0.8, 0.1, 0.8, 0.1],
        }
    )
    result = transition_intervals(frame)
    assert result["0.5"]["nominal_rf_low"] == 2.0
    assert result["1"]["nominal_rf_low"] == 1.0
    assert result["0.5"]["effective_rf_high"] == 2.0


def test_rf_equivalence_spread_compares_equal_effective_rf() -> None:
    frame = pd.DataFrame(
        {
            "effective_relocation_friction": [1.0, 1.0],
            "relocation_cost_coefficient_multiplier": [0.5, 1.0],
            "cost_per_resident": [0.2, 0.2],
            "relocated_share": [0.3, 0.3],
            "digital_share": [0.1, 0.1],
            "cross_boundary_share": [0.4, 0.4],
        }
    )
    assert rf_equivalence_spread(frame)["cost_per_resident"] == 0.0


def test_sensitivity_stability_summary_tracks_top_driver() -> None:
    records = []
    for outcome in (
        "cost_per_resident",
        "relocated_share",
        "digital_share",
        "cross_boundary_share",
    ):
        for sample in ("full", "holdout_fold_0", "holdout_fold_1"):
            for rank, parameter in enumerate(
                (
                    "relocation_friction",
                    "population_decline",
                    "mobile_cost_multiplier",
                    "digital_cost_multiplier",
                    "impedance_multiplier",
                    "capacity_multiplier",
                    "mfg_multiplier",
                ),
                start=1,
            ):
                records.append(
                    {
                        "outcome": outcome,
                        "sample": sample,
                        "parameter": parameter,
                        "rank": rank,
                    }
                )
    result = sensitivity_stability_summary(pd.DataFrame(records), 2)
    assert result["cost_per_resident"]["top_driver"] == "relocation_friction"
    assert (
        result["cost_per_resident"]["top_driver_holdout_consistency"] == 1.0
    )


def test_capacity_nonmonotonicity_ignores_endpoint_peak() -> None:
    capacity = pd.DataFrame(
        {
            "capacity_multiplier": [0.6, 1.0, 1.4],
            "relocated_share": [0.01, 0.005, 0.001],
        }
    )
    result = capacity_nonmonotonicity_diagnostic(
        capacity,
        pd.DataFrame(),
        pd.DataFrame(),
        {},
    )
    assert result["classification"] == "no_interior_relocation_peak"
