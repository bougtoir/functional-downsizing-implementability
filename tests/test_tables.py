import pandas as pd

from src.tables.build_tables import table_5, table_6


def test_sensitivity_table_marks_stable_leading_driver():
    stability = pd.DataFrame(
        [
            {
                "outcome": "digital_share",
                "sample": "full",
                "rank": 1,
                "parameter": "digital_cost_multiplier",
                "spearman_rho": -0.9,
                "absolute_spearman_rho": 0.9,
            },
            {
                "outcome": "digital_share",
                "sample": "full",
                "rank": 2,
                "parameter": "mfg_multiplier",
                "spearman_rho": 0.2,
                "absolute_spearman_rho": 0.2,
            },
            {
                "outcome": "digital_share",
                "sample": "holdout_0",
                "rank": 1,
                "parameter": "digital_cost_multiplier",
                "spearman_rho": -0.8,
                "absolute_spearman_rho": 0.8,
            },
        ]
    )
    summary = {
        "sensitivity_rank_stability": {
            "digital_share": {"top_driver_holdout_consistency": 1.0}
        }
    }

    table = table_5(stability, summary)

    assert table["rank"].tolist() == [1, 2]
    assert table.loc[table["rank"] == 1, "leading_driver_holdout_consistency"].item() == 1
    assert pd.isna(
        table.loc[table["rank"] == 2, "leading_driver_holdout_consistency"].item()
    )


def test_capacity_table_uses_normalized_objective_label():
    capacity = pd.DataFrame(
        [
            {
                "capacity_multiplier": 1.0,
                "cost_per_resident": 0.0015,
                "facilities_open": 46,
                "relocated_share": 0.0014,
                "mobile_share": 0.0,
                "digital_share": 0.04,
                "cross_boundary_share": 0.18,
                "mfg_violation_share": 0.0,
                "maximum_capacity_utilization_open": 1.0,
                "sites_at_least_95_percent_capacity": 16,
                "solver_success": True,
                "solver_mip_gap": 0.003,
            }
        ]
    )

    table = table_6(capacity)

    assert "normalized_objective_per_resident" in table
    assert "cost_per_resident" not in table
    assert table["solver_success"].item()
