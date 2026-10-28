from pathlib import Path

import pandas as pd
import pytest

from src.common import ROOT, load_config, reporting_parameters
from src.optimization.service_model import solve_service_model, weighted_gini


def load_processed():
    processed = ROOT / "data" / "processed"
    if not (processed / "demand.csv").exists():
        pytest.skip("Run preprocessing before integration tests")
    dtype = {"municipality_code": str, "pref_code": str}
    return (
        pd.read_csv(processed / "demand.csv", dtype=dtype),
        pd.read_csv(processed / "facilities.csv", dtype=dtype),
    )


def test_weighted_gini_zero_for_equal_values():
    assert weighted_gini(
        values=pd.Series([3.0, 3.0, 3.0]).to_numpy(),
        weights=pd.Series([1.0, 2.0, 4.0]).to_numpy(),
    ) == pytest.approx(0.0)


@pytest.mark.integration
def test_relocation_friction_suppresses_relocation():
    demand, facilities = load_processed()
    config = load_config()
    low = solve_service_model(demand, facilities, config, "D", 0.0, 0.4)
    high = solve_service_model(demand, facilities, config, "D", 16.0, 0.4)
    assert low.summary["solver_success"]
    assert high.summary["solver_success"]
    assert low.summary["relocated_share"] > high.summary["relocated_share"] + 0.5


@pytest.mark.integration
def test_cheap_substitution_increases_functional_provision():
    demand, facilities = load_processed()
    config = load_config()
    cheap = solve_service_model(
        demand,
        facilities,
        config,
        "D",
        16.0,
        0.4,
        mobile_cost_multiplier=0.01,
        digital_cost_multiplier=0.01,
    )
    expensive = solve_service_model(
        demand,
        facilities,
        config,
        "D",
        16.0,
        0.4,
        mobile_cost_multiplier=100.0,
        digital_cost_multiplier=100.0,
    )
    cheap_share = cheap.summary["mobile_share"] + cheap.summary["digital_share"]
    expensive_share = expensive.summary["mobile_share"] + expensive.summary["digital_share"]
    expensive_physical_share = (
        expensive.summary["fixed_share"] + expensive.summary["relocated_share"]
    )
    assert cheap.summary["solver_success"]
    assert expensive.summary["solver_success"]
    assert cheap_share > expensive_share + 0.05
    assert expensive_physical_share > 0.95


@pytest.mark.integration
def test_boundary_relaxation_cannot_worsen_optimum():
    demand, facilities = load_processed()
    config = load_config()
    reporting = reporting_parameters(config)
    declines = sorted(config["experiment"]["population_decline"])
    comparison_decline = declines[len(declines) // 2]
    constrained = solve_service_model(
        demand,
        facilities,
        config,
        "C",
        reporting["central_rf"],
        comparison_decline,
    )
    relaxed = solve_service_model(
        demand,
        facilities,
        config,
        "D",
        reporting["central_rf"],
        comparison_decline,
    )
    assert constrained.summary["solver_success"]
    assert relaxed.summary["solver_success"]
    tolerance = (
        abs(constrained.summary["objective_total"])
        * config["solver"]["mip_relative_gap"]
        / (1 - config["solver"]["mip_relative_gap"])
        + 1e-5
    )
    assert (
        relaxed.summary["objective_total"]
        <= constrained.summary["objective_total"] + tolerance
    )


@pytest.mark.integration
@pytest.mark.parametrize("scenario", ["A", "D"])
def test_objective_components_match_solver_objective(scenario):
    demand, facilities = load_processed()
    solution = solve_service_model(
        demand,
        facilities,
        load_config(),
        scenario,
        2.0,
        0.2,
    )
    assert solution.summary["solver_success"]
    assert abs(solution.summary["objective_solver_difference"]) < 1e-5


def test_raw_ledger_and_snapshots_exist():
    ledger = ROOT / "data" / "raw" / "acquisition_ledger.csv"
    assert ledger.exists()
    frame = pd.read_csv(ledger)
    assert set(frame["dataset_id"]) == {
        "worldpop_jpn_population",
        "geofabrik_chugoku_osm",
        "nlni_n03_31",
        "nlni_n03_32",
        "nlni_n03_33",
        "nlni_p04_31",
        "nlni_p04_32",
        "nlni_p04_33",
    }
    for path in frame["storage_path"]:
        assert (ROOT / Path(path)).stat().st_size > 0
