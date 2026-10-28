import numpy as np
import pandas as pd
import pytest

import src.analysis.run_analysis as analysis
from src.analysis.run_analysis import (
    cache_key,
    require_solver_results,
    rounded_sensitivity_parameter,
)
from src.validation.run_validation import results_within_tolerance


def test_sensitivity_parameters_use_configured_precision():
    config = {"sensitivity": {"parameter_decimal_places": 12}}
    assert rounded_sensitivity_parameter(3.5976916347140437, config) == pytest.approx(
        3.597691634714
    )


def test_cache_key_ignores_nonmathematical_validation_sections():
    config = {
        "solver": {
            "extended_retry_multiplier": 3.0,
            "transition_mip_relative_gap": 0.006,
        },
        "sensitivity": {"parameter_decimal_places": 12},
        "costs": {"fixed_facility_open": 28.0},
    }
    baseline = cache_key({"scenario": "D"}, config)
    config["network_validation"] = {"snapshot": "2026-09-01"}
    config["robustness"] = {"capacity_multipliers": [0.6, 1.0, 1.4]}
    assert cache_key({"scenario": "D"}, config) == baseline


def test_solver_result_gate_rejects_failed_sensitivity_draw():
    frame = pd.DataFrame(
        [
            {
                "draw": 36,
                "solver_success": False,
                "solver_status": 1,
                "solver_mip_gap": 0.006177,
            }
        ]
    )
    with pytest.raises(RuntimeError, match="Sensitivity analysis"):
        require_solver_results(frame, "Sensitivity analysis", 0.003)


def test_solver_result_gate_accepts_declared_tolerance():
    frame = pd.DataFrame(
        [
            {
                "solver_success": True,
                "solver_status": 0,
                "solver_mip_gap": 0.002999,
            }
        ]
    )
    require_solver_results(frame, "Primary analysis", 0.003)


def test_solver_result_gate_rejects_unknown_mip_gap():
    frame = pd.DataFrame(
        [
            {
                "solver_success": True,
                "solver_status": 0,
                "solver_mip_gap": np.nan,
            }
        ]
    )
    with pytest.raises(RuntimeError, match="Primary analysis"):
        require_solver_results(frame, "Primary analysis", 0.003)
    assert not results_within_tolerance(frame, 0.003)


def test_main_stops_before_later_stages_after_primary_failure(monkeypatch):
    failed = pd.DataFrame(
        [
            {
                "solver_success": False,
                "solver_status": 1,
                "solver_mip_gap": 0.004,
            }
        ]
    )

    monkeypatch.setattr(analysis, "load_inputs", lambda: (pd.DataFrame(), pd.DataFrame()))
    monkeypatch.setattr(
        analysis,
        "load_config",
        lambda: {
            "solver": {
                "mip_relative_gap": 0.003,
                "transition_mip_relative_gap": 0.006,
            }
        },
    )
    monkeypatch.setattr(analysis, "primary_analysis", lambda *_: failed)

    def unexpected_stage(*_):
        pytest.fail("Later analysis stage ran after a primary solver failure")

    monkeypatch.setattr(analysis, "transition_refinement", unexpected_stage)
    monkeypatch.setattr(analysis, "sensitivity_analysis", unexpected_stage)

    with pytest.raises(RuntimeError, match="Primary analysis"):
        analysis.main()


def test_main_stops_before_sensitivity_after_refinement_failure(monkeypatch):
    valid = pd.DataFrame(
        [
            {
                "solver_success": True,
                "solver_status": 0,
                "solver_mip_gap": 0.002,
            }
        ]
    )
    failed = pd.DataFrame(
        [
            {
                "solver_success": False,
                "solver_status": 1,
                "solver_mip_gap": 0.007,
            }
        ]
    )

    monkeypatch.setattr(analysis, "load_inputs", lambda: (pd.DataFrame(), pd.DataFrame()))
    monkeypatch.setattr(
        analysis,
        "load_config",
        lambda: {
            "solver": {
                "mip_relative_gap": 0.003,
                "transition_mip_relative_gap": 0.006,
            }
        },
    )
    monkeypatch.setattr(analysis, "primary_analysis", lambda *_: valid)
    monkeypatch.setattr(analysis, "transition_refinement", lambda *_: failed)

    def unexpected_stage(*_):
        pytest.fail("Sensitivity analysis ran after a refinement solver failure")

    monkeypatch.setattr(analysis, "sensitivity_analysis", unexpected_stage)

    with pytest.raises(RuntimeError, match="Transition refinement"):
        analysis.main()
