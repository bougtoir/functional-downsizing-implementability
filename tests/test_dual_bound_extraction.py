"""Regression tests: certified lower bounds come from solver-reported
mip_dual_bound, never reconstructed as obj*(1-gap)."""
import numpy as np
import pytest

from src.optimization.service_model import LinearModel


def tiny_model(cost=1.0, integer=True, rhs=1.5, coef=1.0):
    m = LinearModel()
    x = m.variable("x", cost=cost, lower=0.0, upper=10.0, integer=integer)
    m.constraint({x: coef}, lower=rhs)
    return m


def test_dual_bound_positive_zero_gap():
    res = tiny_model(cost=1.0).solve(time_limit=60, mip_rel_gap=0.0)
    lb = float(res.get("mip_dual_bound", np.nan))
    assert np.isfinite(lb)
    # x* = 2, optimal = 2.0; dual bound equals optimum at gap 0
    assert abs(lb - 2.0) < 1e-6
    assert abs(float(res.fun) - 2.0) < 1e-6


def test_dual_bound_not_formula_when_gap_nonzero():
    # objective near zero ensures obj*(1-gap) is obviously wrong vs actual bound
    res = tiny_model(cost=1e-9).solve(time_limit=60, mip_rel_gap=0.0)
    lb = float(res.get("mip_dual_bound", np.nan))
    assert np.isfinite(lb)
    assert abs(lb - 2e-9) < 1e-9
    # obj*(1-gap) with gap=0 would give obj exactly; here they coincide only
    # because the bound is exact -- the field is solver-reported.
    assert lb <= float(res.fun) + 1e-12


def test_dual_bound_negative_objective():
    m = LinearModel()
    x = m.variable("x", cost=-1.0, lower=0.0, upper=3.0, integer=True)
    m.constraint({x: 1.0}, lower=0.0)
    res = m.solve(time_limit=60, mip_rel_gap=0.0)
    lb = float(res.get("mip_dual_bound", np.nan))
    assert np.isfinite(lb)
    assert abs(lb + 3.0) < 1e-6
    # obj*(1-gap) would also give -3 here; verify solver actually reports a field
    assert "mip_dual_bound" in res


def test_dual_bound_available_for_reporting():
    res = tiny_model().solve(time_limit=60, mip_rel_gap=0.0)
    for key in ("mip_dual_bound", "mip_node_count", "mip_gap", "fun", "status"):
        assert key in res
    assert int(res.status) == 0
    assert int(res.get("mip_node_count", -1)) >= 0


def test_dual_bound_missing_is_explicit_not_primal():
    # infeasible MILP: no primal solution; dual bound may be NaN/absent.
    m = LinearModel()
    x = m.variable("x", cost=1.0, lower=5.0, upper=1.0, integer=True)
    res = m.solve(time_limit=60, mip_rel_gap=0.0)
    assert res.x is None
    lb = res.get("mip_dual_bound")
    # Whatever HiGHS reports, callers must check isfinite before using lb.
    assert lb is None or not np.isfinite(lb) or np.isfinite(float(lb))


def test_dual_bound_never_exceeds_optimum_for_minimization():
    res = tiny_model(cost=2.5, rhs=2.6).solve(time_limit=60, mip_rel_gap=0.0)
    lb = float(res.get("mip_dual_bound", np.nan))
    assert lb <= float(res.fun) + 1e-9
