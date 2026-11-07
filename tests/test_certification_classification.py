"""Regression tests for bound-based certificate classification.

Proves the old mover-count criterion is insufficient: certification is a
pure function of the solver-reported dual bound vs the maintain reference,
not of the returned incumbent's mover count.
"""
import numpy as np
import pytest

from src.analysis.certification import (
    classify_cell,
    NUMERICAL_NON_IMPROVEMENT_BOUND,
    UNKNOWN,
    INVALID,
    NUMERICAL_CERTIFICATION_THRESHOLD,
)
from src.optimization.service_model import LinearModel

E0 = 100.0


def milp_bound(cost, rhs, coef=1.0, upper=10.0):
    """Small deterministic MILP; optimum = ceil(rhs/coef)*cost."""
    m = LinearModel()
    x = m.variable("x", cost=cost, lower=0.0, upper=upper, integer=True)
    m.constraint({x: coef}, lower=rhs)
    return m.solve(time_limit=60, mip_rel_gap=0.0)


def test_A_zero_movers_but_no_valid_certificate():
    # incumbent has zero movers yet the bound leaves a material improvement
    r = classify_cell(
        relaxation_dual_bound=90.0,
        maintain_reference_objective=E0,
        solver_status=0,
        moved_population=0.0,
    )
    assert r["returned_zero_mover_solution"] is True
    assert r["numerical_certificate_class"] == UNKNOWN
    assert r["reported_max_unexcluded_improvement"] == 10.0


def test_B_nonzero_movers_but_valid_certificate():
    # mover-carrying optimum tied with baseline: certificate still holds
    r = classify_cell(
        relaxation_dual_bound=E0,
        maintain_reference_objective=E0,
        solver_status=0,
        moved_population=250.0,
    )
    assert r["returned_zero_mover_solution"] is False
    assert r["numerical_certificate_class"] == NUMERICAL_NON_IMPROVEMENT_BOUND
    assert r["reported_max_unexcluded_improvement"] == 0.0


def test_C_strict_improvement():
    res = milp_bound(cost=1.0, rhs=1.5)  # optimum 2.0
    lb = float(res["mip_dual_bound"])
    r = classify_cell(
        relaxation_dual_bound=lb,
        maintain_reference_objective=E0,  # baseline far above optimum
        solver_status=int(res.status),
        moved_population=0.0,
    )
    # bound certifies a 98-unit improvement: certificate must FAIL
    assert r["numerical_certificate_class"] == UNKNOWN
    assert r["reported_max_unexcluded_improvement"] == pytest.approx(98.0)


def test_D_missing_dual_bound_is_invalid_never_primal():
    r = classify_cell(
        relaxation_dual_bound=None,
        maintain_reference_objective=E0,
        solver_status=0,
        moved_population=0.0,
    )
    assert r["numerical_certificate_class"] == INVALID
    assert not np.isfinite(r["reported_max_unexcluded_improvement"])
    # NaN bound likewise
    r2 = classify_cell(
        relaxation_dual_bound=np.nan,
        maintain_reference_objective=E0,
        solver_status=0,
    )
    assert r2["numerical_certificate_class"] == INVALID


def test_E_negative_objective():
    # baseline -50, bound -50: no improvement; and bound -60 -> 10 units
    r_ok = classify_cell(
        relaxation_dual_bound=-50.0,
        maintain_reference_objective=-50.0,
        solver_status=0,
    )
    assert r_ok["numerical_certificate_class"] == NUMERICAL_NON_IMPROVEMENT_BOUND
    r_imp = classify_cell(
        relaxation_dual_bound=-60.0,
        maintain_reference_objective=-50.0,
        solver_status=0,
    )
    assert r_imp["numerical_certificate_class"] == UNKNOWN
    assert r_imp["reported_max_unexcluded_improvement"] == 10.0


def test_F_numerical_near_equality_deterministic():
    eps = NUMERICAL_CERTIFICATION_THRESHOLD
    below = classify_cell(
        relaxation_dual_bound=E0 - 0.5 * eps,
        maintain_reference_objective=E0,
        solver_status=0,
    )
    above = classify_cell(
        relaxation_dual_bound=E0 - 2.0 * eps,
        maintain_reference_objective=E0,
        solver_status=0,
    )
    assert below["numerical_certificate_class"] == NUMERICAL_NON_IMPROVEMENT_BOUND
    assert above["numerical_certificate_class"] == UNKNOWN
    # bound ABOVE the reference is not a spurious improvement
    over = classify_cell(
        relaxation_dual_bound=E0 + 1e-12,
        maintain_reference_objective=E0,
        solver_status=0,
    )
    assert over["reported_max_unexcluded_improvement"] == 0.0
    assert over["numerical_certificate_class"] == NUMERICAL_NON_IMPROVEMENT_BOUND


def test_bad_status_is_invalid():
    r = classify_cell(
        relaxation_dual_bound=E0,
        maintain_reference_objective=E0,
        solver_status=2,  # e.g. infeasible per HiGHS
        moved_population=0.0,
    )
    assert r["numerical_certificate_class"] == INVALID
