"""Bound-based certification classification for the all-coalitions
sufficient certificate (Proposition 2).

The mathematical sufficient condition is exact and conditional:

    L* >= E0  ==>  S3 = {}   (no voluntary coalition strictly improves)

Numerically, L* is bounded below by the solver-reported MIP dual bound
LB_R of the free-move relaxation, and E0 is the numerically established
maintain optimum UB_0. The reported maximum unexcluded improvement is

    Delta_max^reported = max(0, UB_0 - LB_R)

which is a solver-reported bound, NOT a rigorous exact-arithmetic
enclosure. Classification is a pure function of solver evidence; the
mover count of the returned incumbent is a separate diagnostic and never
enters the classification.
"""
from __future__ import annotations

import numpy as np

# Explicit numerical threshold: a reported improvement allowance at or
# below this many absolute model-cost units is classified as a numerical
# non-improvement bound (not exact mathematical equality).
NUMERICAL_CERTIFICATION_THRESHOLD = 1e-9

NUMERICAL_NON_IMPROVEMENT_BOUND = "NUMERICAL_NON_IMPROVEMENT_BOUND"
UNKNOWN = "UNKNOWN"
INVALID = "INVALID"
EXACT_CERTIFICATE = "EXACT_CERTIFICATE"


def classify_cell(
    *,
    relaxation_dual_bound,
    maintain_reference_objective: float,
    solver_status=None,
    solver_success=None,
    solver_mip_gap=np.nan,
    moved_population=np.nan,
    threshold: float = NUMERICAL_CERTIFICATION_THRESHOLD,
) -> dict:
    """Classify one certificate-grid cell.

    Returns explicit fields; `numerical_certificate_class` is one of
    NUMERICAL_NON_IMPROVEMENT_BOUND / UNKNOWN / INVALID. EXACT_CERTIFICATE
    is never emitted from floating-point solver output alone.
    """
    lb = (
        float(relaxation_dual_bound)
        if relaxation_dual_bound is not None
        else np.nan
    )
    e0 = float(maintain_reference_objective)
    status_ok = (solver_status == 0) if solver_status is not None else True
    if solver_success is False:
        status_ok = False

    if not np.isfinite(lb) or not status_ok or not np.isfinite(e0):
        allowance = np.nan
        cls = INVALID
    else:
        allowance = max(0.0, e0 - lb)
        cls = (
            NUMERICAL_NON_IMPROVEMENT_BOUND
            if allowance <= threshold
            else UNKNOWN
        )

    moved = float(moved_population) if moved_population is not None else np.nan
    return {
        "returned_zero_mover_solution": bool(
            np.isfinite(moved) and moved < 1e-6
        ),
        "relaxation_dual_bound": lb,
        "maintain_reference_objective": e0,
        "reported_max_unexcluded_improvement": allowance,
        "numerical_certificate_class": cls,
        "numerical_certification_threshold": threshold,
        "solver_status": solver_status,
        "solver_mip_gap": float(solver_mip_gap)
        if solver_mip_gap is not None
        else np.nan,
    }
