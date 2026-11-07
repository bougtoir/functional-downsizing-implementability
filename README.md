# Functional downsizing — implementability reproduction package

Reproduction code and canonical results for "When moving people is costly, move functions: voluntary coalitional relocation, home functions, and public-service infrastructure in a depopulating region".

Contents:
- `src/optimization/`: planner + voluntary coalitional relocation MILP models (scipy HiGHS)
- `src/analysis/`: all-coalitions certificate, lexicographic second stage, sensitivity grid, voluntary/coalition runs
- `src/preprocess/`, `src/common.py`, `src/download/`: preprocessing and public-data acquisition scripts
- `src/validation/`: audit scripts
- `src/tables/`, `src/figures/`: table/figure builders reading canonical outputs
- `outputs/`: canonical machine-readable results (all_coalitions certificate, lexicographic, toy counterexample, sensitivity grid, baselines)
- `tests/`: pytest suite incl. toy counterexample regression
- `data/processed/`: derived demand/facility/geography inputs (generated from public sources; included so canonical analyses run end-to-end)
- `data/public_inputs/`: acquisition ledger (URL, version, SHA-256, license) for public sources; raw third-party data are NOT redistributed — re-acquire via `src/download/public_data.py` per the ledger

Environment: `pip install -r requirements.txt` (Python 3.10, scipy HiGHS MILP).

Reproduce the headline numbers:
- `python -m src.analysis.run_all_coalitions` — sufficient certificate over 2^144 coalitions
- `python -m src.analysis.run_lexicographic` — stage-2 max-movers tolerance check
- `python -m pytest tests/` — toy counterexample + model regression

Numerical certification: `run_all_coalitions` reports a per-cell certified
lower bound (`certified_lb`) taken **directly from the solver-reported
`mip_dual_bound`** (HiGHS dual bound of the relaxed implementability MILP),
plus `solver_status`, `solver_mip_gap`, `solver_primal_objective` and
`solver_mip_node_count`. `certified_lb` is never reconstructed from the
primal objective and optimality gap.

Classification (`src/analysis/certification.py`) is a pure function of
solver evidence — dual bound vs the maintain reference objective — via
`reported_max_unexcluded_improvement = max(0, E0 - LB_R)` against an
explicit threshold (1e-9 absolute model-cost units). Classes:
NUMERICAL_NON_IMPROVEMENT_BOUND / UNKNOWN / INVALID. The incumbent's mover
count (`returned_zero_mover_solution`) is a separate diagnostic and never
enters the classification. `no_switch_certified` is a deprecated
bound-derived alias kept only for downstream readers. See
`outputs/all_coalitions/all_coalitions_certificate.csv`,
`tests/test_dual_bound_extraction.py` and
`tests/test_certification_classification.py`.

No LICENSE file is included: reuse rights reserved pending the author's license decision.
