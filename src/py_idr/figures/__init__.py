"""Figure reproducers for the report figures.

Each submodule reproduces one figure from ``docs/report/figures/``:

- :mod:`py_idr.figures.f1_calibration` — Figure 1, realised FDR vs
  nominal $\\alpha$ calibration across regimes S1-S5. Consumes a
  long-format sweep CSV (see :mod:`py_idr.simulation.sweep`).

Future reproducers (planned, plans/09):

- F2 — PY vs DP cluster occupancy (cluster-count posterior).
- F3 — Posterior contraction in $n$ and $K$.
- F4 — Bernstein-Dirichlet idr calibration.
- F5 — Real-data ROC / PR (after the ENCODE pipeline lands, plan 08).
- F6 — Runtime scaling.

All reproducers split a pure data helper (testable without matplotlib)
from the plotting wrapper, so the figure pipeline can be exercised end
to end in CI without rendering PDFs.
"""

from py_idr.figures.f1_calibration import (
    CalibrationCurve,
    compute_calibration_curves,
    make_calibration_figure,
)
from py_idr.figures.f2_cluster_count import (
    TClusterCountHistogram,
    compute_T_histograms,
    make_cluster_count_figure,
)
from py_idr.figures.f3_runtime import (
    RuntimeCurve,
    compute_runtime_curves,
    make_runtime_figure,
)
from py_idr.figures.f4_roc_pr import (
    ROCPRSummary,
    compute_roc_pr_curves,
    make_roc_pr_figure,
)

__all__ = [
    "CalibrationCurve",
    "ROCPRSummary",
    "RuntimeCurve",
    "TClusterCountHistogram",
    "compute_T_histograms",
    "compute_calibration_curves",
    "compute_roc_pr_curves",
    "compute_runtime_curves",
    "make_calibration_figure",
    "make_cluster_count_figure",
    "make_roc_pr_figure",
    "make_runtime_figure",
]
