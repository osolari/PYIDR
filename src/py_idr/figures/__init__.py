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

__all__ = [
    "CalibrationCurve",
    "compute_calibration_curves",
    "make_calibration_figure",
]
