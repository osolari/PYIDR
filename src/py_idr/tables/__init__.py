"""LaTeX table reproducers for the report.

Each submodule reproduces one table from ``docs/report/tables/``:

- :mod:`py_idr.tables.t4_sim_results` — Table 4, realised FDR / power
  at $\\alpha = 0.05$ across regimes S1-S5. Consumes a long-format
  sweep CSV (see :mod:`py_idr.simulation.sweep`). Methods that we do
  not yet have implementations for (Vanilla IDR, eCV, etc.) emit as
  "—" placeholder rows so the table structure matches the report's
  ``tab:sim-fdr`` skeleton.

Future reproducers (planned, plans/09):

- T5 — datasets.
- T6 — real-data results.
- T7 — ablations.

All reproducers follow the same two-tier shape as the figure
reproducers: a pure data helper (``compute_*``) returning a pandas
DataFrame, plus a LaTeX-emit wrapper (``make_*_table``) that renders
the tabular into a ``.tex`` file.
"""

from py_idr.tables.t4_sim_results import (
    compute_summary_table,
    format_latex_table,
    make_simulation_results_table,
)
from py_idr.tables.t7_K_effect import (
    compute_K_effect_table,
    make_K_effect_table,
)
from py_idr.tables.t7_K_effect import format_latex_table as format_K_effect_latex_table

__all__ = [
    "compute_K_effect_table",
    "compute_summary_table",
    "format_K_effect_latex_table",
    "format_latex_table",
    "make_K_effect_table",
    "make_simulation_results_table",
]
