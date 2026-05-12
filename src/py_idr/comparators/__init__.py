"""Baseline / competitor reproducibility methods used in the §4 benchmarks.

Each submodule implements one comparator and exposes a single
``fit(...)`` function that returns a per-feature local idr vector
$\\widehat{\\mathrm{idr}}_i \\in [0, 1]$. The interface matches what
:func:`py_idr.decision.sun_cai.step_up_threshold` and
:func:`py_idr.simulation.evaluation.evaluate_recovery` consume, so
comparator rows drop into the same long-format sweep CSV that PY-IDR
already populates.

Available comparators
---------------------

- :mod:`py_idr.comparators.vanilla_idr` — Li, Brown, Huang & Bickel
  (2011) original IDR: two-component bivariate Gaussian-copula
  mixture fit by EM. For $K \\ge 3$ we apply it pairwise across all
  $\\binom{K}{2}$ replicate pairs and aggregate by a configurable rule
  (default: average pairwise local idr).
- :mod:`py_idr.comparators.maxrank` — non-parametric rank-based
  baseline (Philtron-Lyu-Li-Ghosh 2018 MaRR-style). Per-feature
  scaled max rank across replicates; deterministic, no EM.

Planned (TODO)
--------------

- eCV (Iakovidis et al. 2024)
- nestedIDR (Ranalli et al. 2026)
- Full MaRR (Philtron, Lyu, Li, Ghosh 2018) with NPMLE mixture fit on
  the max-rank statistic (the version in :mod:`maxrank` skips that
  step and uses the raw scaled statistic).
- ChIP-R (Newell et al. 2021)
- GMCM-AD (Kasa & Rajan 2020)
- BNP-Archimedean (Pan, Nieto-Barajas & Craiu 2024)

See plan 09 (figures and tables) §"Comparators" and §4.2 of the report.
"""

from py_idr.comparators.maxrank import fit_maxrank
from py_idr.comparators.vanilla_idr import (
    VanillaIDRResult,
    fit_vanilla_idr,
    fit_vanilla_idr_pairwise,
)

__all__ = [
    "VanillaIDRResult",
    "fit_maxrank",
    "fit_vanilla_idr",
    "fit_vanilla_idr_pairwise",
]
