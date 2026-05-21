r"""eCV: enhanced coefficient-of-variation reproducibility comparator.

A rank-based multi-replicate reproducibility statistic in the spirit of
\citet{Iakovidis2024eCV} (Gonzalez-Reymundez et al., 2024, "eCV: Enhanced
Coefficient of Variation and IDR Extensions"). The original method is
distributed as the R package `eCV` (CRAN); the implementation here
captures the statistic's defining behaviour --- a small per-feature
dispersion across replicate ranks signals reproducibility --- but does
not reproduce the full mixture-fit step the upstream package uses for
FDR calibration. The output is therefore a sortable score, not a
calibrated posterior probability, and is reported on metrics that are
monotone-invariant (ROC, PR) in addition to Sun--Cai's step-up rule.

Statistic
---------

For each feature $i$ and replicate $j \in \{1, \ldots, K\}$, compute the
*ascending* rank $r_{ij} \in \{1, \ldots, n\}$ of the score $X_{ij}$, so
that the most-reproducible feature has the largest rank-value (or
smallest, depending on convention --- we use *ascending* ranks so that
``argsort`` gives the canonical Sun--Cai-style ordering by reproducibility).
Scale to $[1/n, 1]$ by dividing by $n$: $u_{ij} = r_{ij} / n$.

The per-feature **enhanced coefficient of variation** is

$$
\widehat{\mathrm{eCV}}_i \;=\;
\frac{\mathrm{SD}_j(u_{ij})}{\bar u_{i\cdot} + \varepsilon},
$$

where $\bar u_{i\cdot}$ is the mean across the K replicates, $\mathrm{SD}_j$
is the unbiased sample standard deviation, and $\varepsilon = 10^{-6}$
guards the denominator against features whose ranks collapse to 0. A
feature whose replicate-ranks cluster tightly has a small eCV; a feature
whose ranks spread widely has a large eCV. We then report

$$
\widehat{\mathrm{idr}}^{\mathrm{eCV}}_i \;=\; \frac{\mathrm{eCV}_i}{\max_i \mathrm{eCV}_i},
$$

so that the score lives in $[0, 1]$ and the canonical Sun--Cai step-up
rule applies (smaller = more reproducible). Ties are broken via
:func:`scipy.stats.rankdata` with ``method="average"``, matching the
empirical-rank pilot used elsewhere in the package.

Caveats relative to the full eCV paper
--------------------------------------

The full eCV procedure couples the statistic to an IDR-style EM step
that recovers a posterior reproducibility probability calibrated to a
nominal FDR. The implementation here intentionally skips that step,
because (a) the EM is package-internal and not re-derivable from the
preprint alone, and (b) for the package's role as a *comparator* in our
benchmark, the rank-quality of the score is the part that matters for
ROC/PR curves. Treat the output of :func:`fit_ecv` as a sortable
ranking rather than a calibrated posterior probability.

See plan 09 §"Comparators" and the open question §8 (Q1) on
comparator availability and licensing.
"""

from __future__ import annotations

import numpy as np
import scipy.stats as sps

_EPS = 1.0e-6


def fit_ecv(
    X: np.ndarray,
    *,
    tie_method: str = "average",
) -> np.ndarray:
    r"""Return the eCV-based local idr for each feature.

    Parameters
    ----------
    X
        Length-$(n, K)$ raw replicate scores. Higher = more confident.
    tie_method
        Tie-breaking rule for the per-replicate ranks. Passed to
        :func:`scipy.stats.rankdata`. Default ``"average"`` matches
        the empirical-rank pilot :func:`py_idr.marginals.transform.empirical_rank_transform`.

    Returns
    -------
    Length-$n$ array in $[0, 1]$. Smaller = more reproducible. Suitable
    as a drop-in replacement for the local idr in the Sun--Cai step-up
    rule, with the caveat that the score is a sortable ranking rather
    than a calibrated posterior probability.

    Raises
    ------
    ValueError
        If ``X`` is not 2-D or has fewer than 2 replicates.
    """
    X = np.asarray(X)
    if X.ndim != 2:
        raise ValueError(f"X must be 2-D (n, K); got shape {X.shape}")
    n, K = X.shape
    if K < 2:
        raise ValueError(f"K must be >= 2 for a multi-replicate comparator; got {K}")

    # Per-replicate ascending ranks; high score -> high rank. Scale to (0, 1].
    ranks = np.column_stack([sps.rankdata(X[:, j], method=tie_method) for j in range(K)])
    u = ranks.astype(np.float64) / float(n)

    mean_u = u.mean(axis=1)  # (n,)
    sd_u = u.std(axis=1, ddof=1)  # (n,); unbiased SD

    ecv = sd_u / (mean_u + _EPS)
    # Normalize to [0, 1]. A degenerate all-zero column (every feature
    # perfectly reproducible) maps to the all-zero idr; cap at 1.0
    # for the typical case.
    denom = float(np.max(ecv))
    if denom <= _EPS:
        return np.zeros_like(ecv)
    return ecv / denom
