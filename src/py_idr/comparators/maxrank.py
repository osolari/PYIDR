r"""MaxRank: non-parametric rank-based reproducibility comparator.

A simple baseline in the spirit of Philtron, Lyu, Li & Ghosh (2018,
"MaRR"). For each feature $i$ and each replicate $j$, let
$r_{ij}$ be the *descending* rank of the score $X_{ij}$ (so $r = 1$ is
the best score). Define the **maximum rank** across replicates,

$$R_i \;=\; \max_{j = 1, \ldots, K} r_{ij},$$

and the per-feature MaxRank local idr,

$$\widehat{\mathrm{idr}}^{\mathrm{MR}}_i \;=\; \frac{R_i}{n}.$$

The intuition: a feature is "reproducible" iff it ranks at the top of
*every* replicate, so its worst rank is small relative to $n$.
Features with a low max rank get a small local idr → most likely
reproducible; features with high max rank get a large local idr →
most likely irreproducible.

This is the simplest rank-based comparator that still discriminates
reproducible from irreproducible features. It has no parameters and
no EM step, so it runs in $O(nK \log n)$ regardless of the data. Use
it as a **null baseline** against which to measure improvement from
parametric methods (Vanilla IDR, PY-IDR).

Caveats relative to the full Philtron et al. method
---------------------------------------------------

The full MaRR paper fits a two-component non-parametric mixture model
on $R_i$ via NPMLE / EM, separating the reproducible-class CDF from
the irreproducible-class CDF (which is the discrete uniform on
$\{1, \ldots, n\}$). The implementation here skips that mixture-fit
step and uses the raw scaled max-rank statistic — sufficient for a
ranking-quality comparator (ROC / PR curves are invariant to monotone
transforms of the score), but **not equivalent to MaRR for FDR
calibration**. Treat the output as a sortable score rather than a
calibrated posterior probability.

See plan 09 §"Comparators".
"""

from __future__ import annotations

import numpy as np
import scipy.stats as sps


def fit_maxrank(
    X: np.ndarray,
    *,
    tie_method: str = "average",
) -> np.ndarray:
    r"""Return the MaxRank local idr for each feature.

    Parameters
    ----------
    X
        Length-$(n, K)$ raw replicate scores. Higher = more confident.
    tie_method
        Tie-breaking rule for the per-replicate ranks. Passed to
        :func:`scipy.stats.rankdata`. Default ``"average"`` matches
        the mid-rank convention used by the empirical-rank pilot
        :func:`py_idr.marginals.transform.empirical_rank_transform`.

    Returns
    -------
    Length-$n$ array of local-idr-like values in $[0, 1]$. Smaller =
    more reproducible.

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

    # Per-replicate descending ranks: best score → rank 1.
    # rankdata returns ascending ranks by default, so we rank the
    # negated scores. Mid-rank ties match the empirical-rank pilot.
    desc_ranks = np.column_stack([sps.rankdata(-X[:, j], method=tie_method) for j in range(K)])
    max_rank_per_feature = desc_ranks.max(axis=1)  # (n,)
    # Scale into [1/n, 1]; smaller = better.
    return max_rank_per_feature.astype(float) / float(n)
