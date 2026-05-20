r"""ChIP-R: rank-product reproducibility comparator.

A multi-replicate reproducibility statistic in the spirit of
\citet{Newell2021ChIPR} (Newell et al., 2021, "ChIP-R: Assembling
reproducible sets of ChIP-seq and ATAC-seq peaks from multiple
replicates"). ChIP-R's contribution is a rank-product aggregation
over $K \ge 3$ replicates with a closed-form null distribution under
the assumption of independent uniform per-replicate ranks. The full
upstream tool fits a per-feature significance value (and an
FDR-controlled rejection set) via the rank-product null; the
implementation here captures the *statistic* used to order features
but skips the null-fit step — the output is a sortable score rather
than a calibrated $p$-value.

Statistic
---------

For each feature $i$ and replicate $j$, let $r_{ij} \in \{1, \ldots, n\}$
be the *descending* rank of the score $X_{ij}$ (so $r = 1$ is the
most-confident feature on replicate $j$). The **rank product** for
feature $i$ is

$$
\widehat{\mathrm{RP}}_i \;=\; \left( \prod_{j=1}^K r_{ij} \right)^{1/K},
$$

scaled to a $[1/n, 1]$ range by dividing by $n$:

$$
\widehat{\mathrm{idr}}^{\mathrm{ChIP-R}}_i \;=\;
\frac{\widehat{\mathrm{RP}}_i}{n}.
$$

The geometric-mean form (vs. the raw product) keeps the score numerically
stable for large $K$ and lives in $[1/n, 1]$ regardless. Features that
rank consistently near the top of every replicate get a small RP →
small idr → most likely reproducible.

Caveats relative to the full ChIP-R paper
-----------------------------------------

The upstream ChIP-R tool reports a $p$-value derived from the
rank-product null distribution (closed-form for independent uniform
ranks; corrected for the discrete-rank ties via a continuity
correction) and applies a Benjamini-Hochberg FDR control. The
implementation here returns only the geometric-mean rank-product as a
sortable score and lets the downstream Sun--Cai step-up rule do the
FDR control. ROC / PR / power curves are invariant to monotone
transforms of the score, so the comparator is faithful for those
metrics; calibrated FDR control would require the upstream $p$-value
machinery.

See plan 09 §"Comparators" and `docs/data_acquisition_checklist.md`
(REAL-01 ENCODE CTCF, REAL-02 ENCODE ATAC) where ChIP-R is one of the
expected baselines.
"""

from __future__ import annotations

import numpy as np
import scipy.stats as sps
from jaxtyping import Array, Float


def fit_chipr(
    X: Float[Array, "n K"],
    *,
    tie_method: str = "average",
) -> np.ndarray:
    r"""Return the ChIP-R-style geometric-mean rank-product local idr.

    Parameters
    ----------
    X
        Length-$(n, K)$ raw replicate scores. Higher = more confident.
    tie_method
        Tie-breaking rule for the per-replicate ranks. Passed to
        :func:`scipy.stats.rankdata`. Default ``"average"`` matches
        the empirical-rank pilot used elsewhere in the package.

    Returns
    -------
    Length-$n$ array in $[1/n, 1]$. Smaller = more reproducible.
    Suitable as a drop-in local idr for the Sun--Cai step-up rule on
    monotone-invariant metrics; for calibrated FDR see the caveats in
    this module's docstring.

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

    # Per-replicate descending ranks via rankdata(-X). best score -> rank 1.
    desc_ranks = np.column_stack([sps.rankdata(-X[:, j], method=tie_method) for j in range(K)])

    # Geometric mean of per-replicate ranks. Compute in log-space to avoid
    # overflow for large n × large K.
    log_ranks = np.log(desc_ranks)
    log_rp = log_ranks.mean(axis=1)  # geometric mean
    rp = np.exp(log_rp)

    # Scale to [1/n, 1].
    return rp.astype(float) / float(n)
