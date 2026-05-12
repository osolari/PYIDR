r"""EM-based warm start for the PY-IDR chain (Algorithm 1).

The Pólya-urn / NUTS sweep in :mod:`py_idr.inference.mcmc.sweep_multi`
mixes correctly from any initial $(\pi, \rho)$, but the default
non-informative start $(\pi = 0.5, \rho = 0.3)$ is far from the truth
under any realistic signal regime. On short warmup the chain can
spend its budget *moving* toward the typical set rather than
*sampling within* it. The §3.3.1 of the report mandates a Vanilla-IDR-
based warm start; this module provides it.

The procedure
-------------

Given an $(n, K)$ pilot CDF $F_0$:

1. For each of the $\binom{K}{2}$ replicate pairs $(j, k)$, fit the
   bivariate Vanilla IDR EM (Li-Brown-Huang-Bickel 2011) to obtain
   per-feature local idr $\widehat{\mathrm{idr}}^{(j,k)}_i$ and scalar
   $(\widehat\pi^{(j,k)}, \widehat\rho^{(j,k)})$.
2. Aggregate:

   $$\widehat\pi_{\mathrm{init}} = \mathrm{mean}_{(j,k)}\widehat\pi^{(j,k)},
   \qquad \widehat\rho_{\mathrm{init}} = \mathrm{mean}_{(j,k)}\widehat\rho^{(j,k)}.$$

3. Form the per-feature aggregated local idr (mean across pairs);
   threshold at 0.5 to obtain the initial class indicators $Z_i$.

The aggregation rule is **mean** (not median or max) because the EM's
scalar estimates are themselves means of per-feature posterior
responsibilities — averaging across pairs is the natural extension.

This is a strictly weaker assumption than the report's full plan-04
spec (which also runs $k$-means on the rank-transformed reproducible
features to seed multi-cluster assignments). For the T = 1 path it
suffices; for the multi-cluster path it gives the Pólya-urn step a
much better starting partition than "all reproducible features in
cluster 0".

See plans/04_inference_mcmc.md.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from jaxtyping import Array, Float

from py_idr.comparators.vanilla_idr import fit_vanilla_idr


@dataclass(frozen=True)
class EMInit:
    """Vanilla-IDR-based warm start for the PY-IDR sampler.

    Attributes
    ----------
    pi
        Initial reproducible mass, in $(0, 1)$. Mean of the pairwise
        $\\widehat\\pi^{(j,k)}$.
    rho
        Initial exchangeable Gaussian-copula correlation, in
        $(-1/(K-1), 1)$. Clamped away from the boundary for numerical
        safety. Mean of the pairwise $\\widehat\\rho^{(j,k)}$.
    Z
        Initial per-feature class indicators, $(n,)$ int. Equal to 1
        wherever the aggregated local idr is below 0.5 (i.e., where
        the EM thinks the feature is *more likely than not*
        reproducible).
    idr_per_feature
        Aggregated per-feature local idr (mean across pairs);
        $(n,)$ float in $[0, 1]$. Useful for downstream diagnostics
        (e.g., comparing PY-IDR's posterior idr against the EM init's
        scalar estimate).
    """

    pi: float
    rho: float
    Z: np.ndarray
    idr_per_feature: np.ndarray


def compute_em_init(
    F0: Float[Array, "n K"],
    *,
    pi_init: float = 0.5,
    rho_init: float = 0.5,
    em_max_iter: int = 200,
    em_tol: float = 1e-5,
    z_threshold: float = 0.5,
    pi_floor: float = 0.02,
    pi_ceiling: float = 0.98,
    rho_clamp: float = 0.95,
) -> EMInit:
    r"""Compute the EM warm start from a pilot-CDF matrix.

    Parameters
    ----------
    F0
        $(n, K)$ pilot-CDF values, in $[0, 1]$. Passing the raw scores
        $X$ also works (the EM re-ranks them internally), but the
        pilot CDF is the canonical input.
    pi_init, rho_init
        Starting values for each per-pair EM. Default $(0.5, 0.5)$
        gives the EM a balanced start.
    em_max_iter, em_tol
        EM convergence configuration. The default 200 / $10^{-5}$
        suffices for clean S1 data; tighten for harder regimes.
    z_threshold
        Per-feature initial-class cutoff on the aggregated local idr.
        Default 0.5 picks the "more likely than not reproducible"
        features.
    pi_floor, pi_ceiling
        Clamp on the aggregated $\\widehat\\pi$ so the chain doesn't
        start with $\\pi = 0$ or 1 (boundary points break the
        Beta posterior update).
    rho_clamp
        Clamp on the aggregated $\\widehat\\rho$ so the initial atom
        stays away from the $\\rho = \\pm 1$ boundary where the
        Cholesky factor becomes singular.

    Returns
    -------
    A :class:`EMInit` dataclass.

    Raises
    ------
    ValueError
        If ``F0`` is not 2-D with $K \\ge 2$, or if any of the clamps
        is non-finite / out of range.
    """
    F0 = np.asarray(F0)
    if F0.ndim != 2:
        raise ValueError(f"F0 must be 2-D (n, K); got shape {F0.shape}")
    n, K = F0.shape
    if K < 2:
        raise ValueError(f"K must be >= 2; got {K}")
    if not (0.0 < pi_floor < pi_ceiling < 1.0):
        raise ValueError(f"need 0 < pi_floor < pi_ceiling < 1; got {pi_floor}, {pi_ceiling}")
    if not (0.0 < rho_clamp < 1.0):
        raise ValueError(f"rho_clamp must be in (0, 1); got {rho_clamp}")

    # Fit Vanilla IDR on each of the binom(K, 2) pairs.
    pi_hats: list[float] = []
    rho_hats: list[float] = []
    per_pair_idr: list[np.ndarray] = []
    for j in range(K):
        for k in range(j + 1, K):
            res = fit_vanilla_idr(
                F0[:, [j, k]],
                pi_init=pi_init,
                rho_init=rho_init,
                max_iter=em_max_iter,
                tol=em_tol,
            )
            pi_hats.append(res.pi_hat)
            rho_hats.append(res.rho_hat)
            per_pair_idr.append(res.idr)

    pi_agg = float(np.clip(np.mean(pi_hats), pi_floor, pi_ceiling))
    # Lower bound on rho: must remain admissible for the exchangeable
    # correlation matrix to be PSD at K replicates.
    rho_lower = -1.0 / (K - 1) + 1e-6
    rho_agg = float(np.clip(np.mean(rho_hats), rho_lower, rho_clamp))

    idr_agg = np.stack(per_pair_idr, axis=0).mean(axis=0)  # (n,)
    Z = (idr_agg < z_threshold).astype(np.int32)

    return EMInit(pi=pi_agg, rho=rho_agg, Z=Z, idr_per_feature=idr_agg)
