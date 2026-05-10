"""Pitman--Yor exchangeable partition probability function (EPPF).

Given an unordered partition of $n$ reproducible features into $T$ blocks of
sizes $(n_1, \\ldots, n_T)$, the Pitman--Yor EPPF (Pitman & Yor 1997, Thm 3.2) is

.. math::
   p(n_1, \\ldots, n_T \\mid \\alpha, \\sigma)
   = \\frac{\\prod_{j=1}^{T-1}(\\alpha + j\\sigma)}{(\\alpha + 1)_{n-1}}\\;
       \\prod_{m=1}^{T} (1 - \\sigma)_{n_m - 1},

where $(x)_k = x(x+1)\\cdots(x+k-1)$ is the rising factorial. Identifying
$(\\alpha + 1)_{n-1} = \\Gamma(\\alpha+n)/\\Gamma(\\alpha+1)$ and
$(1-\\sigma)_{n_m - 1} = \\Gamma(n_m - \\sigma)/\\Gamma(1-\\sigma)$, the
log-EPPF is the closed form

.. math::
   \\log p = \\log\\Gamma(\\alpha + 1) - \\log\\Gamma(\\alpha + n)
       + \\sum_{j=1}^{T-1}\\log(\\alpha + j\\sigma)
       + \\sum_{m=1}^{T}\\log\\Gamma(n_m - \\sigma) - T\\,\\log\\Gamma(1 - \\sigma).

At $\\sigma = 0$ the formula reduces continuously to the Dirichlet-process EPPF
$T \\log\\alpha + \\log\\Gamma(\\alpha) - \\log\\Gamma(\\alpha + n)
+ \\sum_m \\log\\Gamma(n_m)$.

This module provides the differentiable log-EPPF used by the α/σ NUTS
update; it is the single source of truth for the partition log-likelihood.

See plans/04_inference_mcmc.md §3.3.1.
"""

from __future__ import annotations

import jax.numpy as jnp
from jax.scipy.special import gammaln
from jaxtyping import Array, Float, Integer


def log_eppf(
    alpha: Float[Array, ""],
    sigma: Float[Array, ""],
    cluster_sizes: Integer[Array, "T"],
    n: int | Float[Array, ""],
) -> Float[Array, ""]:
    """Log Pitman--Yor EPPF given $(\\alpha, \\sigma, n_1, \\ldots, n_T, n)$.

    Parameters
    ----------
    alpha
        PY concentration; scalar JAX float.
    sigma
        PY discount in $[0, 1)$; scalar JAX float.
    cluster_sizes
        Strictly positive cluster sizes $(n_1, \\ldots, n_T)$. The function
        assumes the input is the *active* prefix — empty clusters must be
        dropped by the caller.
    n
        Total reproducible feature count; consistent with
        ``jnp.sum(cluster_sizes) == n``.

    Returns
    -------
    Scalar log probability of the partition under PY$(\\alpha, \\sigma)$.

    Notes
    -----
    Differentiable in $(\\alpha, \\sigma)$ — the NumPyro NUTS step relies on this.
    Numerical stability: the $\\sum \\log(\\alpha + j\\sigma)$ inner term uses
    direct float64 evaluation; for $T$ up to a few hundred clusters this is
    well-conditioned.
    """
    alpha = jnp.asarray(alpha)
    sigma = jnp.asarray(sigma)
    cluster_sizes = jnp.asarray(cluster_sizes)
    n_arr = jnp.asarray(n)
    T = cluster_sizes.shape[0]

    # log α(α+σ)...(α + (T-1)σ) — the numerator's T-1 product terms.
    j = jnp.arange(1, T)
    log_alpha_prod = jnp.sum(jnp.log(alpha + j * sigma))

    # log Γ(α+1) - log Γ(α+n) is the denominator (α+1)_{n-1} in inverse form.
    log_denom = gammaln(alpha + 1.0) - gammaln(alpha + n_arr)

    # Per-cluster Γ(n_m - σ) / Γ(1 - σ) factor.
    log_block = jnp.sum(gammaln(cluster_sizes - sigma)) - T * gammaln(1.0 - sigma)

    return log_alpha_prod + log_denom + log_block
