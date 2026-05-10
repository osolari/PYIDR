"""Latent basis-membership indicators + conjugate Dirichlet update for Bernstein weights.

Algorithm 1 step 5 of the revised report introduces, for each $(i, j)$, a latent
indicator $S_{i,j} \\in \\{1, \\ldots, M_j\\}$ with

.. math::
   \\Pr(S_{i,j} = r \\mid \\cdot) \\propto p_{j,r}\\, b_r(F_j^{(0)}(X_{i,j}); M_j).

Given the latent indicators and counts $n_{j,r} = \\#\\{i : S_{i,j} = r\\}$, the
conditional posterior is conjugate:

.. math::
   \\bm p_j \\mid \\cdot \\sim \\mathrm{Dir}(\\alpha_F + n_{j,1}, \\ldots, \\alpha_F + n_{j,M_j}).

This module implements both steps as pure functions; the broader sweep in
:mod:`py_idr.inference.mcmc.sweep` orchestrates them.

See plans/03_marginals.md and §3.3.1 of the revised report.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Integer

from py_idr.marginals.bernstein import density_basis


def sample_latent_S(
    key: jax.Array,
    F0_X: Float[Array, "n"],
    weights: Float[Array, "M"],
    M: int,
) -> Integer[Array, "n"]:
    """Sample $S_i \\in \\{1, \\ldots, M\\}$ for each observation.

    Parameters
    ----------
    key
        JAX PRNG key.
    F0_X
        Length-$n$ vector of pilot-CDF values $F^{(0)}(X_i) \\in (0, 1)$.
    weights
        Current Bernstein-weight vector $\\bm p \\in \\Delta^{M-1}$.
    M
        Bernstein degree.

    Returns
    -------
    Length-$n$ vector of integer indicators $S_i \\in \\{1, \\ldots, M\\}$. Returned
    in 1-based indexing to match the manuscript notation.
    """
    weights = jnp.asarray(weights)
    if weights.shape[-1] != M:
        raise ValueError(f"weights length must equal M={M}; got {weights.shape}")

    # Per-observation categorical: Pr(S_i = r) ∝ p_r * b_r(F_0(X_i); M).
    basis = density_basis(F0_X, M)  # (n, M)
    log_weights = jnp.log(weights)[None, :]  # (1, M)
    log_probs = jnp.log(basis + 1e-300) + log_weights
    samples = jax.random.categorical(key, log_probs, axis=-1)
    return samples.astype(jnp.int32) + 1  # 1-based per the manuscript


def update_weights(
    key: jax.Array,
    S: Integer[Array, "n"],
    M: int,
    alpha_F: float,
) -> Float[Array, "M"]:
    """Conjugate Dirichlet draw given the latent indicators.

    Returns a sample from $\\mathrm{Dir}(\\alpha_F + n_1, \\ldots, \\alpha_F + n_M)$
    where $n_r = \\#\\{i : S_i = r\\}$.

    Parameters
    ----------
    key
        JAX PRNG key.
    S
        Length-$n$ vector of 1-based latent indicators.
    M
        Bernstein degree.
    alpha_F
        Smoothing hyperparameter. The report's prior is
        ``α_F ~ Gamma(1, 1)``; the broader sweep updates it via Metropolis.
    """
    if alpha_F <= 0.0:
        raise ValueError(f"alpha_F must be positive; got {alpha_F}")
    counts_with_zero = jnp.bincount(S, length=M + 1)  # index 0 is never used
    counts = counts_with_zero[1:]  # 1..M
    dtype = jnp.zeros(()).dtype
    concentration = alpha_F + counts.astype(dtype)
    return jax.random.dirichlet(key, concentration)
