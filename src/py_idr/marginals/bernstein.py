"""Bernstein basis for the per-replicate Bernstein-Dirichlet marginals.

Eq. (3.7) of the report:
    F_j(x) = Σ_r p_{j,r} B_r(F_j^{(0)}(x); M_j)
with B_r(t; M) = I_t(r, M-r+1) the regularized incomplete beta function.

The corresponding density basis is the Beta(r, M-r+1) PDF:
    b_r(t; M) = M · choose(M-1, r-1) · t^(r-1) · (1-t)^(M-r)
            = Γ(M+1) / (Γ(r) Γ(M-r+1)) · t^(r-1) · (1-t)^(M-r)

This module exposes both bases plus the corresponding CDF / PDF evaluators
that mix the basis under a probability vector ``p`` of Dirichlet weights.
The Bernstein–Dirichlet construction is canonical for densities on [0, 1]
(Petrone 1999; Petrone–Wasserman 2002).

Numerical notes
---------------
The cumulative basis uses :func:`jax.scipy.special.betainc`, which is stable
across the full $M \\in \\{5, 10, 20, 40, 80\\}$ grid and avoids the ``M!``
overflow that would dominate a naive direct evaluation.

The density basis is computed in log-space (``gammaln`` + ``log1p``) so the
$M = 80$ case stays well-conditioned in float32; the float64 default of the
test conftest is sufficient even at the boundary.

See plans/03_marginals.md.
"""

from __future__ import annotations

import jax.numpy as jnp
from jax.scipy.special import betainc, gammaln
from jaxtyping import Array, Float


def _r_indices(M: int) -> Float[Array, "M"]:
    """Return the integer index range r = 1, …, M as a float array.

    Uses JAX's current default float dtype so the module respects the
    ``jax_enable_x64`` configuration set by the caller.
    """
    dtype = jnp.zeros(()).dtype
    return jnp.arange(1, M + 1, dtype=dtype)


def cumulative_basis(t: Float[Array, "n"], M: int) -> Float[Array, "n M"]:
    """Cumulative Bernstein basis :math:`B_r(t;M) = I_t(r, M-r+1)` for $r = 1, \\ldots, M$.

    Parameters
    ----------
    t
        1-D vector of inputs in $[0, 1]$.
    M
        Bernstein degree; the report uses $M \\in \\{5, 10, 20, 40, 80\\}$.

    Returns
    -------
    Array of shape ``(len(t), M)`` whose ``[i, r-1]`` entry is $I_{t_i}(r, M-r+1)$.

    Notes
    -----
    Each $B_r(\\cdot; M)$ is the CDF of a $\\mathrm{Beta}(r, M-r+1)$ random variable.
    They are non-decreasing in $t$ and span the whole [0, 1] image as $r$ varies
    from 1 to $M$, satisfying $B_1(t; M) \\to 1$ and $B_M(t; M) \\to 0$ at $t \\to 1$
    in the obvious way.
    """
    dtype = jnp.zeros(()).dtype
    t = jnp.atleast_1d(jnp.asarray(t, dtype=dtype))
    r = _r_indices(M)
    M_f = jnp.asarray(M, dtype=dtype)
    a = r[None, :]
    b = (M_f - r + 1.0)[None, :]
    return betainc(a, b, t[:, None])


def density_basis(t: Float[Array, "n"], M: int) -> Float[Array, "n M"]:
    """Density Bernstein basis :math:`b_r(t; M)` for $r = 1, \\ldots, M$.

    Computed in log-space for numerical stability at the higher $M$ values:

    .. math::
       \\log b_r(t; M) = \\log\\Gamma(M+1) - \\log\\Gamma(r) - \\log\\Gamma(M-r+1)
                       + (r-1)\\log t + (M-r)\\log(1-t).
    """
    dtype = jnp.zeros(()).dtype
    t = jnp.atleast_1d(jnp.asarray(t, dtype=dtype))
    r = _r_indices(M)
    M_f = jnp.asarray(M, dtype=dtype)
    log_norm = gammaln(M_f + 1.0) - gammaln(r) - gammaln(M_f - r + 1.0)
    log_t = jnp.log(jnp.clip(t, 1e-300, 1.0))
    log_1mt = jnp.log1p(-jnp.clip(t, 0.0, 1.0 - 1e-300))
    log_b = (
        log_norm[None, :]
        + (r - 1.0)[None, :] * log_t[:, None]
        + (M_f - r)[None, :] * log_1mt[:, None]
    )
    return jnp.exp(log_b)


def cdf(
    t: Float[Array, "n"],
    weights: Float[Array, "M"],
    M: int,
) -> Float[Array, "n"]:
    """Bernstein-Dirichlet CDF :math:`\\sum_r p_r B_r(t; M)` evaluated at each ``t``.

    The function does not apply the pilot $F_j^{(0)}$ — callers are expected to
    pre-transform inputs via :mod:`py_idr.marginals.transform`.
    """
    weights = jnp.asarray(weights)
    if weights.shape[-1] != M:
        raise ValueError(f"weights must have length M={M}; got {weights.shape}")
    basis = cumulative_basis(t, M)
    return basis @ weights


def pdf(
    t: Float[Array, "n"],
    weights: Float[Array, "M"],
    M: int,
) -> Float[Array, "n"]:
    """Bernstein-Dirichlet density :math:`\\sum_r p_r b_r(t; M)` evaluated at each ``t``."""
    weights = jnp.asarray(weights)
    if weights.shape[-1] != M:
        raise ValueError(f"weights must have length M={M}; got {weights.shape}")
    basis = density_basis(t, M)
    return basis @ weights
