"""Per-replicate score transformation $X_{i,j} \\to U_{i,j} \\in (0, 1)$.

Two paths:

1. **Empirical-rank approximation** (initialization / diagnostics): convert raw
   scores to rank-based pseudo-uniforms. Per the revised §3.2.1, this is *not*
   a replacement for the Bernstein-Dirichlet marginals in the theoretical
   statements; it is an initialization aid.
2. **Fitted Bernstein-Dirichlet CDF**: apply the current
   $F_j(x) = \\sum_r p_{j,r} B_r(F_j^{(0)}(x); M_j)$ to a column.

The first path is wired here; the second is a thin wrapper around
:func:`py_idr.marginals.bernstein.cdf`.

See plans/03_marginals.md.
"""

from __future__ import annotations

from typing import Literal

import jax.numpy as jnp
from jaxtyping import Array, Float

from py_idr.marginals.bernstein import cdf as bernstein_cdf
from py_idr.marginals.pilot import DEFAULT_EPS, kde_pilot_cdf


def empirical_rank_transform(
    x: Float[Array, "n"],
    *,
    tie_policy: Literal["mid_rank", "random", "truncate"] = "mid_rank",
    eps: float = DEFAULT_EPS,
) -> Float[Array, "n"]:
    """Empirical scaled-rank transform $u_i = (\\mathrm{rank}_i) / (n + 1)$.

    Output is clipped to $(eps, 1-eps)$ so downstream copula log-densities
    do not see exact 0 or 1.

    Parameters
    ----------
    x
        Length-$n$ score vector.
    tie_policy
        How to break ties:

        - ``"mid_rank"`` (default): tied scores share the mean rank.
        - ``"random"``: random-permutation rank — JAX-friendly via argsort with
          a small noise jitter; not implemented here, defers to ``mid_rank``
          with a deprecation warning policy in W2.
        - ``"truncate"``: floor the rank to the lowest tied position.

        The runtime currently implements ``"mid_rank"`` and ``"truncate"``;
        ``"random"`` will be added when the data manifest exposes a per-dataset
        tie-handling key (plan 08).
    eps
        Boundary clip; matches the convention used by
        :func:`py_idr.marginals.pilot.kde_pilot_cdf`.
    """
    dtype = jnp.zeros(()).dtype
    x = jnp.asarray(x, dtype=dtype)
    n = x.shape[0]
    if tie_policy == "mid_rank":
        # Average rank: order + reverse-order, halved.
        order = jnp.argsort(x)
        rev_order = jnp.argsort(-x)
        rank = jnp.empty(n, dtype=dtype)
        rank = rank.at[order].set(jnp.arange(1, n + 1, dtype=dtype))
        rev_rank = jnp.empty(n, dtype=dtype)
        rev_rank = rev_rank.at[rev_order].set(jnp.arange(n, 0, -1, dtype=dtype))
        rank = 0.5 * (rank + rev_rank)
    elif tie_policy == "truncate":
        order = jnp.argsort(x)
        rank = jnp.empty(n, dtype=dtype)
        rank = rank.at[order].set(jnp.arange(1, n + 1, dtype=dtype))
    elif tie_policy == "random":
        # See docstring: defer to mid_rank for now; full implementation belongs
        # alongside the per-dataset tie-handling key in plan 08.
        return empirical_rank_transform(x, tie_policy="mid_rank", eps=eps)
    else:
        raise ValueError(
            f"tie_policy must be one of {{'mid_rank', 'random', 'truncate'}}; got {tie_policy!r}"
        )
    u = rank / jnp.asarray(n + 1.0, dtype=dtype)
    return jnp.clip(u, eps, 1.0 - eps)


def transform_X_to_U(
    x: Float[Array, "n"],
    weights: Float[Array, "M"],
    M: int,
    F0_eval_points: Float[Array, "n"],
    *,
    eps: float = DEFAULT_EPS,
) -> Float[Array, "n"]:
    """Apply the fitted Bernstein-Dirichlet CDF to a column of scores.

    Parameters
    ----------
    x
        Length-$n$ score vector. Used only as a positional reference; the
        actual transformation operates on the pre-computed pilot-CDF values.
    weights
        Current Bernstein-weight vector $\\bm p \\in \\Delta^{M-1}$.
    M
        Bernstein degree.
    F0_eval_points
        Pre-computed pilot-CDF values $F^{(0)}(x_i) \\in (eps, 1-eps)$.
    eps
        Boundary clip on the output.

    Returns
    -------
    Length-$n$ vector of $U_i = F_j(x_i)$ values, strictly in $(eps, 1-eps)$.
    """
    del x  # documentation-only: enforced via the F0 input
    u = bernstein_cdf(F0_eval_points, weights, M)
    return jnp.clip(u, eps, 1.0 - eps)


def fit_pilot_and_transform(
    x: Float[Array, "n"],
    *,
    bandwidth: float | None = None,
    eps: float = DEFAULT_EPS,
) -> Float[Array, "n"]:
    """Fit the wide-bandwidth pilot CDF on $x$ and return $F^{(0)}(x)$.

    Used as the input to the Bernstein-Dirichlet marginal at initialisation —
    the Bernstein mixture corrects from this coarse start.
    """
    return kde_pilot_cdf(x, x, bandwidth=bandwidth, eps=eps)
