r"""Pitman-Yor stick-breaking (GEM) sampler.

A draw $(w_1, w_2, \ldots) \sim \mathrm{GEM}(\alpha, \sigma)$ is built
by the **stick-breaking** construction:

- Independent $v_m \sim \mathrm{Beta}(1 - \sigma,\ \alpha + m\sigma)$ for
  $m = 1, 2, \ldots$.
- $w_m = v_m \prod_{k < m} (1 - v_k)$ — the fraction of the remaining
  stick at step $m$.

The full GEM is an infinite measure on the simplex, but every practical
use truncates to a finite $T$. This module returns the first $T$
weights; the residual $1 - \sum_{m=1}^{T} w_m$ is left unallocated.
The truncation error decays exponentially in $T$ as long as
$\alpha + T\sigma$ is comfortably large.

The Pitman-Yor mixture in PY-IDR uses these weights for the
reproducible copula clusters. See the
:mod:`py_idr.pym` package docstring and §3.2.1 of the report.

References
----------
- Pitman & Yor (1997, Ann. Probab.), Theorem 3.2.
- Ishwaran & James (2001, JASA), stick-breaking representation.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float


def gem_sample(
    key: jax.Array,
    alpha: float,
    sigma: float,
    T: int,
) -> Float[Array, "T"]:
    r"""Return the truncated stick-breaking weights $w_{1:T}$ drawn from GEM($\alpha$, $\sigma$).

    Parameters
    ----------
    key
        JAX PRNG key.
    alpha
        Pitman-Yor concentration. Must satisfy $\alpha > -\sigma$;
        when $\sigma = 0$ this reduces to the usual DP constraint
        $\alpha > 0$.
    sigma
        Pitman-Yor discount in $[0, 1)$. $\sigma = 0$ recovers the
        Dirichlet process special case.
    T
        Truncation level. Must be at least 1.

    Returns
    -------
    Length-$T$ JAX array of weights, each in $[0, 1]$. By construction
    the weights are monotonically more likely to be small as $m$
    grows; they do **not** necessarily sum to 1 (the residual
    $1 - \sum_m w_m$ is left unallocated, hence "truncated").

    Raises
    ------
    ValueError
        If $\sigma \notin [0, 1)$, $\alpha + \sigma \le 0$, or $T < 1$.
    """
    if not (0.0 <= sigma < 1.0):
        raise ValueError(f"sigma must be in [0, 1); got {sigma}")
    if alpha + sigma <= 0.0:
        raise ValueError(
            f"need alpha + sigma > 0 for Beta(1-sigma, alpha + m*sigma) to be "
            f"well-defined; got alpha={alpha}, sigma={sigma}"
        )
    if T < 1:
        raise ValueError(f"T must be >= 1; got {T}")

    # Beta(1 - sigma, alpha + m * sigma) for m = 1, ..., T.
    m_range = jnp.arange(1, T + 1, dtype=jnp.float32)
    a = jnp.full((T,), 1.0 - sigma, dtype=jnp.float32)
    b = alpha + m_range * sigma  # shape (T,)
    v = jax.random.beta(key, a, b)  # shape (T,)

    # w_m = v_m * prod_{k < m}(1 - v_k). Log space avoids underflow at large T.
    log_one_minus_v = jnp.log1p(-v)
    log_remaining = jnp.concatenate([jnp.zeros((1,)), jnp.cumsum(log_one_minus_v[:-1])])
    log_w = jnp.log(v) + log_remaining
    return jnp.exp(log_w)
