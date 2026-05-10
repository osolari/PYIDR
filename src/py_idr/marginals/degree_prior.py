"""Discrete prior on the Bernstein degree $M_j \\in \\{5, 10, 20, 40, 80\\}$.

Eq. (3.8) of the revised report places mass proportional to $1/M$:

.. math::
   \\Pr(M_j = m) = \\frac{1/m}{\\sum_{m' \\in \\{5,10,20,40,80\\}} 1/m'}.

This module implements the prior as a simple categorical with the support and
weights frozen as module constants.

See plans/03_marginals.md.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

DEGREE_SUPPORT: tuple[int, ...] = (5, 10, 20, 40, 80)


def prior_weights() -> Float[Array, "K"]:
    """Return the $1/M$ prior weights normalised to sum to 1, in the current default dtype.

    Computed lazily (rather than as a module constant) so the function respects
    runtime ``jax_enable_x64`` toggles without needing a module reload.
    """
    dtype = jnp.zeros(()).dtype
    weights = jnp.asarray([1.0 / m for m in DEGREE_SUPPORT], dtype=dtype)
    return weights / jnp.sum(weights)


def log_prior(m: int) -> float:
    """Return $\\log \\Pr(M_j = m)$ for $m$ in the discrete support; $-\\infty$ otherwise."""
    try:
        idx = DEGREE_SUPPORT.index(m)
    except ValueError:
        return float("-inf")
    return float(jnp.log(prior_weights()[idx]))


def sample(key: jax.Array) -> int:
    """Draw $M_j \\sim$ prior."""
    idx = int(jax.random.categorical(key, jnp.log(prior_weights())))
    return DEGREE_SUPPORT[idx]
