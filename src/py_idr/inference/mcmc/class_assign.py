"""Algorithm 1 step 1: class assignment $Z_i \\sim \\mathrm{Bernoulli}(\\widehat\\pi_i)$.

Per Eq. (3.4) of the revised report,

.. math::
   \\widehat\\pi_i = \\Pr(Z_i = 1 \\mid \\bU_i, \\theta, w, F)
   = \\frac{\\pi\\, c_1(\\bU_i; \\theta, w)}{\\pi\\, c_1(\\bU_i; \\theta, w)
                                              + (1 - \\pi)\\, c_0(\\bU_i)},

where $c_0$ is the independence copula. Working in log-space to avoid
underflow on tail-probability features:

.. math::
   \\log \\widehat\\pi_i = \\mathrm{logit}^{-1}\\!
       \\bigl(\\log \\pi - \\log(1 - \\pi)
                + \\log c_1(\\bU_i) - \\log c_0(\\bU_i)\\bigr).

For the standard convention $c_0 = 1$, the $\\log c_0$ term drops; we keep it
as a parameter so a future weak-dependence $c_0$ can drop in.

See plans/04_inference_mcmc.md.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Integer


def conditional_class_log_odds(
    pi: float | Float[Array, ""],
    log_c1: Float[Array, "n"],
    log_c0: Float[Array, "n"] | None = None,
) -> Float[Array, "n"]:
    r"""Return the per-feature log-odds $\log\widehat\pi_i / (1 - \widehat\pi_i)$.

    Parameters
    ----------
    pi
        Current scalar mixing proportion $\pi \in (0, 1)$.
    log_c1
        Log of the reproducible-component density at each feature,
        $\log c_1(\bU_i)$.
    log_c0
        Optional log of the irreproducible-component density at each feature.
        Defaults to ``None`` (interpreted as $c_0 = 1$, the independence copula).

    Returns
    -------
    Length-$n$ vector of class log-odds, ready to feed into a Bernoulli
    sampler via ``jax.random.bernoulli(key, jax.nn.sigmoid(log_odds))``.
    """
    pi_scalar = jnp.asarray(pi, dtype=log_c1.dtype)
    log_pi = jnp.log(pi_scalar)
    log_one_minus_pi = jnp.log1p(-pi_scalar)
    log_odds = log_pi - log_one_minus_pi + log_c1
    if log_c0 is not None:
        log_odds = log_odds - log_c0
    return log_odds


def class_assign(
    key: jax.Array,
    pi: float | Float[Array, ""],
    log_c1: Float[Array, "n"],
    log_c0: Float[Array, "n"] | None = None,
) -> Integer[Array, "n"]:
    r"""Sample $Z_i \sim \mathrm{Bernoulli}(\widehat\pi_i)$ at every feature.

    Parameters
    ----------
    key
        JAX PRNG key.
    pi, log_c1, log_c0
        See :func:`conditional_class_log_odds`.

    Returns
    -------
    Length-$n$ array of int32 class indicators in $\{0, 1\}$.
    """
    log_odds = conditional_class_log_odds(pi, log_c1, log_c0)
    pi_hat = jax.nn.sigmoid(log_odds)
    z = jax.random.bernoulli(key, pi_hat)
    return z.astype(jnp.int32)
