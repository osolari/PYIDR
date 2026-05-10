"""Bayes-mFDR with the revised ``max(1, E[R])`` denominator.

Per Eq. (3.10) of the revised report,

.. math::
   \\mathrm{Bayes\\text{-}mFDR}(\\gamma)
   = \\frac{\\E[V(\\gamma) \\mid \\mathrm{data}]}
          {\\max\\{1, \\E[R(\\gamma) \\mid \\mathrm{data}]\\}},

with $V(\\gamma) = \\sum_i (1 - Z_i)\\,\\bm{1}\\{\\widehat\\idr_i < \\gamma\\}$ and
$R(\\gamma) = \\sum_i \\bm{1}\\{\\widehat\\idr_i < \\gamma\\}$.

The denominator change (from $\\E[R(\\gamma) \\mid \\mathrm{data}]$ to
$\\max\\{1, \\E[R(\\gamma) \\mid \\mathrm{data}]\\}$) makes the empty-rejection branch
explicit. When no feature crosses the threshold, the original definition was
$0/0$; the revised definition is just $0$. Theorem 3.3 of the revised report
*requires* the denominator to be bounded away from zero in a neighbourhood of the
target threshold (Assumption 3.5), and the operational ``max(1, ...)`` form
enforces that.

Operationally, given $\\widehat\\idr_i$ already integrated over the posterior:

- $\\E[V(\\gamma) \\mid \\mathrm{data}]
   = \\sum_i \\widehat\\idr_i\\,\\bm{1}\\{\\widehat\\idr_i < \\gamma\\}$
- $\\E[R(\\gamma) \\mid \\mathrm{data}]
   = \\sum_i \\bm{1}\\{\\widehat\\idr_i < \\gamma\\}$

— the rejection counts are deterministic given $\\widehat\\idr$, and the $V$
expectation reduces to the sum of idrs in the rejection set.

See plans/06_decision_theory.md.
"""

from __future__ import annotations

import jax.numpy as jnp
from jaxtyping import Array, Float


def bayes_mfdr_curve(
    idr: Float[Array, "n"],
    gammas: Float[Array, "g"],
) -> Float[Array, "g"]:
    """Bayes-mFDR at each threshold $\\gamma$, with the ``max(1, E[R])`` denominator.

    Parameters
    ----------
    idr
        Posterior local idrs $\\widehat\\idr_i \\in [0, 1]$ — already integrated over
        the posterior (e.g., from :func:`py_idr.decision.local_idr.from_mcmc`).
    gammas
        Vector of thresholds $\\gamma \\in (0, 1)$ at which to evaluate the curve.

    Returns
    -------
    Length-``g`` array of $\\mathrm{Bayes\\text{-}mFDR}(\\gamma)$ values.
    """
    idr = jnp.asarray(idr)
    gammas = jnp.asarray(gammas)
    # Boolean rejection mask: shape (n, g).
    reject = idr[:, None] < gammas[None, :]
    expected_R = jnp.sum(reject, axis=0)
    expected_V = jnp.sum(idr[:, None] * reject, axis=0)
    return expected_V / jnp.maximum(1.0, expected_R)


def posterior_decomposition(
    idr: Float[Array, "n"],
    gamma: float,
) -> tuple[float, float]:
    """Return $(\\E[V(\\gamma)], \\E[R(\\gamma)])$ separately for caller audit.

    Useful when the caller wants to plot numerator and denominator independently
    (the denominator floor in :func:`bayes_mfdr_curve` is silent on its own).
    """
    idr = jnp.asarray(idr)
    reject = idr < gamma
    expected_R = float(jnp.sum(reject))
    expected_V = float(jnp.sum(idr * reject))
    return expected_V, expected_R
