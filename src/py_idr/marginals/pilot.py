"""Coarse pilot CDF $F_j^{(0)}$ used as the index in the Bernstein basis.

Per Eq. (3.7) of the report: ``F_j(x) = Σ_r p_{j,r} B_r(F_j^{(0)}(x); M_j)``.
The pilot is intentionally crude — a wide-bandwidth Gaussian KDE on the
empirical distribution. The Bernstein-Dirichlet mixture corrects it.

The implementation uses Silverman's bandwidth rule with a clipping floor and
returns values strictly in ``(eps, 1 - eps)`` to keep ``log(F_0)`` and
``log(1 - F_0)`` finite for the density basis.

See plans/03_marginals.md.
"""

from __future__ import annotations

import jax.numpy as jnp
from jax.scipy.stats import norm as jnorm
from jaxtyping import Array, Float

DEFAULT_EPS: float = 1e-6


def silverman_bandwidth(x: Float[Array, "n"]) -> float:
    """Silverman's plug-in bandwidth: $h = 1.06 \\hat\\sigma n^{-1/5}$.

    Used as the *coarse* default; the pilot is then deliberately wider via the
    multiplier in :func:`kde_pilot_cdf` so the Bernstein mixture has clear room
    to correct.
    """
    n = x.shape[0]
    sigma = jnp.std(x, ddof=1)
    return float(1.06 * sigma * (n ** (-1.0 / 5.0)))


def kde_pilot_cdf(
    x: Float[Array, "n"],
    eval_points: Float[Array, "m"] | None = None,
    *,
    bandwidth: float | None = None,
    bandwidth_multiplier: float = 2.0,
    eps: float = DEFAULT_EPS,
) -> Float[Array, "m"]:
    """Wide-bandwidth Gaussian-KDE empirical CDF, clamped to ``(eps, 1 - eps)``.

    .. math::
       F^{(0)}(t) = \\frac{1}{n} \\sum_i \\Phi\\!\\left(\\frac{t - x_i}{h}\\right).

    Parameters
    ----------
    x
        Length-$n$ training sample.
    eval_points
        Points at which to evaluate the pilot. Defaults to ``x``.
    bandwidth
        Pre-specified bandwidth $h$. If ``None``, ``bandwidth_multiplier`` is
        applied to :func:`silverman_bandwidth` for an intentionally wide pilot.
    bandwidth_multiplier
        Multiplier on the Silverman default. Defaults to ``2.0``.
    eps
        Lower / upper clip applied to the output so the value is in $(eps, 1-eps)$.

    Returns
    -------
    Length-$m$ vector of pilot-CDF values, strictly in $(eps, 1-eps)$.
    """
    dtype = jnp.zeros(()).dtype
    x = jnp.asarray(x, dtype=dtype)
    if eval_points is None:
        eval_points = x
    eval_points = jnp.asarray(eval_points, dtype=dtype)
    if bandwidth is None:
        bandwidth = bandwidth_multiplier * silverman_bandwidth(x)
    if bandwidth <= 0.0:
        raise ValueError(f"bandwidth must be positive; got {bandwidth}")

    # Φ-transformed empirical CDF, averaging over the training points.
    z = (eval_points[:, None] - x[None, :]) / bandwidth
    cdf_at_eval = jnp.mean(jnorm.cdf(z), axis=1)
    return jnp.clip(cdf_at_eval, eps, 1.0 - eps)
