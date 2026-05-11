"""Clayton (Archimedean) copula.

The $K$-dimensional Clayton copula with parameter $\\theta > 0$ has CDF and density

.. math::
   C_\\theta(u) = \\bigl(\\sum_{j=1}^K u_j^{-\\theta} - K + 1\\bigr)^{-1/\\theta},
   \\qquad
   c_\\theta(u) = \\prod_{j=0}^{K-1}(1 + j\\theta)\\,
       \\bigl(\\prod_j u_j\\bigr)^{-(\\theta + 1)}\\,
       \\bigl(S(u) - K + 1\\bigr)^{-(K + 1/\\theta)},

with $S(u) = \\sum_j u_j^{-\\theta}$. The Clayton family captures lower-tail dependence
and satisfies strict PLOD for every $\\theta > 0$ (Joe 1997, §4.5).

See plans/02_algebra_and_copulas.md.
"""

from __future__ import annotations

import jax.numpy as jnp
from jaxtyping import Array, Float

from py_idr.copulas.base import Copula


class ClaytonCopula(Copula):
    r"""Archimedean Clayton copula with **lower-tail dependence**.

    The Clayton copula concentrates mass near $u = (0, \ldots, 0)$ —
    features that are *jointly low-ranked* across replicates are more
    likely under Clayton than under the Gaussian copula at the same
    Kendall's tau. PY-IDR uses Clayton in the reproducible mixture when
    the experiment-specific notion of "agreement" is "both replicates
    agree at the bottom of the score distribution" rather than "both
    agree at the top." (Real-world example: agreement on *background*
    regions in a ChIP-seq peak call.)

    Tail-dependence coefficient: $\lambda_L = 2^{-1/\theta}$.
    PLOD compatibility: holds for every $\theta > 0$ (Joe 1997, §4.5).

    Parameters
    ----------
    theta
        Dependence parameter. Must be positive. May be a Python float or
        a JAX scalar (traced or concrete) — the type-update step samples
        the prior $\theta \sim \mathrm{Exponential}(1)$ truncated to
        $\theta > 0$.
    K
        Replicate count. The Clayton density is well-defined for every
        $K \ge 2$; PY-IDR uses the closed-form density below.
    """

    def __init__(self, theta: float | Float[Array, ""], K: int) -> None:
        """Bind dependence parameter and replicate dimension.

        ``theta`` may be a Python float or a JAX scalar (concrete or traced).
        Concrete values are validated against the ``theta > 0`` constraint;
        traced values skip the check (NumPyro NUTS samples flow through).
        """
        self.theta = jnp.asarray(theta)
        self.K = int(K)
        # Validate only at concrete-time — traced tracers raise on float() with
        # ConcretizationTypeError / TracerArrayConversionError.
        try:
            theta_value = float(self.theta)
        except Exception:  # noqa: BLE001  -- JAX tracer conversion errors
            return
        if theta_value <= 0.0:
            raise ValueError(f"Clayton requires theta > 0; got {theta_value}.")

    def log_density(self, u: Float[Array, "n K"]) -> Float[Array, "n"]:
        """Return $\\log c_\\theta(u_i)$ at each row."""
        theta = self.theta
        K = self.K
        # Constant prefactor: sum_{j=0}^{K-1} log(1 + j theta)
        js = jnp.arange(K)
        log_prefactor = jnp.sum(jnp.log1p(js * theta))
        # Sum and product terms.
        log_u = jnp.log(u)  # (n, K)
        S = jnp.sum(u ** (-theta), axis=1)  # (n,)
        term_a = -(theta + 1.0) * jnp.sum(log_u, axis=1)
        term_b = -(K + 1.0 / theta) * jnp.log(S - K + 1.0)
        return log_prefactor + term_a + term_b

    def cdf_diagonal(self, u: Float[Array, "n"]) -> Float[Array, "n"]:
        """Return $C_\\theta(u, \\ldots, u)$ in closed form."""
        theta = self.theta
        K = self.K
        return (K * u ** (-theta) - K + 1.0) ** (-1.0 / theta)
