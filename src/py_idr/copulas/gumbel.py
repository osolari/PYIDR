"""Gumbel (Archimedean) copula.

The $K$-dimensional Gumbel copula with parameter $\\theta \\geq 1$ has CDF

.. math::
   C_\\theta(u) = \\exp\\!\\left(-S(u)^{1/\\theta}\\right),
   \\qquad S(u) = \\textstyle\\sum_j (-\\log u_j)^\\theta.

The density for general $K$ involves a polynomial in $t = S(u)^{1/\\theta}$ whose
coefficients are computable via Stirling numbers of the first kind (Hofert 2008,
Theorem 3.7). The bivariate case has a clean closed form which we implement here.
The multivariate case ($K \\geq 3$) is deferred to a follow-on (see TODO).

The Gumbel family captures upper-tail dependence and satisfies strict PLOD for
every $\\theta > 1$ (Joe 1997, §4.5; $\\theta = 1$ is the independence copula).

See plans/02_algebra_and_copulas.md.
"""

from __future__ import annotations

import jax.numpy as jnp
from jaxtyping import Array, Float

from py_idr.copulas.base import Copula


class GumbelCopula(Copula):
    """Archimedean Gumbel copula; $\\theta > 1$ (strict PLOD).

    Notes
    -----
    The density evaluator is implemented for $K = 2$ only at the W1 stage. The
    multivariate density (Hofert 2008, Theorem 3.7) is a follow-on; the
    :meth:`cdf_diagonal` method works for any $K$.
    """

    def __init__(self, theta: float | Float[Array, ""], K: int) -> None:
        """Bind dependence parameter and replicate dimension.

        ``theta`` may be a Python float or a JAX scalar (concrete or traced).
        Concrete values are validated against $\\theta > 1$; traced values skip
        the check so NumPyro NUTS samples flow through.
        """
        self.theta = jnp.asarray(theta)
        self.K = int(K)
        try:
            theta_value = float(self.theta)
        except Exception:  # noqa: BLE001  -- JAX tracer conversion errors
            return
        if theta_value <= 1.0:
            raise ValueError(
                f"Gumbel requires theta > 1 for strict PLOD; got {theta_value}. "
                "theta == 1 is the independence copula."
            )

    def log_density(self, u: Float[Array, "n K"]) -> Float[Array, "n"]:
        """Return $\\log c_\\theta(u_i)$ at each row.

        Implemented in closed form for $K = 2$. For $K \\geq 3$ this raises
        :class:`NotImplementedError`; a Hofert (2008)–style multivariate density is
        scheduled for the W4 follow-on.
        """
        if self.K != 2:
            raise NotImplementedError(
                f"Gumbel.log_density currently supports K=2 only; got K={self.K}. "
                "Multivariate Gumbel density via Hofert (2008) is a W4 follow-on; "
                "see plans/02_algebra_and_copulas.md."
            )
        theta = self.theta
        nlog_u = -jnp.log(u[:, 0])
        nlog_v = -jnp.log(u[:, 1])
        S = nlog_u**theta + nlog_v**theta  # (n,)
        t = S ** (1.0 / theta)  # = -log C(u, v)
        # Bivariate Gumbel density (Joe 1997, eq. 4.6.3):
        # c(u,v) = C(u,v) * (uv)^(-1) * (a^theta + b^theta)^(-2 + 2/theta)
        #         * (a*b)^(theta-1) * (a^theta + b^theta + theta - 1)^(?)
        # The standard textbook form is:
        # c(u,v) = (uv)^{-1} * (a*b)^{theta-1} * (S)^{-2 + 1/theta}
        #         * (theta - 1 + S^{1/theta}) * exp(-S^{1/theta})
        log_c = (
            -jnp.log(u[:, 0])
            - jnp.log(u[:, 1])
            + (theta - 1.0) * (jnp.log(nlog_u) + jnp.log(nlog_v))
            + (-2.0 + 1.0 / theta) * jnp.log(S)
            + jnp.log(theta - 1.0 + t)
            - t
        )
        return log_c

    def cdf_diagonal(self, u: Float[Array, "n"]) -> Float[Array, "n"]:
        """Return $C_\\theta(u, \\ldots, u)$ in closed form.

        $S(u, \\ldots, u) = K (-\\log u)^\\theta$, so
        $C(u, \\ldots, u) = \\exp(-K^{1/\\theta} (-\\log u))$.
        """
        K = self.K
        theta = self.theta
        return jnp.exp(-(K ** (1.0 / theta)) * (-jnp.log(u)))
