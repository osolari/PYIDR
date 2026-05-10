"""Gaussian copula parameterised by a Cholesky factor.

The Gaussian copula on $[0, 1]^K$ with correlation matrix $R = L L^\\top$ has density

.. math::
   c_R(u) \\;=\\; |R|^{-1/2} \\exp\\!\\left(-\\tfrac{1}{2}\\, z^\\top (R^{-1} - I)\\, z\\right),
   \\qquad z = \\Phi^{-1}(u),

where $\\Phi^{-1}$ is the inverse standard-normal CDF applied componentwise. The
diagonal section $C_R(u, \\ldots, u) = \\Phi_R(z, \\ldots, z)$ requires the multivariate
normal CDF and is implemented for the PLOD probe via SciPy on the CPU host.

PLOD: a Gaussian copula with all pairwise correlations $\\rho_{jk} > 0$ satisfies the
strict PLOD inequality (Joe 1997, §2.1.4). Mixed-sign correlations may violate it;
the report's base measure accordingly uses LKJ$(\\eta=2)$ with a sign-positive
post-processing layer enforced by the registry.

See plans/02_algebra_and_copulas.md.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.stats import norm as jnorm
from jaxtyping import Array, Float
from scipy.stats import multivariate_normal as _scipy_mvn

from py_idr.copulas.base import Copula


class GaussianCopula(Copula):
    """Gaussian copula in the Cholesky parameterization $R = L L^\\top$.

    Parameters
    ----------
    L
        Lower-triangular Cholesky factor of $R$. Must have unit row-norm so $R$ has
        unit diagonal (use :func:`py_idr.algebra.correlation.unit_normalise_cholesky`
        if needed).
    """

    def __init__(self, L: Float[Array, "K K"]) -> None:
        """Cache derived quantities used by the log-density."""
        self.K = L.shape[0]
        self.L = L
        self.R = L @ L.T
        # log|R| = 2 sum log diag(L)
        self._log_det_R = 2.0 * jnp.sum(jnp.log(jnp.diag(L)))
        # R^{-1} - I as a single matrix (precomputed; small K).
        K = self.K
        eye = jnp.eye(K)
        # Solve R x = I via L L^T x = I — cheaper than direct inverse.
        Rinv = jax.scipy.linalg.cho_solve((L, True), eye)
        self._Rinv_minus_I = Rinv - eye

    def log_density(self, u: Float[Array, "n K"]) -> Float[Array, "n"]:
        """Return $\\log c_R(u_i)$ at each row.

        Parameters
        ----------
        u
            Pseudo-uniform observations in the open cube $(0, 1)^K$.
        """
        z = jnorm.ppf(u)  # shape (n, K)
        # z^T (R^{-1} - I) z, vectorised over the leading axis.
        quad = jnp.einsum("ni,ij,nj->n", z, self._Rinv_minus_I, z)
        return -0.5 * (self._log_det_R + quad)

    def cdf_diagonal(self, u: Float[Array, "n"]) -> Float[Array, "n"]:
        """Return $\\Phi_R(z, \\ldots, z)$ where $z = \\Phi^{-1}(u)$.

        Implemented via SciPy's multivariate normal CDF on the host CPU; this routine
        is used by the PLOD probe and not in inner-loop sampling.
        """
        z_grid = np.asarray(jnorm.ppf(u))
        R = np.asarray(self.R)
        out = np.empty_like(z_grid)
        mean = np.zeros(self.K)
        for i, z in enumerate(z_grid):
            out[i] = _scipy_mvn.cdf(np.full(self.K, z), mean=mean, cov=R)
        return jnp.asarray(out)
