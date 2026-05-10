"""Student-$t_5$ copula with correlation matrix $R = L L^\\top$.

The $t_\\nu$ copula has density (Demarta & McNeil 2005)

.. math::
   c_{R,\\nu}(u) \\;=\\; |R|^{-1/2}\\,
       \\frac{\\Gamma((\\nu+K)/2)\\,
              \\Gamma(\\nu/2)^{K-1}}{\\Gamma((\\nu+1)/2)^K\\, \\Gamma(\\nu/2)}
       \\;\\cdot\\;
       \\frac{\\prod_j (1 + z_j^2/\\nu)^{(\\nu+1)/2}}
            {(1 + z^\\top R^{-1} z / \\nu)^{(\\nu+K)/2}},

with $z_j = T_\\nu^{-1}(u_j)$ the inverse univariate-$t$ CDF. The report fixes
$\\nu = 5$.

Like the Gaussian copula, the $t_5$ copula with all positive pairwise correlations
satisfies PLOD; mixed-sign correlations may violate it.

See plans/02_algebra_and_copulas.md.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.special import gammaln
from jaxtyping import Array, Float
from scipy import stats as _sps
from scipy.stats import multivariate_t as _scipy_mvt

from py_idr.copulas.base import Copula

NU: float = 5.0  # fixed degrees of freedom per the report (§3.3.2)


def _t_ppf(u: jax.Array) -> jax.Array:
    """JIT-compatible wrapper around scipy.stats.t.ppf via :func:`jax.pure_callback`.

    JAX's ``jax.scipy.stats.t`` only exposes ``pdf`` / ``logpdf``; the inverse CDF
    requires the inverse incomplete beta which is not yet wrapped. We bridge to SciPy
    on the host. A native JAX implementation is a follow-on (see W4 / plans/05_inference_vi.md).
    """
    return jax.pure_callback(
        lambda x: _sps.t.ppf(np.asarray(x), df=NU).astype(x.dtype),
        jax.ShapeDtypeStruct(u.shape, u.dtype),
        u,
    )


class StudentTCopula(Copula):
    """$t_5$ copula in the Cholesky parameterization $R = L L^\\top$."""

    def __init__(self, L: Float[Array, "K K"]) -> None:
        """Cache the precision matrix and log-determinant."""
        self.K = L.shape[0]
        self.L = L
        self.R = L @ L.T
        self._log_det_R = 2.0 * jnp.sum(jnp.log(jnp.diag(L)))
        self._Rinv = jax.scipy.linalg.cho_solve((L, True), jnp.eye(self.K))

    def log_density(self, u: Float[Array, "n K"]) -> Float[Array, "n"]:
        """Return $\\log c_{R, 5}(u_i)$ at each row."""
        K = self.K
        nu = NU
        z = _t_ppf(u)  # (n, K)
        quad = jnp.einsum("ni,ij,nj->n", z, self._Rinv, z)
        # numerator: prod_j (1 + z_j^2/nu)^{(nu+1)/2} -> sum log
        log_num = ((nu + 1.0) / 2.0) * jnp.sum(jnp.log1p(z * z / nu), axis=1)
        log_denom = ((nu + K) / 2.0) * jnp.log1p(quad / nu)
        # Constant prefactor (Demarta & McNeil 2005, eq. 6):
        #   log Γ((ν+K)/2) + (K-1) log Γ(ν/2) - K log Γ((ν+1)/2)
        log_c = (
            gammaln((nu + K) / 2.0) + (K - 1.0) * gammaln(nu / 2.0) - K * gammaln((nu + 1.0) / 2.0)
        )
        return -0.5 * self._log_det_R + log_c + log_num - log_denom

    def cdf_diagonal(self, u: Float[Array, "n"]) -> Float[Array, "n"]:
        """Return $C_{R,5}(u, \\ldots, u)$ via SciPy's multivariate-$t$ CDF on the host."""
        z_grid = _sps.t.ppf(np.asarray(u), df=NU)
        R = np.asarray(self.R)
        out = np.empty_like(z_grid)
        mean = np.zeros(self.K)
        for i, z in enumerate(z_grid):
            out[i] = _scipy_mvt.cdf(np.full(self.K, z), loc=mean, shape=R, df=NU)
        return jnp.asarray(out)
