"""Cholesky parameterization, LKJ prior, and structured-correlation classes.

The reproducible-component Gaussian copula atoms in PY-IDR are parameterised by a
$K \\times K$ correlation matrix $R$ via its lower-triangular Cholesky factor
$L$, where $R = L L^\\top$ and $L_{j,j} > 0$.  Inference operates on the
unconstrained Cholesky coordinates so the support is open in $\\mathbb{R}^{K(K+1)/2}$
and NUTS is well-defined.

Per the report (§3.3.5), the prior on $R$ at cluster instantiation is one of three
structural classes — *exchangeable*, *one-factor*, or *two-factor* — selected
uniformly, with an LKJ$(\\eta=2)$ prior on the unconstrained correlation entries.

References
----------
Lewandowski, Kurowicka, Joe (2009). "Generating random correlation matrices based
on vines and extended onion method." *Journal of Multivariate Analysis*.

See Also
--------
plans/02_algebra_and_copulas.md
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float


def cholesky_to_corr(L: Float[Array, "K K"]) -> Float[Array, "K K"]:
    """Materialise the correlation matrix R = L L^T from a Cholesky factor.

    Parameters
    ----------
    L
        Lower-triangular Cholesky factor with strictly positive diagonal.

    Returns
    -------
    R
        Symmetric positive-definite correlation matrix; diagonal is unit only if
        L's row norms are unit (use :func:`unit_normalise_cholesky` to enforce).
    """
    return L @ L.T


def corr_to_cholesky(R: Float[Array, "K K"]) -> Float[Array, "K K"]:
    """Return the lower-triangular Cholesky factor L such that R = L L^T.

    Adds a tiny diagonal jitter when the input is mildly non-PSD due to round-off.
    """
    K = R.shape[0]
    jitter = 1e-12 * jnp.eye(K)
    return jnp.linalg.cholesky(R + jitter)


def unit_normalise_cholesky(L: Float[Array, "K K"]) -> Float[Array, "K K"]:
    """Rescale rows of ``L`` so that ``L @ L.T`` has unit diagonal.

    The PY-IDR Gaussian-copula atoms must have unit diagonal (it is a *correlation*
    matrix, not a covariance). Sampling from the unconstrained Cholesky coordinates
    can violate this; this helper enforces it post hoc.
    """
    row_norms = jnp.linalg.norm(L, axis=1, keepdims=True)
    return L / row_norms


def lkj_log_density(L: Float[Array, "K K"], eta: float = 2.0) -> Float[Array, ""]:
    """Log-density of the LKJ($\\eta$) prior in the unit-row-norm Cholesky parameterization.

    For $R = L L^\\top$ with unit diagonal the LKJ density is
    $p(R) \\propto |R|^{\\eta - 1}$. Working on the Cholesky factor adds a Jacobian
    correction $\\prod_{k=1}^{K-1} L_{k+1,k+1}^{K - k - 1 + 2(\\eta - 1)}$ — see
    Lewandowski–Kurowicka–Joe (2009, Theorem 4) and Stan's `lkj_corr_cholesky` reference.

    Parameters
    ----------
    L
        Lower-triangular Cholesky factor with positive diagonal and unit row-norm.
    eta
        Shape parameter; ``eta=1`` is uniform on correlation matrices, ``eta>1`` is
        peaked at the identity. The report uses ``eta=2`` (mild concentration).

    Returns
    -------
    Scalar log density (up to a normalisation constant that does not depend on ``L``).
    """
    K = L.shape[0]
    diag = jnp.diag(L)
    # Power on each diagonal entry per Stan's parameterisation.
    powers = (K - jnp.arange(1, K + 1)) + 2.0 * (eta - 1.0)
    return jnp.sum(powers * jnp.log(diag))


# --------------------------------------------------------------------------------------
# Structured correlation classes (§3.3.2)
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class StructuredCorrelation:
    """Base for the three structured-correlation classes used at cluster instantiation."""

    K: int

    def matrix(self) -> Float[Array, "K K"]:  # pragma: no cover - abstract
        """Return the dense $K \\times K$ correlation matrix."""
        raise NotImplementedError


@dataclass(frozen=True)
class Exchangeable(StructuredCorrelation):
    """$R = (1 - \\rho) I + \\rho \\mathbf{1}\\mathbf{1}^\\top$ with $\\rho \\in (-1/(K-1), 1)$."""

    rho: float

    def matrix(self) -> Float[Array, "K K"]:
        """Materialise the dense exchangeable correlation matrix."""
        return (1.0 - self.rho) * jnp.eye(self.K) + self.rho * jnp.ones((self.K, self.K))


@dataclass(frozen=True)
class OneFactor(StructuredCorrelation):
    """$R = D + \\lambda \\lambda^\\top$, $D$ chosen so the diagonal is unit."""

    loadings: Float[Array, "K"]

    def matrix(self) -> Float[Array, "K K"]:
        """Materialise the one-factor correlation matrix with unit diagonal."""
        lam = self.loadings
        outer = jnp.outer(lam, lam)
        diag = jnp.diag(jnp.maximum(1.0 - lam * lam, 1e-8))
        return diag + outer


@dataclass(frozen=True)
class TwoFactor(StructuredCorrelation):
    """$R = D + \\Lambda \\Lambda^\\top$, $\\Lambda \\in \\mathbb{R}^{K\\times 2}$."""

    loadings: Float[Array, "K 2"]

    def matrix(self) -> Float[Array, "K K"]:
        """Materialise the two-factor correlation matrix with unit diagonal."""
        lam = self.loadings
        outer = lam @ lam.T
        # Unit diagonal: diag(D) = 1 - row-sum-of-squares(Λ)
        row_ss = jnp.sum(lam * lam, axis=1)
        diag = jnp.diag(jnp.maximum(1.0 - row_ss, 1e-8))
        return diag + outer


def sample_lkj_cholesky(key: jax.Array, K: int, eta: float = 2.0) -> Float[Array, "K K"]:
    """Draw a Cholesky factor from the LKJ($\\eta$) prior on $K \\times K$ correlation matrices.

    Uses the canonical "onion" construction of Lewandowski–Kurowicka–Joe (2009). The
    returned factor has unit row-norm so that ``cholesky_to_corr(L)`` is a valid
    correlation matrix.
    """
    L = jnp.zeros((K, K))
    L = L.at[0, 0].set(1.0)
    keys = jax.random.split(key, K - 1)
    for k in range(1, K):
        beta = jax.random.beta(keys[k - 1], (k + 2.0 * eta - 2.0) / 2.0, k / 2.0)
        # Surface point on the (k-1)-sphere
        sub = jax.random.normal(jax.random.fold_in(keys[k - 1], k), (k,))
        sub = sub / jnp.linalg.norm(sub)
        scale = jnp.sqrt(beta)
        L = L.at[k, :k].set(scale * sub)
        L = L.at[k, k].set(jnp.sqrt(1.0 - beta))
    return L
