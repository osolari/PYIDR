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


def cov_to_corr(Sigma: Float[Array, "K K"]) -> Float[Array, "K K"]:
    """Diagonal-normalise a positive-definite covariance to a correlation matrix.

    Computes $R = D^{-1/2} \\Sigma D^{-1/2}$ where $D = \\mathrm{diag}(\\Sigma)$.
    Per §3.2.2 of the revised report, the structured Gaussian-copula atom classes
    operate on a free unconstrained $\\Sigma$ (built as $D + \\Lambda \\Lambda^\\top$
    for the factor classes) and feed the normalised $R$ to the copula density and
    the PLOD probe.

    Algorithm 1 step 3 also applies this to NUTS / Metropolis proposals before the
    PLOD support check, so that proposals never need a defensive clamp.

    Parameters
    ----------
    Sigma
        Positive-definite $K \\times K$ matrix with strictly positive diagonal.

    Returns
    -------
    R
        Correlation matrix with unit diagonal and the same off-diagonal sign pattern
        as ``Sigma``.
    """
    diag_inv_sqrt = 1.0 / jnp.sqrt(jnp.diag(Sigma))
    return Sigma * diag_inv_sqrt[:, None] * diag_inv_sqrt[None, :]


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
    """$R = (1-\\rho) I + \\rho \\mathbf{1}\\mathbf{1}^\\top$ with $0 < \\rho < 1$.

    The strict positivity of $\\rho$ is the §3.2.2 PLOD-compatible constraint: the
    independence boundary $\\rho = 0$ is excluded from the reproducible-component
    base measure.
    """

    rho: float

    def matrix(self) -> Float[Array, "K K"]:
        """Materialise the dense exchangeable correlation matrix."""
        return (1.0 - self.rho) * jnp.eye(self.K) + self.rho * jnp.ones((self.K, self.K))


@dataclass(frozen=True)
class OneFactor(StructuredCorrelation):
    """One-factor structured correlation in the PSD-then-normalise parameterisation.

    Per §3.2.2 of the revised report, the structured class is parameterised through
    an unconstrained positive-definite $\\Sigma = D + \\lambda \\lambda^\\top$ which is
    then normalised:
    $R = \\mathrm{diag}(\\Sigma)^{-1/2}\\,\\Sigma\\,\\mathrm{diag}(\\Sigma)^{-1/2}$.

    PLOD requires every off-diagonal entry of the resulting $R$ to be strictly
    positive. The implementation either constrains the loadings to a common sign
    (the recommended default) or rejects/proposes again until
    :func:`stick_idr.algebra.plod.plod_with_positive_pairwise` is satisfied.

    Parameters
    ----------
    K
        Replicate dimension.
    loadings
        Length-$K$ factor loading vector. Must be common-sign for PLOD compatibility.
    idiosyncratic
        Strictly positive length-$K$ diagonal. Defaults to ones (the standard
        "unit idiosyncratic variance" choice); supplying a non-trivial diagonal
        lets the sampler fit a free per-replicate scale that is then absorbed by
        :func:`cov_to_corr`.
    """

    loadings: Float[Array, "K"]
    idiosyncratic: Float[Array, "K"] | None = None

    def matrix(self) -> Float[Array, "K K"]:
        """Materialise $R$ via :func:`cov_to_corr` on $\\mathrm{diag}(D)+\\lambda\\lambda^\\top$."""
        lam = self.loadings
        d = self.idiosyncratic if self.idiosyncratic is not None else jnp.ones(self.K)
        Sigma = jnp.diag(d) + jnp.outer(lam, lam)
        return cov_to_corr(Sigma)


@dataclass(frozen=True)
class TwoFactor(StructuredCorrelation):
    """Two-factor structured correlation in the PSD-then-normalise parameterisation.

    $\\Sigma = D + \\Lambda \\Lambda^\\top$ with $\\Lambda \\in \\mathbb{R}^{K\\times 2}$,
    normalised via :func:`cov_to_corr`. As with :class:`OneFactor`, PLOD requires
    every pairwise off-diagonal entry of the normalised $R$ to be strictly positive;
    use a common-sign restriction on the columns of $\\Lambda$ or the
    `plod_with_positive_pairwise` rejection logic in the sampler.
    """

    loadings: Float[Array, "K 2"]
    idiosyncratic: Float[Array, "K"] | None = None

    def matrix(self) -> Float[Array, "K K"]:
        """Materialise $R$ via :func:`cov_to_corr` on $\\mathrm{diag}(D)+\\Lambda\\Lambda^\\top$."""
        lam = self.loadings
        d = self.idiosyncratic if self.idiosyncratic is not None else jnp.ones(self.K)
        Sigma = jnp.diag(d) + lam @ lam.T
        return cov_to_corr(Sigma)


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
