"""Unit tests for the four atomic-type copula log-densities + the independence copula.

Cross-checked against scipy where applicable.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest
from scipy import stats as sps

from stick_idr.copulas.clayton import ClaytonCopula
from stick_idr.copulas.gaussian import GaussianCopula
from stick_idr.copulas.gumbel import GumbelCopula
from stick_idr.copulas.independence import IndependenceCopula
from stick_idr.copulas.student_t import NU, StudentTCopula


@pytest.mark.unit
def test_independence_log_density_is_zero() -> None:
    """log c_0(u) = 0 for any u."""
    K = 4
    cop = IndependenceCopula(K=K)
    u = jnp.array([[0.1, 0.4, 0.7, 0.9], [0.5, 0.5, 0.5, 0.5]])
    out = cop.log_density(u)
    assert jnp.allclose(out, jnp.zeros(2))


@pytest.mark.unit
def test_independence_diagonal_section_is_uK() -> None:
    """C_0(u, ..., u) = u^K identically."""
    K = 5
    cop = IndependenceCopula(K=K)
    u = jnp.linspace(0.05, 0.95, 10)
    assert jnp.allclose(cop.cdf_diagonal(u), u**K)


# ---- Gaussian ----------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.parametrize("K", [2, 3, 5])
def test_gaussian_density_matches_scipy(K: int) -> None:
    """Gaussian-copula log-density agrees with scipy: log c = log f_R(z) - sum log φ(z)."""
    rho = 0.5
    R = (1.0 - rho) * np.eye(K) + rho * np.ones((K, K))
    L = jnp.linalg.cholesky(jnp.asarray(R))
    cop = GaussianCopula(L)
    rng = np.random.default_rng(0)
    u = rng.uniform(0.1, 0.9, size=(7, K))
    out = np.asarray(cop.log_density(jnp.asarray(u)))
    z = sps.norm.ppf(u)
    log_joint = sps.multivariate_normal.logpdf(z, mean=np.zeros(K), cov=R)
    log_marg = sps.norm.logpdf(z).sum(axis=1)
    expected = log_joint - log_marg
    assert np.allclose(out, expected, atol=1e-10)


# ---- Student-t (nu = 5) ------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.parametrize("K", [2, 3, 5])
def test_student_t_density_matches_scipy(K: int) -> None:
    """t_5 copula log-density agrees with scipy via the multivariate-t / univariate-t ratio."""
    rho = 0.5
    R = (1.0 - rho) * np.eye(K) + rho * np.ones((K, K))
    L = jnp.linalg.cholesky(jnp.asarray(R))
    cop = StudentTCopula(L)
    rng = np.random.default_rng(1)
    u = rng.uniform(0.1, 0.9, size=(7, K))
    out = np.asarray(cop.log_density(jnp.asarray(u)))
    z = sps.t.ppf(u, df=NU)
    log_joint = sps.multivariate_t.logpdf(z, loc=np.zeros(K), shape=R, df=NU)
    log_marg = sps.t.logpdf(z, df=NU).sum(axis=1)
    expected = log_joint - log_marg
    assert np.allclose(out, expected, atol=1e-9)


# ---- Clayton ----------------------------------------------------------------------


@pytest.mark.unit
def test_clayton_diagonal_section_closed_form() -> None:
    """Clayton C(u, ..., u) = (K u^{-θ} - K + 1)^{-1/θ}."""
    K = 4
    theta = 2.0
    cop = ClaytonCopula(theta=theta, K=K)
    u = jnp.linspace(0.1, 0.9, 9)
    expected = (K * u ** (-theta) - K + 1.0) ** (-1.0 / theta)
    assert jnp.allclose(cop.cdf_diagonal(u), expected)


@pytest.mark.unit
def test_clayton_density_bivariate_matches_partial_derivatives() -> None:
    """Bivariate Clayton: dC/du dv yields the closed-form density (sanity vs autodiff).

    c(u, v) = (1 + θ) (uv)^{-θ - 1} (u^{-θ} + v^{-θ} - 1)^{-1/θ - 2}
    """
    theta = 2.0
    cop = ClaytonCopula(theta=theta, K=2)
    rng = np.random.default_rng(2)
    uv = rng.uniform(0.1, 0.9, size=(5, 2))
    out = np.asarray(cop.log_density(jnp.asarray(uv)))
    u, v = uv[:, 0], uv[:, 1]
    expected = (
        np.log1p(theta)
        - (theta + 1.0) * (np.log(u) + np.log(v))
        + (-1.0 / theta - 2.0) * np.log(u ** (-theta) + v ** (-theta) - 1.0)
    )
    assert np.allclose(out, expected, atol=1e-12)


# ---- Gumbel (bivariate only at W1) -------------------------------------------------


@pytest.mark.unit
def test_gumbel_diagonal_section_closed_form() -> None:
    """Gumbel C(u, ..., u) = exp(-K^{1/θ} (-log u))."""
    K = 4
    theta = 1.5
    cop = GumbelCopula(theta=theta, K=K)
    u = jnp.linspace(0.1, 0.9, 9)
    expected = jnp.exp(-(K ** (1.0 / theta)) * (-jnp.log(u)))
    assert jnp.allclose(cop.cdf_diagonal(u), expected)


@pytest.mark.unit
def test_gumbel_log_density_bivariate_matches_textbook() -> None:
    """Bivariate Gumbel density matches Joe (1997) eq. 4.6.3."""
    theta = 1.7
    cop = GumbelCopula(theta=theta, K=2)
    rng = np.random.default_rng(3)
    uv = rng.uniform(0.1, 0.9, size=(5, 2))
    out = np.asarray(cop.log_density(jnp.asarray(uv)))
    u, v = uv[:, 0], uv[:, 1]
    a, b = -np.log(u), -np.log(v)
    S = a**theta + b**theta
    t = S ** (1.0 / theta)
    log_C = -t  # C(u, v) = exp(-t)
    log_density = (
        log_C
        - np.log(u)
        - np.log(v)
        + (theta - 1.0) * (np.log(a) + np.log(b))
        + (-2.0 + 1.0 / theta) * np.log(S)
        + np.log(theta - 1.0 + t)
    )
    assert np.allclose(out, log_density, atol=1e-12)


@pytest.mark.unit
def test_gumbel_log_density_raises_for_K_geq_3() -> None:
    """Multivariate Gumbel density is deferred to a follow-on; raises NotImplementedError."""
    cop = GumbelCopula(theta=1.5, K=3)
    u = jnp.full((1, 3), 0.5)
    with pytest.raises(NotImplementedError, match="Hofert"):
        cop.log_density(u)
