"""Unit tests for py_idr.algebra.correlation."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from py_idr.algebra import correlation as C


@pytest.mark.unit
@pytest.mark.parametrize("K", [2, 3, 5, 8])
def test_cholesky_round_trip(K: int) -> None:
    """corr_to_cholesky ∘ cholesky_to_corr is identity on PSD matrices."""
    key = jax.random.PRNGKey(0)
    A = jax.random.normal(key, (K, K))
    R = A @ A.T + jnp.eye(K)
    # Normalise to a correlation matrix.
    d = jnp.sqrt(jnp.diag(R))
    R = R / jnp.outer(d, d)
    L = C.corr_to_cholesky(R)
    R2 = C.cholesky_to_corr(L)
    assert jnp.allclose(R, R2, atol=1e-10)


@pytest.mark.unit
@pytest.mark.parametrize("K", [2, 5, 10])
def test_unit_normalise_cholesky_yields_unit_diagonal(K: int) -> None:
    """unit_normalise_cholesky enforces unit diagonal on R = L L^T."""
    key = jax.random.PRNGKey(1)
    L = jax.random.normal(key, (K, K)) + jnp.eye(K)
    L = jnp.tril(L)
    L = C.unit_normalise_cholesky(L)
    R = C.cholesky_to_corr(L)
    assert jnp.allclose(jnp.diag(R), jnp.ones(K), atol=1e-10)


@pytest.mark.unit
@pytest.mark.parametrize("K", [2, 5, 8])
def test_lkj_log_density_finite_on_identity(K: int) -> None:
    """LKJ log-density is finite at L = I."""
    L = jnp.eye(K)
    val = C.lkj_log_density(L, eta=2.0)
    assert jnp.isfinite(val)


@pytest.mark.unit
@pytest.mark.parametrize("K", [2, 3, 5])
def test_sample_lkj_cholesky_yields_correlation_matrix(K: int) -> None:
    """Sampled L gives unit row-norm and a PSD correlation matrix."""
    key = jax.random.PRNGKey(42)
    L = C.sample_lkj_cholesky(key, K, eta=2.0)
    R = C.cholesky_to_corr(L)
    # Unit diagonal:
    assert jnp.allclose(jnp.diag(R), jnp.ones(K), atol=1e-8)
    # PSD: smallest eigenvalue ≥ 0 (within tolerance).
    eigs = jnp.linalg.eigvalsh(R)
    assert eigs.min() > -1e-8


@pytest.mark.unit
def test_exchangeable_structure() -> None:
    """Exchangeable correlation: off-diagonal is constant rho, diagonal is one."""
    rho = 0.4
    R = C.Exchangeable(K=5, rho=rho).matrix()
    assert jnp.allclose(jnp.diag(R), jnp.ones(5))
    off = R - jnp.diag(jnp.diag(R))
    # Every off-diagonal entry equals rho.
    assert jnp.allclose(off + rho * jnp.eye(5), rho * jnp.ones((5, 5)))


@pytest.mark.unit
def test_one_factor_unit_diagonal() -> None:
    """One-factor structure has unit diagonal by construction."""
    loadings = jnp.array([0.6, 0.5, 0.4, 0.3, 0.2])
    R = C.OneFactor(K=5, loadings=loadings).matrix()
    assert jnp.allclose(jnp.diag(R), jnp.ones(5), atol=1e-10)


@pytest.mark.unit
def test_two_factor_unit_diagonal() -> None:
    """Two-factor structure has unit diagonal by construction."""
    loadings = jnp.array([[0.4, 0.2], [0.3, 0.3], [0.5, 0.1], [0.2, 0.4], [0.1, 0.5]])
    R = C.TwoFactor(K=5, loadings=loadings).matrix()
    assert jnp.allclose(jnp.diag(R), jnp.ones(5), atol=1e-10)
