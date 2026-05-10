"""Property tests: PLOD holds for every registered atomic type on prior-supported parameters."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from py_idr.algebra.correlation import (
    Exchangeable,
    OneFactor,
    TwoFactor,
    corr_to_cholesky,
    cov_to_corr,
)
from py_idr.algebra.plod import assert_plod, holds, plod_with_positive_pairwise
from py_idr.copulas.clayton import ClaytonCopula
from py_idr.copulas.gaussian import GaussianCopula
from py_idr.copulas.gumbel import GumbelCopula
from py_idr.copulas.independence import IndependenceCopula
from py_idr.copulas.student_t import StudentTCopula


@pytest.mark.property
@pytest.mark.parametrize("K", [2, 3, 5])
@given(rho=st.floats(min_value=0.2, max_value=0.95))
@settings(
    max_examples=15,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
def test_gaussian_with_positive_rho_satisfies_plod(K: int, rho: float) -> None:
    """A Gaussian copula with all-positive exchangeable correlation is PLOD.

    SciPy's multivariate_normal.cdf has ~5e-4 absolute error at K=5 and worse at
    higher dimensions. We sweep K up to 5 with Hypothesis and provide a single
    high-rho K=10 spot-check below where the slack comfortably exceeds the noise.
    """
    R = Exchangeable(K=K, rho=rho).matrix()
    L = corr_to_cholesky(R)
    cop = GaussianCopula(L)
    u_grid = jnp.linspace(0.2, 0.9, 36)
    assert bool(jnp.all(holds(cop, K, u_grid, tol=2e-3)))


@pytest.mark.property
def test_gaussian_K10_high_rho_satisfies_plod() -> None:
    """Spot-check at K=10 with rho=0.7 — slack >> SciPy mvn-cdf noise floor."""
    K = 10
    R = Exchangeable(K=K, rho=0.7).matrix()
    L = corr_to_cholesky(R)
    cop = GaussianCopula(L)
    u_grid = jnp.linspace(0.3, 0.8, 11)
    assert bool(jnp.all(holds(cop, K, u_grid, tol=5e-3)))


@pytest.mark.property
@pytest.mark.parametrize("K", [2, 5])
@given(rho=st.floats(min_value=0.05, max_value=0.95))
@settings(
    max_examples=8,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
def test_t5_with_positive_rho_satisfies_plod(K: int, rho: float) -> None:
    """A t_5 copula with all-positive exchangeable correlation is PLOD."""
    R = Exchangeable(K=K, rho=rho).matrix()
    L = corr_to_cholesky(R)
    cop = StudentTCopula(L)
    assert bool(jnp.all(holds(cop, K)))


@pytest.mark.property
@pytest.mark.parametrize("K", [2, 3, 5, 10])
@given(theta=st.floats(min_value=0.5, max_value=10.0))
@settings(max_examples=15, deadline=None)
def test_clayton_satisfies_plod_for_all_positive_theta(K: int, theta: float) -> None:
    """Clayton with theta > 0 is PLOD.

    Lower bound on theta keeps the slack above float-tolerance — at theta -> 0 the
    Clayton copula tends to the independence copula and the slack vanishes, which
    is correct math but trips the strict-inequality test on a finite grid.
    """
    cop = ClaytonCopula(theta=theta, K=K)
    assert bool(jnp.all(holds(cop, K)))


@pytest.mark.property
@given(theta=st.floats(min_value=1.05, max_value=10.0))
@settings(max_examples=15, deadline=None)
def test_gumbel_satisfies_plod_for_all_theta_gt_one(theta: float) -> None:
    """Gumbel with theta > 1 is PLOD (bivariate; multivariate not implemented yet)."""
    cop = GumbelCopula(theta=theta, K=2)
    assert bool(jnp.all(holds(cop, 2)))


@pytest.mark.property
def test_independence_copula_fails_plod() -> None:
    """The independence copula has zero PLOD slack — assert_plod must raise."""
    cop = IndependenceCopula(K=4)
    with pytest.raises(ValueError, match="PLOD violated"):
        assert_plod(cop, 4)


@pytest.mark.property
@pytest.mark.parametrize("K", [3, 5])
def test_one_factor_gaussian_with_random_loadings_is_plod(K: int) -> None:  # noqa: D103
    """A one-factor Gaussian with random positive loadings in (0, 1) is PLOD."""
    key = jax.random.PRNGKey(7)
    loadings = 0.4 + 0.4 * jax.random.uniform(key, (K,))
    R = OneFactor(K=K, loadings=loadings).matrix()
    L = corr_to_cholesky(R)
    cop = GaussianCopula(L)
    u_grid = jnp.linspace(0.2, 0.9, 16)
    assert bool(jnp.all(holds(cop, K, u_grid, tol=2e-3)))


# ---- PSD-then-normalize parameterization (revised §3.2.2) -----------------------------


@pytest.mark.property
@pytest.mark.parametrize("K", [2, 3, 5, 10])
def test_one_factor_psd_normalize_unit_diagonal(K: int) -> None:
    """OneFactor.matrix() always has unit diagonal (cov_to_corr enforces it)."""
    key = jax.random.PRNGKey(11)
    loadings = 0.3 + 0.5 * jax.random.uniform(key, (K,))
    R = OneFactor(K=K, loadings=loadings).matrix()
    assert jnp.allclose(jnp.diag(R), jnp.ones(K), atol=1e-6)


@pytest.mark.property
@pytest.mark.parametrize("K", [2, 3, 5, 10])
def test_one_factor_psd_normalize_is_psd(K: int) -> None:
    """OneFactor matrix is PSD by construction (Σ = D + λλᵀ is PSD; cov_to_corr preserves)."""
    key = jax.random.PRNGKey(12)
    loadings = 0.3 + 0.5 * jax.random.uniform(key, (K,))
    R = OneFactor(K=K, loadings=loadings).matrix()
    eigs = jnp.linalg.eigvalsh(R)
    assert eigs.min() > -1e-8


@pytest.mark.property
@pytest.mark.parametrize("K", [3, 5, 10])
def test_two_factor_psd_normalize_unit_diagonal(K: int) -> None:
    """TwoFactor.matrix() always has unit diagonal."""
    key = jax.random.PRNGKey(13)
    loadings = jax.random.uniform(key, (K, 2)) * 0.4 + 0.1
    R = TwoFactor(K=K, loadings=loadings).matrix()
    assert jnp.allclose(jnp.diag(R), jnp.ones(K), atol=1e-6)


@pytest.mark.property
@pytest.mark.parametrize("K", [3, 5, 10])
def test_two_factor_psd_normalize_is_psd(K: int) -> None:
    """TwoFactor matrix is PSD by construction."""
    key = jax.random.PRNGKey(14)
    loadings = jax.random.uniform(key, (K, 2)) * 0.4 + 0.1
    R = TwoFactor(K=K, loadings=loadings).matrix()
    eigs = jnp.linalg.eigvalsh(R)
    assert eigs.min() > -1e-8


@pytest.mark.property
def test_one_factor_with_idiosyncratic_diagonal() -> None:
    """OneFactor accepts a free idiosyncratic diagonal; cov_to_corr absorbs the scale."""
    K = 4
    loadings = jnp.array([0.5, 0.5, 0.5, 0.5])
    # Two different idiosyncratic diagonals — both should normalise to unit-diagonal R.
    R1 = OneFactor(K=K, loadings=loadings).matrix()
    R2 = OneFactor(K=K, loadings=loadings, idiosyncratic=2.5 * jnp.ones(K)).matrix()
    assert jnp.allclose(jnp.diag(R1), jnp.ones(K))
    assert jnp.allclose(jnp.diag(R2), jnp.ones(K))


# ---- positive-pairwise check (revised §3.2.2 PLOD scoping) ---------------------------


@pytest.mark.property
def test_plod_positive_pairwise_passes_common_sign_loadings() -> None:
    """Common-sign loadings produce all-positive pairwise correlations."""
    loadings = jnp.array([0.6, 0.5, 0.4, 0.3])
    R = OneFactor(K=4, loadings=loadings).matrix()
    assert plod_with_positive_pairwise(R) is True


@pytest.mark.property
def test_plod_positive_pairwise_fails_mixed_sign_loadings() -> None:
    """Mixed-sign loadings produce at least one negative off-diagonal — must be rejected."""
    loadings = jnp.array([0.6, -0.5, 0.4, 0.3])
    R = OneFactor(K=4, loadings=loadings).matrix()
    assert plod_with_positive_pairwise(R) is False


@pytest.mark.property
def test_plod_positive_pairwise_passes_exchangeable_with_positive_rho() -> None:
    """Exchangeable with rho > 0 has every off-diagonal equal to rho — passes."""
    R = Exchangeable(K=5, rho=0.4).matrix()
    assert plod_with_positive_pairwise(R) is True


@pytest.mark.property
def test_plod_positive_pairwise_fails_when_any_correlation_is_zero() -> None:
    """A correlation matrix with even one zero off-diagonal entry must fail (strict positive)."""
    R = jnp.eye(3).at[0, 1].set(0.0).at[1, 0].set(0.0).at[0, 2].set(0.5).at[2, 0].set(0.5)
    R = R.at[1, 2].set(0.5).at[2, 1].set(0.5)
    assert plod_with_positive_pairwise(R) is False


# ---- cov_to_corr direct tests --------------------------------------------------------


@pytest.mark.property
def test_cov_to_corr_unit_diagonal() -> None:
    """cov_to_corr returns a unit-diagonal R for any positive-definite Σ."""
    Sigma = jnp.array([[2.5, 0.6, 0.2], [0.6, 1.4, 0.3], [0.2, 0.3, 0.9]])
    R = cov_to_corr(Sigma)
    assert jnp.allclose(jnp.diag(R), jnp.ones(3))


@pytest.mark.property
def test_cov_to_corr_preserves_off_diagonal_signs() -> None:
    """cov_to_corr is a positive scaling; off-diagonal signs are preserved."""
    Sigma = jnp.array([[2.0, -0.3, 0.4], [-0.3, 1.5, 0.1], [0.4, 0.1, 1.0]])
    R = cov_to_corr(Sigma)
    # Same sign pattern off-diagonal.
    assert jnp.sign(R[0, 1]) == jnp.sign(Sigma[0, 1])
    assert jnp.sign(R[0, 2]) == jnp.sign(Sigma[0, 2])
    assert jnp.sign(R[1, 2]) == jnp.sign(Sigma[1, 2])


@pytest.mark.property
def test_cov_to_corr_idempotent_on_correlation_matrix() -> None:
    """Applying cov_to_corr to an already-correlation matrix is a no-op."""
    R = Exchangeable(K=4, rho=0.3).matrix()
    R2 = cov_to_corr(R)
    assert jnp.allclose(R, R2)
