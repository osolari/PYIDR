"""Property tests: PLOD holds for every registered atomic type on prior-supported parameters."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from stick_idr.algebra.correlation import (
    Exchangeable,
    OneFactor,
    corr_to_cholesky,
)
from stick_idr.algebra.plod import assert_plod, holds
from stick_idr.copulas.clayton import ClaytonCopula
from stick_idr.copulas.gaussian import GaussianCopula
from stick_idr.copulas.gumbel import GumbelCopula
from stick_idr.copulas.independence import IndependenceCopula
from stick_idr.copulas.student_t import StudentTCopula


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
