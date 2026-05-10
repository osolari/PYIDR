"""NUTS atom-parameter updates: synthetic-data recovery on the canonical baselines.

Marked ``@pytest.mark.slow`` because each NUTS chain spends real wall-clock
on warmup. Run only when explicitly selected:

    pytest -m "inference and slow" tests/inference/test_nuts_atoms.py
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy import stats as sps

from py_idr.inference.mcmc.nuts_atoms import (
    update_clayton,
    update_exchangeable_gaussian,
    update_gumbel_k2,
)

# ---- exchangeable Gaussian ----------------------------------------------------------


def _simulate_gaussian_copula(K: int, n: int, rho: float, seed: int) -> jnp.ndarray:
    """Draw from an exchangeable Gaussian copula with the given rho."""
    rng = np.random.default_rng(seed)
    R = (1.0 - rho) * np.eye(K) + rho * np.ones((K, K))
    z = rng.multivariate_normal(np.zeros(K), R, size=n)
    return jnp.asarray(sps.norm.cdf(z))


@pytest.mark.inference
@pytest.mark.slow
@pytest.mark.parametrize("rho_true", [0.4, 0.7, 0.9])
def test_nuts_gaussian_recovers_rho(rho_true: float) -> None:
    """NUTS posterior mean is within 0.05 of the truth on K=4, n=500 data; 0 divergences."""
    K = 4
    n = 500
    U = _simulate_gaussian_copula(K, n, rho_true, seed=0)

    # Run a longer chain and average post-warmup samples for the recovery check.
    key = jax.random.PRNGKey(0)
    rhos: list[float] = []
    sub_keys = jax.random.split(key, 5)
    rho_curr = 0.5
    total_div = 0
    for k in sub_keys:
        rho_curr, diag = update_exchangeable_gaussian(
            k, U, rho_init=rho_curr, n_warmup=200, n_samples=200
        )
        rhos.append(rho_curr)
        total_div += diag["num_divergences"]
    rho_estimate = float(np.mean(rhos))
    assert abs(rho_estimate - rho_true) < 0.05
    # NUTS at target_accept_prob=0.95 should give 0 divergences on this smooth posterior.
    assert total_div == 0


@pytest.mark.inference
def test_nuts_gaussian_returns_value_in_unit_interval() -> None:
    """Single-step recovery: any output rho is in (0, 1)."""
    U = _simulate_gaussian_copula(K=2, n=100, rho=0.5, seed=1)
    rho, _ = update_exchangeable_gaussian(
        jax.random.PRNGKey(0), U, rho_init=0.5, n_warmup=50, n_samples=1
    )
    assert 0.0 < rho < 1.0


# ---- Clayton -----------------------------------------------------------------------


def _simulate_clayton_bivariate(n: int, theta: float, seed: int) -> jnp.ndarray:
    """Bivariate Clayton via the conditional-inversion sampler."""
    rng = np.random.default_rng(seed)
    v = rng.uniform(size=n)
    w = rng.uniform(size=n)
    u1 = v
    u2 = (((w ** (-theta / (1.0 + theta))) - 1.0) * v ** (-theta) + 1.0) ** (-1.0 / theta)
    return jnp.stack([jnp.asarray(u1), jnp.asarray(u2)], axis=1)


@pytest.mark.inference
@pytest.mark.slow
@pytest.mark.parametrize("theta_true", [0.5, 2.0])
def test_nuts_clayton_recovers_theta(theta_true: float) -> None:
    """NUTS recovers Clayton θ within 25% on n=500 bivariate data; 0 divergences.

    Restricted to moderate θ. At very high θ ≥ 5 the Clayton posterior is
    sharply peaked near the data and the inverse-CDF synthetic sampler used
    here can produce numerically extreme draws that propagate Infs through
    the log-density evaluation. The §4.6 stress tests will exercise high-θ
    cases with proper bounded-support synthesis.
    """
    n = 500
    U = _simulate_clayton_bivariate(n, theta_true, seed=2)
    key = jax.random.PRNGKey(0)
    sub_keys = jax.random.split(key, 5)
    theta_curr = 1.0
    thetas: list[float] = []
    total_div = 0
    for k in sub_keys:
        theta_curr, diag = update_clayton(k, U, theta_init=theta_curr, n_warmup=200, n_samples=200)
        thetas.append(theta_curr)
        total_div += diag["num_divergences"]
    theta_estimate = float(np.mean(thetas))
    # Clayton θ has heavier-tailed posterior than Gaussian ρ; allow 25% relative error.
    assert abs(theta_estimate - theta_true) / theta_true < 0.25
    assert total_div == 0


@pytest.mark.inference
def test_nuts_clayton_returns_positive_theta() -> None:
    """update_clayton output is always > 0."""
    U = _simulate_clayton_bivariate(n=100, theta=1.5, seed=3)
    theta, _ = update_clayton(jax.random.PRNGKey(0), U, theta_init=1.0, n_warmup=50, n_samples=1)
    assert theta > 0.0


# ---- Gumbel K=2 --------------------------------------------------------------------


def _simulate_gumbel_bivariate(n: int, theta: float, seed: int) -> jnp.ndarray:
    """Bivariate Gumbel via the conditional-inversion sampler.

    Solves $C_2(u_2 \\mid u_1) = w$ for $u_2$ where the conditional CDF
    follows from the Archimedean generator. We use a simple bisection on the
    monotone conditional, accurate enough for the recovery test.
    """
    rng = np.random.default_rng(seed)
    v = rng.uniform(size=n)
    w = rng.uniform(size=n)
    out = np.empty((n, 2))
    out[:, 0] = v

    def conditional_C2_given_C1(u2: float, u1: float, theta: float) -> float:
        """∂C(u1, u2)/∂u1 — the conditional CDF used for inverse sampling."""
        a, b = -np.log(u1), -np.log(u2)
        S = a**theta + b**theta
        return (
            np.exp(-(S ** (1.0 / theta))) * (S ** (1.0 / theta - 1.0)) * (a ** (theta - 1.0)) / u1
        )

    for i in range(n):
        # Bisection on u2 ∈ (eps, 1 - eps).
        lo, hi = 1e-6, 1.0 - 1e-6
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            cond = conditional_C2_given_C1(mid, v[i], theta)
            if cond < w[i]:
                lo = mid
            else:
                hi = mid
        out[i, 1] = 0.5 * (lo + hi)
    return jnp.asarray(out)


@pytest.mark.inference
@pytest.mark.slow
def test_nuts_gumbel_recovers_theta() -> None:
    """NUTS recovers bivariate Gumbel θ within 0.2 on n=500 data."""
    theta_true = 1.7
    n = 500
    U = _simulate_gumbel_bivariate(n, theta_true, seed=4)
    key = jax.random.PRNGKey(0)
    sub_keys = jax.random.split(key, 5)
    theta_curr = 1.5
    thetas: list[float] = []
    total_div = 0
    for k in sub_keys:
        theta_curr, diag = update_gumbel_k2(
            k, U, theta_init=theta_curr, n_warmup=200, n_samples=200
        )
        thetas.append(theta_curr)
        total_div += diag["num_divergences"]
    theta_estimate = float(np.mean(thetas))
    assert abs(theta_estimate - theta_true) < 0.2
    assert total_div == 0


@pytest.mark.inference
def test_nuts_gumbel_rejects_invalid_theta_init() -> None:
    """update_gumbel_k2 refuses theta_init <= 1."""
    U = jnp.full((10, 2), 0.5)
    with pytest.raises(ValueError, match="theta_init"):
        update_gumbel_k2(jax.random.PRNGKey(0), U, theta_init=1.0)


@pytest.mark.inference
def test_nuts_gumbel_returns_theta_above_one() -> None:
    """Output is always > 1 — the prior support excludes the independence boundary."""
    rng = np.random.default_rng(99)
    U = jnp.asarray(rng.uniform(0.05, 0.95, size=(100, 2)))
    theta, _ = update_gumbel_k2(jax.random.PRNGKey(0), U, theta_init=1.5, n_warmup=50, n_samples=1)
    assert theta > 1.0
