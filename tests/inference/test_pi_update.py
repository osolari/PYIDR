"""Algorithm 1 step 8 — π update tests."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from py_idr.inference.mcmc.pi_update import update_pi


@pytest.mark.inference
def test_pi_update_returns_scalar_in_open_unit() -> None:
    """update_pi returns a 0-D float in (0, 1)."""
    Z = jnp.array([1, 0, 1, 1, 0], dtype=jnp.int32)
    pi = update_pi(jax.random.PRNGKey(0), Z)
    assert pi.shape == ()
    assert 0.0 < float(pi) < 1.0


@pytest.mark.inference
def test_pi_update_posterior_mean_matches_conjugate_formula() -> None:
    """Empirical π posterior mean matches (1 + n_1) / (2 + n) on a 5000-draw MC."""
    n1, n0 = 30, 70
    Z = jnp.concatenate([jnp.ones(n1), jnp.zeros(n0)]).astype(jnp.int32)
    expected_mean = (1.0 + n1) / (2.0 + n1 + n0)
    keys = jax.random.split(jax.random.PRNGKey(0), 5000)
    samples = jnp.stack([update_pi(k, Z) for k in keys])
    assert pytest.approx(float(samples.mean()), abs=5e-3) == expected_mean


@pytest.mark.inference
def test_pi_update_with_all_zero_Z_is_beta_1_n_plus_1() -> None:
    """When every Z=0, posterior is Beta(1, 1+n) with mean 1/(2+n)."""
    n = 100
    Z = jnp.zeros(n, dtype=jnp.int32)
    expected_mean = 1.0 / (2.0 + n)
    keys = jax.random.split(jax.random.PRNGKey(1), 5000)
    samples = jnp.stack([update_pi(k, Z) for k in keys])
    assert pytest.approx(float(samples.mean()), abs=5e-3) == expected_mean


@pytest.mark.inference
def test_pi_update_with_all_one_Z_is_beta_n_plus_1_1() -> None:
    """When every Z=1, posterior is Beta(1+n, 1) with mean (1+n)/(2+n)."""
    n = 100
    Z = jnp.ones(n, dtype=jnp.int32)
    expected_mean = (1.0 + n) / (2.0 + n)
    keys = jax.random.split(jax.random.PRNGKey(2), 5000)
    samples = jnp.stack([update_pi(k, Z) for k in keys])
    assert pytest.approx(float(samples.mean()), abs=5e-3) == expected_mean


@pytest.mark.inference
def test_pi_update_rejects_nonpositive_prior() -> None:
    """Prior shape parameters must be strictly positive."""
    Z = jnp.array([1, 0, 1], dtype=jnp.int32)
    with pytest.raises(ValueError, match="positive"):
        update_pi(jax.random.PRNGKey(0), Z, prior_a=0.0, prior_b=1.0)
    with pytest.raises(ValueError, match="positive"):
        update_pi(jax.random.PRNGKey(0), Z, prior_a=1.0, prior_b=-1.0)


@pytest.mark.inference
def test_pi_update_with_jeffreys_prior() -> None:
    """Jeffreys prior Beta(0.5, 0.5) shifts the posterior mean appropriately."""
    n1, n0 = 20, 30
    Z = jnp.concatenate([jnp.ones(n1), jnp.zeros(n0)]).astype(jnp.int32)
    # Posterior is Beta(0.5 + n1, 0.5 + n0); mean = (0.5 + n1) / (1 + n)
    expected_mean = (0.5 + n1) / (1.0 + n1 + n0)
    keys = jax.random.split(jax.random.PRNGKey(3), 5000)
    samples = jnp.stack([update_pi(k, Z, prior_a=0.5, prior_b=0.5) for k in keys])
    assert pytest.approx(float(samples.mean()), abs=5e-3) == expected_mean
