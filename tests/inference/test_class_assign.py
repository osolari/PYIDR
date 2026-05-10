"""Algorithm 1 step 1 — class assignment tests."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from py_idr.inference.mcmc.class_assign import (
    class_assign,
    conditional_class_log_odds,
)


@pytest.mark.inference
def test_log_odds_match_bayes_rule_at_pi_half() -> None:
    """At π = 1/2 with c_0 = 1, log-odds equal log c_1."""
    log_c1 = jnp.array([-1.0, 0.0, 2.5])
    log_odds = conditional_class_log_odds(0.5, log_c1)
    assert jnp.allclose(log_odds, log_c1, atol=1e-6)


@pytest.mark.inference
def test_log_odds_use_log_c0_when_supplied() -> None:
    """A non-trivial c_0 shifts the log-odds by -log c_0(U_i)."""
    log_c1 = jnp.array([0.0, 0.0, 0.0])
    log_c0 = jnp.array([0.0, 1.0, 2.0])
    log_odds = conditional_class_log_odds(0.5, log_c1, log_c0)
    # log π/(1-π) = 0 at π = 0.5; log_odds = log c_1 - log c_0 = -log_c0.
    assert jnp.allclose(log_odds, -log_c0, atol=1e-6)


@pytest.mark.inference
def test_log_odds_handle_extreme_pi() -> None:
    """π near 0 or 1 yields very low / very high log-odds without overflow."""
    log_c1 = jnp.zeros(3)
    log_odds_low = conditional_class_log_odds(1e-6, log_c1)
    log_odds_high = conditional_class_log_odds(1.0 - 1e-6, log_c1)
    assert bool(jnp.all(log_odds_low < -10.0))
    assert bool(jnp.all(log_odds_high > 10.0))
    assert bool(jnp.all(jnp.isfinite(log_odds_low)))
    assert bool(jnp.all(jnp.isfinite(log_odds_high)))


@pytest.mark.inference
def test_class_assign_returns_int32_in_zero_one() -> None:
    """Output is int32 with values in {0, 1}."""
    log_c1 = jnp.array([0.0, -1.0, 1.0])
    z = class_assign(jax.random.PRNGKey(0), 0.5, log_c1)
    assert z.dtype == jnp.int32
    assert bool(jnp.all((z == 0) | (z == 1)))


@pytest.mark.inference
def test_class_assign_extreme_log_c1_almost_deterministic() -> None:
    """At very large positive log c_1, Z ≈ 1; at very negative log c_1, Z ≈ 0."""
    log_c1 = jnp.array([20.0, 20.0, -20.0, -20.0])
    keys = jax.random.split(jax.random.PRNGKey(0), 200)
    zs = jnp.stack([class_assign(k, 0.5, log_c1) for k in keys])
    means = jnp.mean(zs.astype(jnp.float32), axis=0)
    assert means[0] > 0.99 and means[1] > 0.99
    assert means[2] < 0.01 and means[3] < 0.01


@pytest.mark.inference
def test_class_assign_monte_carlo_matches_pi_hat() -> None:
    """Empirical Z=1 frequencies match the analytic π̂_i across a 5000-draw MC."""
    log_c1 = jnp.array([0.0, jnp.log(2.0), jnp.log(0.5)])
    pi = 0.3
    expected_pi_hat = jax.nn.sigmoid(conditional_class_log_odds(pi, log_c1))
    keys = jax.random.split(jax.random.PRNGKey(0), 5000)
    zs = jnp.stack([class_assign(k, pi, log_c1) for k in keys])
    empirical = jnp.mean(zs.astype(jnp.float32), axis=0)
    assert jnp.allclose(empirical, expected_pi_hat, atol=2e-2)
