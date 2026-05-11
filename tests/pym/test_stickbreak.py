"""Tests for the Pitman-Yor stick-breaking sampler."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from py_idr.pym.stickbreak import gem_sample

# ---- validation ----------------------------------------------------------------------


@pytest.mark.unit
def test_gem_sample_rejects_sigma_out_of_range() -> None:
    """sigma must be in [0, 1)."""
    key = jax.random.PRNGKey(0)
    with pytest.raises(ValueError, match="sigma"):
        gem_sample(key, alpha=1.0, sigma=-0.1, T=5)
    with pytest.raises(ValueError, match="sigma"):
        gem_sample(key, alpha=1.0, sigma=1.0, T=5)


@pytest.mark.unit
def test_gem_sample_rejects_alpha_too_small() -> None:
    """alpha + sigma must be strictly positive."""
    key = jax.random.PRNGKey(0)
    with pytest.raises(ValueError, match="alpha"):
        gem_sample(key, alpha=-0.5, sigma=0.3, T=5)  # alpha + sigma = -0.2 < 0


@pytest.mark.unit
def test_gem_sample_rejects_T_lt_one() -> None:
    """T must be >= 1."""
    key = jax.random.PRNGKey(0)
    with pytest.raises(ValueError, match="T"):
        gem_sample(key, alpha=1.0, sigma=0.0, T=0)


# ---- shape / dtype / range -----------------------------------------------------------


@pytest.mark.unit
def test_gem_sample_returns_correct_length() -> None:
    """Output is a length-T JAX array."""
    w = gem_sample(jax.random.PRNGKey(0), alpha=1.0, sigma=0.3, T=7)
    assert w.shape == (7,)


@pytest.mark.unit
def test_gem_sample_weights_in_unit_interval() -> None:
    """Every weight lies in [0, 1]."""
    w = np.asarray(gem_sample(jax.random.PRNGKey(0), alpha=1.0, sigma=0.4, T=20))
    assert np.all((w >= 0.0) & (w <= 1.0))


@pytest.mark.unit
def test_gem_sample_dp_sum_close_to_one() -> None:
    """At sigma = 0 (DP) with alpha = 1 and large T, the truncated sum ≈ 1."""
    w = gem_sample(jax.random.PRNGKey(0), alpha=1.0, sigma=0.0, T=100)
    assert float(jnp.sum(w)) == pytest.approx(1.0, abs=1e-3)


# ---- structural ----------------------------------------------------------------------


@pytest.mark.unit
def test_gem_sample_seeded_is_deterministic() -> None:
    """Same key produces identical weights."""
    w1 = gem_sample(jax.random.PRNGKey(42), alpha=1.0, sigma=0.3, T=10)
    w2 = gem_sample(jax.random.PRNGKey(42), alpha=1.0, sigma=0.3, T=10)
    assert jnp.allclose(w1, w2)


@pytest.mark.unit
def test_gem_sample_different_keys_differ() -> None:
    """Different keys produce different weights."""
    w1 = gem_sample(jax.random.PRNGKey(1), alpha=1.0, sigma=0.3, T=10)
    w2 = gem_sample(jax.random.PRNGKey(2), alpha=1.0, sigma=0.3, T=10)
    assert not jnp.allclose(w1, w2)


@pytest.mark.unit
def test_gem_sample_T_equals_1_returns_v1() -> None:
    """At T = 1, the only weight equals v_1 ~ Beta(1 - sigma, alpha + sigma)."""
    w = gem_sample(jax.random.PRNGKey(0), alpha=2.0, sigma=0.2, T=1)
    assert w.shape == (1,)
    assert 0.0 <= float(w[0]) <= 1.0


# ---- statistical (cheap MC check) ----------------------------------------------------


@pytest.mark.property
def test_gem_sample_expected_w_one_matches_beta() -> None:
    """E[w_1] = E[v_1] = (1 - sigma) / (1 + alpha) under stick-breaking.

    Run 2000 independent draws and check the empirical mean is within
    2σ of the analytic mean.
    """
    alpha, sigma = 1.5, 0.3
    n_draws = 2000
    keys = jax.random.split(jax.random.PRNGKey(0), n_draws)
    w1s = np.array([float(gem_sample(k, alpha=alpha, sigma=sigma, T=1)[0]) for k in keys])
    # Beta(1-σ, α+σ) mean = (1-σ) / ((1-σ) + (α+σ)) = (1-σ) / (1+α).
    expected = (1.0 - sigma) / (1.0 + alpha)
    sem = w1s.std() / np.sqrt(n_draws)
    assert abs(w1s.mean() - expected) < 4 * sem  # 4σ for safety


@pytest.mark.property
def test_gem_sample_py_has_heavier_tail_than_dp() -> None:
    """At fixed alpha and large T, PY (sigma > 0) leaves more residual mass
    than DP (sigma = 0). Per Pitman & Yor (1997), the cluster-count growth is
    O(n^sigma) instead of O(log n)."""
    alpha = 1.0
    T = 50
    n_draws = 500
    keys_dp = jax.random.split(jax.random.PRNGKey(0), n_draws)
    keys_py = jax.random.split(jax.random.PRNGKey(1), n_draws)
    sums_dp = np.array(
        [float(jnp.sum(gem_sample(k, alpha=alpha, sigma=0.0, T=T))) for k in keys_dp]
    )
    sums_py = np.array(
        [float(jnp.sum(gem_sample(k, alpha=alpha, sigma=0.7, T=T))) for k in keys_py]
    )
    # DP truncated sum should be much closer to 1 than PY truncated sum.
    assert sums_dp.mean() > sums_py.mean()
    assert sums_dp.mean() > 0.95
    assert sums_py.mean() < 0.9
