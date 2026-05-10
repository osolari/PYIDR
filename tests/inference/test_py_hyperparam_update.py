"""Pitman-Yor (α, σ) update tests (Algorithm 1 step 7 + EPPF closed form)."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.special import gammaln

from py_idr.inference.mcmc.py_hyperparam_update import update_py_hyperparams
from py_idr.pym.eppf import log_eppf

# ---- EPPF closed form ---------------------------------------------------------------


@pytest.mark.unit
def test_eppf_at_sigma_zero_matches_dp_analytical() -> None:
    """At σ=0 the PY EPPF reduces to the DP EPPF analytically.

    DP: T log α + log Γ(α) - log Γ(α + n) + Σ_m log Γ(n_m).
    """
    T = 4
    cluster_sizes = jnp.array([10, 20, 5, 15], dtype=jnp.int32)
    n = 50
    alpha = 2.0
    py_value = float(log_eppf(jnp.asarray(alpha), jnp.asarray(0.0), cluster_sizes, n))
    dp_expected = (
        T * np.log(alpha)
        + gammaln(alpha)
        - gammaln(alpha + n)
        + np.sum(gammaln(np.asarray(cluster_sizes)))
    )
    assert pytest.approx(py_value, abs=1e-8) == dp_expected


@pytest.mark.unit
def test_eppf_T_eq_1_drops_alpha_jsigma_term() -> None:
    """With one cluster, the j=1..T-1 product is empty — only marginal factors remain."""
    cluster_sizes = jnp.array([20], dtype=jnp.int32)
    n = 20
    alpha = jnp.asarray(1.5)
    sigma = jnp.asarray(0.4)
    out = float(log_eppf(alpha, sigma, cluster_sizes, n))
    # Closed form when T = 1: log Γ(α+1) - log Γ(α+n) + log Γ(n - σ) - log Γ(1 - σ).
    expected = (
        gammaln(float(alpha) + 1.0)
        - gammaln(float(alpha) + n)
        + gammaln(20 - float(sigma))
        - gammaln(1.0 - float(sigma))
    )
    assert pytest.approx(out, abs=1e-8) == expected


@pytest.mark.unit
def test_eppf_is_differentiable_in_alpha_sigma() -> None:
    """log_eppf is differentiable — used by NUTS for HMC gradients."""
    cluster_sizes = jnp.array([3, 4, 5], dtype=jnp.int32)
    grad_fn = jax.grad(
        lambda a, s: log_eppf(a, s, cluster_sizes, n=12),
        argnums=(0, 1),
    )
    g_alpha, g_sigma = grad_fn(jnp.asarray(1.0), jnp.asarray(0.3))
    assert jnp.isfinite(g_alpha)
    assert jnp.isfinite(g_sigma)


@pytest.mark.unit
def test_eppf_monotone_in_alpha_for_many_clusters() -> None:
    """At many clusters, larger α gives larger log p (DP intuition)."""
    cluster_sizes = jnp.ones(20, dtype=jnp.int32) * 5  # 20 clusters of size 5
    n = 100
    log_p_low = float(log_eppf(jnp.asarray(0.5), jnp.asarray(0.0), cluster_sizes, n))
    log_p_high = float(log_eppf(jnp.asarray(5.0), jnp.asarray(0.0), cluster_sizes, n))
    assert log_p_high > log_p_low


@pytest.mark.unit
def test_eppf_monotone_in_sigma_for_many_small_clusters() -> None:
    """Many small clusters → larger σ should give larger log p (PY power-law preference)."""
    cluster_sizes = jnp.ones(30, dtype=jnp.int32) * 2  # 30 tiny clusters
    n = 60
    log_p_dp = float(log_eppf(jnp.asarray(1.0), jnp.asarray(0.0), cluster_sizes, n))
    log_p_py = float(log_eppf(jnp.asarray(1.0), jnp.asarray(0.5), cluster_sizes, n))
    assert log_p_py > log_p_dp


# ---- update_py_hyperparams — basic sanity ------------------------------------------


@pytest.mark.inference
def test_update_returns_alpha_positive_sigma_in_open_unit() -> None:
    """The returned (α, σ) live in (0, ∞) × [0, 1)."""
    cluster_sizes = jnp.array([5, 10, 7], dtype=jnp.int32)
    (alpha, sigma), diag = update_py_hyperparams(
        jax.random.PRNGKey(0), cluster_sizes, n=22, n_warmup=50, n_samples=1
    )
    assert alpha > 0.0
    assert 0.0 <= sigma < 1.0
    # Single-sample warmup should produce no divergences on this well-conditioned posterior.
    assert diag["num_divergences"] == 0


@pytest.mark.inference
def test_update_rejects_empty_cluster_sizes() -> None:
    """Empty cluster_sizes is a caller error — raises."""
    with pytest.raises(ValueError, match="non-empty"):
        update_py_hyperparams(jax.random.PRNGKey(0), jnp.array([], dtype=jnp.int32), n=0)


@pytest.mark.inference
def test_update_rejects_zero_or_negative_cluster_sizes() -> None:
    """Strictly positive cluster sizes — drop empty clusters before calling."""
    with pytest.raises(ValueError, match="strictly positive"):
        update_py_hyperparams(jax.random.PRNGKey(0), jnp.array([5, 0, 3], dtype=jnp.int32), n=8)


@pytest.mark.inference
def test_update_rejects_invalid_init() -> None:
    """alpha_init > 0 and sigma_init in (0, 1)."""
    cluster_sizes = jnp.array([5, 5], dtype=jnp.int32)
    with pytest.raises(ValueError, match="alpha_init"):
        update_py_hyperparams(jax.random.PRNGKey(0), cluster_sizes, n=10, alpha_init=0.0)
    with pytest.raises(ValueError, match="sigma_init"):
        update_py_hyperparams(jax.random.PRNGKey(0), cluster_sizes, n=10, sigma_init=1.0)


# ---- recovery: NUTS finds a sensible posterior --------------------------------------


@pytest.mark.inference
@pytest.mark.slow
def test_update_chain_concentrates_on_high_alpha_for_many_clusters() -> None:
    """A partition with many small clusters has a posterior over α concentrated above 1.

    Synthesise n=100 features split into 25 clusters of size 4 each — a configuration
    that DP/PY explains better with α >= 1. Averaging post-warmup samples over a few
    short chains, the chain mean should sit comfortably above 0.5.
    """
    cluster_sizes = jnp.ones(25, dtype=jnp.int32) * 4
    n = 100
    keys = jax.random.split(jax.random.PRNGKey(0), 5)
    alpha_curr, sigma_curr = 1.0, 0.1
    alphas: list[float] = []
    total_div = 0
    for k in keys:
        (alpha_curr, sigma_curr), diag = update_py_hyperparams(
            k,
            cluster_sizes,
            n,
            alpha_init=alpha_curr,
            sigma_init=max(sigma_curr, 0.01),
            n_warmup=200,
            n_samples=200,
        )
        alphas.append(alpha_curr)
        total_div += diag["num_divergences"]
    alpha_mean = float(np.mean(alphas))
    assert alpha_mean > 0.5
    assert total_div == 0


@pytest.mark.inference
@pytest.mark.slow
def test_update_chain_concentrates_on_low_alpha_for_few_large_clusters() -> None:
    """A partition with one giant cluster has α posterior pulled toward 0.

    Synthesise n=100, T=2 with sizes (95, 5). DP/PY favours small α here (few
    clusters at large n implies low concentration). Average several chains.
    """
    cluster_sizes = jnp.array([95, 5], dtype=jnp.int32)
    n = 100
    keys = jax.random.split(jax.random.PRNGKey(1), 5)
    alpha_curr, sigma_curr = 1.0, 0.1
    alphas: list[float] = []
    for k in keys:
        (alpha_curr, sigma_curr), _ = update_py_hyperparams(
            k,
            cluster_sizes,
            n,
            alpha_init=alpha_curr,
            sigma_init=max(sigma_curr, 0.01),
            n_warmup=200,
            n_samples=200,
        )
        alphas.append(alpha_curr)
    alpha_mean = float(np.mean(alphas))
    # Posterior strongly pulls α below 1 for this configuration.
    assert alpha_mean < 1.0
