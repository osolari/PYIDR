"""Pólya-urn auxiliary-atom predictive tests.

Cross-checks the JAX implementation of Eqs. (3.5)–(3.6) of the revised report
against analytic computation, plus Monte-Carlo bucket-frequency checks.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from py_idr.pym.polya_urn import (
    ClusterState,
    predictive_log_weights,
    sample_assignment,
    step,
)

# ---- analytic predictive (Eqs. 3.5–3.6) ---------------------------------------------


@pytest.mark.inference
def test_predictive_log_weights_match_analytic_formula() -> None:
    """Hand-computed log-weights match the JAX output to float64 precision."""
    n_minus_i = jnp.array([3, 1])
    T_minus_i = 2
    alpha = 1.0
    sigma = 0.4
    H = 3
    occupied_log_dens = jnp.array([0.5, 0.2])
    aux_log_dens = jnp.array([0.3, -0.1, 0.4])

    out = predictive_log_weights(
        n_minus_i=n_minus_i,
        T_minus_i=T_minus_i,
        alpha=alpha,
        sigma=sigma,
        H=H,
        occupied_log_density=occupied_log_dens,
        aux_log_density=aux_log_dens,
    )

    expected = jnp.array(
        [
            jnp.log(3.0 - sigma) + 0.5,  # Eq. 3.5, m = 1
            jnp.log(1.0 - sigma) + 0.2,  # Eq. 3.5, m = 2
            jnp.log((alpha + sigma * T_minus_i) / H) + 0.3,  # Eq. 3.6, h = 1
            jnp.log((alpha + sigma * T_minus_i) / H) + (-0.1),
            jnp.log((alpha + sigma * T_minus_i) / H) + 0.4,
        ]
    )
    assert jnp.allclose(out, expected, atol=1e-12)


@pytest.mark.inference
def test_predictive_dp_special_case_sigma_zero() -> None:
    """At σ = 0, the predictive reduces to Neal Algorithm 8 for the DP.

    Occupied weight is n_{m,-i} (no σ subtraction); auxiliary weight is α/H.
    """
    n_minus_i = jnp.array([2, 5])
    T_minus_i = 2
    alpha = 1.5
    sigma = 0.0
    H = 4
    occupied_log_dens = jnp.array([0.0, 0.0])
    aux_log_dens = jnp.zeros(H)

    out = predictive_log_weights(
        n_minus_i=n_minus_i,
        T_minus_i=T_minus_i,
        alpha=alpha,
        sigma=sigma,
        H=H,
        occupied_log_density=occupied_log_dens,
        aux_log_density=aux_log_dens,
    )
    expected = jnp.array(
        [
            jnp.log(2.0),  # n_{1,-i}
            jnp.log(5.0),  # n_{2,-i}
            jnp.log(alpha / H),
            jnp.log(alpha / H),
            jnp.log(alpha / H),
            jnp.log(alpha / H),
        ]
    )
    assert jnp.allclose(out, expected, atol=1e-12)


@pytest.mark.inference
def test_predictive_uses_T_minus_i_not_T() -> None:
    """The auxiliary mass uses T_{-i}, not T — they differ on singleton deletion.

    Bug to catch: an implementation that wrote ((α + σT)/H) instead of
    ((α + σT_{-i})/H). The off-by-one is invisible in the non-singleton path.
    Synthesize T_{-i} = T - 1 (singleton was just deleted) and verify the
    auxiliary multiplier uses T - 1.
    """
    T = 3
    T_minus_i = T - 1  # singleton at z_i was deleted
    alpha = 1.0
    sigma = 0.5
    H = 2
    n_minus_i = jnp.array([4, 4])  # surviving counts
    occupied_log_dens = jnp.zeros(2)
    aux_log_dens = jnp.zeros(H)

    out = predictive_log_weights(
        n_minus_i=n_minus_i,
        T_minus_i=T_minus_i,
        alpha=alpha,
        sigma=sigma,
        H=H,
        occupied_log_density=occupied_log_dens,
        aux_log_density=aux_log_dens,
    )
    expected_aux = jnp.log((alpha + sigma * T_minus_i) / H)
    # The buggy variant would have used T = 3 instead.
    buggy_aux = jnp.log((alpha + sigma * T) / H)
    assert jnp.allclose(out[2], expected_aux)
    assert not jnp.allclose(out[2], buggy_aux)


# ---- Monte Carlo predictive ---------------------------------------------------------


@pytest.mark.inference
def test_monte_carlo_buckets_match_analytic_predictive() -> None:
    """sample_assignment frequencies match the analytic predictive across the (T_{-i}+H) buckets.

    Repeatedly sample with the same log-weight vector and a fresh PRNG key per
    draw; bucket frequencies should match softmax(log_w) within 3 SE on 10⁵
    trials.
    """
    n_minus_i = jnp.array([3, 1, 2])
    T_minus_i = 3
    alpha = 1.0
    sigma = 0.3
    H = 2
    occupied_log_dens = jnp.array([0.0, -0.5, 0.2])
    aux_log_dens = jnp.array([-0.1, 0.3])
    log_w = predictive_log_weights(
        n_minus_i=n_minus_i,
        T_minus_i=T_minus_i,
        alpha=alpha,
        sigma=sigma,
        H=H,
        occupied_log_density=occupied_log_dens,
        aux_log_density=aux_log_dens,
    )

    expected_p = np.asarray(jax.nn.softmax(log_w))
    n_buckets = len(expected_p)

    n_trials = 100_000
    keys = jax.random.split(jax.random.PRNGKey(42), n_trials)
    draws = np.array([sample_assignment(log_w, k) for k in keys])
    counts = np.bincount(draws, minlength=n_buckets)
    empirical_p = counts / n_trials

    # Per-bucket SE of a binomial proportion is sqrt(p(1-p)/n_trials).
    se = np.sqrt(expected_p * (1.0 - expected_p) / n_trials)
    # Allow 3 SE tolerance per bucket.
    assert np.all(np.abs(empirical_p - expected_p) < 3.0 * se + 1e-3)


# ---- step() integration -------------------------------------------------------------


@pytest.mark.inference
def test_step_singleton_deletion_drops_cluster() -> None:
    """When the removed feature was a singleton, T_{-i} = T - 1."""
    state = ClusterState(
        z=jnp.array([0, 0, 0, 1]),
        counts=jnp.array([3, 1, 0, 0]),
        T=2,
        alpha=1.0,
        sigma=0.4,
    )
    occupied_log_dens = jnp.array([0.5, 0.2])
    aux_log_dens = jnp.array([0.0, 0.0])
    key = jax.random.PRNGKey(0)
    _, T_minus_i = step(
        state,
        i=3,
        occupied_log_density_at_U_i=occupied_log_dens,
        aux_log_density_at_U_i=aux_log_dens,
        H=2,
        key=key,
    )
    assert T_minus_i == 1


@pytest.mark.inference
def test_step_non_singleton_keeps_cluster() -> None:
    """When the removed feature is not a singleton in its cluster, T_{-i} = T."""
    state = ClusterState(
        z=jnp.array([0, 0, 0, 1, 1]),
        counts=jnp.array([3, 2, 0, 0]),
        T=2,
        alpha=1.0,
        sigma=0.4,
    )
    occupied_log_dens = jnp.array([0.5, 0.2])
    aux_log_dens = jnp.array([0.0, 0.0])
    key = jax.random.PRNGKey(0)
    _, T_minus_i = step(
        state,
        i=4,
        occupied_log_density_at_U_i=occupied_log_dens,
        aux_log_density_at_U_i=aux_log_dens,
        H=2,
        key=key,
    )
    assert T_minus_i == 2


@pytest.mark.inference
def test_step_returns_index_in_post_removal_indexing() -> None:
    """chosen_index < T_{-i} is occupied; chosen_index >= T_{-i} is auxiliary.

    Stress-tests the index contract by forcing the categorical to almost-surely
    pick an auxiliary atom (giant aux log-density vs tiny occupied densities).
    """
    state = ClusterState(
        z=jnp.array([0, 0, 0, 1, 1]),
        counts=jnp.array([3, 2, 0, 0]),
        T=2,
        alpha=1.0,
        sigma=0.4,
    )
    occupied_log_dens = jnp.array([-1e3, -1e3])  # vanishingly small
    aux_log_dens = jnp.array([0.0, 0.0])
    key = jax.random.PRNGKey(0)
    chosen, T_minus_i = step(
        state,
        i=4,
        occupied_log_density_at_U_i=occupied_log_dens,
        aux_log_density_at_U_i=aux_log_dens,
        H=2,
        key=key,
    )
    assert chosen >= T_minus_i  # auxiliary
