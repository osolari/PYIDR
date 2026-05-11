"""Simulation scenario generator tests."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from py_idr.simulation.scenarios import SimulationResult, simulate_S1


@pytest.mark.unit
def test_simulate_S1_shapes_and_counts() -> None:
    """simulate_S1 produces (X, true_Z) of the requested shapes; counts match π*."""
    K = 3
    n = 100
    sim = simulate_S1(K=K, n=n, pi_star=0.30, rho=0.85, seed=0)
    assert isinstance(sim, SimulationResult)
    assert sim.X.shape == (n, K)
    assert sim.true_Z.shape == (n,)
    assert sim.regime == "S1"
    # n_rep = round(pi_star * n) = 30
    assert int(jnp.sum(sim.true_Z)) == 30
    # K / n properties.
    assert sim.K == K
    assert sim.n == n


@pytest.mark.unit
def test_simulate_S1_is_deterministic_given_seed() -> None:
    """Identical seed → identical (X, true_Z)."""
    sim1 = simulate_S1(K=2, n=50, seed=42)
    sim2 = simulate_S1(K=2, n=50, seed=42)
    assert jnp.allclose(sim1.X, sim2.X)
    assert jnp.array_equal(sim1.true_Z, sim2.true_Z)


@pytest.mark.unit
def test_simulate_S1_different_seeds_differ() -> None:
    """Different seeds produce different data (with overwhelming probability)."""
    sim1 = simulate_S1(K=2, n=50, seed=0)
    sim2 = simulate_S1(K=2, n=50, seed=1)
    assert not jnp.allclose(sim1.X, sim2.X)


@pytest.mark.unit
def test_simulate_S1_reproducible_features_are_correlated() -> None:
    """Reproducible features have higher pairwise correlation than irreproducible."""
    sim = simulate_S1(K=2, n=2000, pi_star=0.30, rho=0.85, seed=0)
    X = np.asarray(sim.X)
    Z = np.asarray(sim.true_Z)
    rep_corr = float(np.corrcoef(X[Z == 1, 0], X[Z == 1, 1])[0, 1])
    irrep_corr = float(np.corrcoef(X[Z == 0, 0], X[Z == 0, 1])[0, 1])
    # Reproducible empirical correlation should be near rho = 0.85; irreproducible near 0.
    assert rep_corr > 0.75
    assert abs(irrep_corr) < 0.10


@pytest.mark.unit
def test_simulate_S1_rejects_invalid_K_n_pi_rho() -> None:
    """Validators on K, n, pi_star, rho."""
    with pytest.raises(ValueError, match="K must be >= 2"):
        simulate_S1(K=1, n=100)
    with pytest.raises(ValueError, match="n must be >= 10"):
        simulate_S1(K=2, n=5)
    with pytest.raises(ValueError, match="pi_star"):
        simulate_S1(K=2, n=50, pi_star=0.0)
    with pytest.raises(ValueError, match="rho"):
        simulate_S1(K=2, n=50, rho=1.0)


@pytest.mark.unit
def test_simulate_S1_features_are_shuffled() -> None:
    """The reproducible and irreproducible features are interleaved by the shuffle."""
    sim = simulate_S1(K=2, n=200, pi_star=0.30, seed=3)
    Z = np.asarray(sim.true_Z)
    # If not shuffled, Z would be [1]*60 + [0]*140 — concentrate at one end.
    # We test that Z=1 features appear both early and late in the index.
    first_half_reps = int(np.sum(Z[:100]))
    second_half_reps = int(np.sum(Z[100:]))
    assert first_half_reps > 0
    assert second_half_reps > 0
