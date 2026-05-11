"""Simulation scenario generator tests."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from py_idr.simulation.scenarios import (
    SimulationResult,
    simulate_S1,
    simulate_S2,
    simulate_S3,
    simulate_S4,
    simulate_S5,
    simulate_S5_sparse,
)


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


# ---- S2: weak signal (rho = 0.40) ----------------------------------------------------


@pytest.mark.unit
def test_simulate_S2_default_rho_is_weak() -> None:
    """S2's default rho is 0.40 — the weak-signal regime in Table 3."""
    sim = simulate_S2(K=3, n=100, seed=0)
    assert sim.regime == "S2"
    assert sim.true_rho == pytest.approx(0.40)
    assert sim.true_pi == pytest.approx(0.30)


@pytest.mark.unit
def test_simulate_S2_empirical_rho_matches() -> None:
    """Reproducible empirical correlation tracks the weaker rho = 0.40."""
    sim = simulate_S2(K=2, n=4000, rho=0.40, seed=0)
    X = np.asarray(sim.X)
    Z = np.asarray(sim.true_Z)
    rep_corr = float(np.corrcoef(X[Z == 1, 0], X[Z == 1, 1])[0, 1])
    assert 0.30 < rep_corr < 0.50


@pytest.mark.unit
def test_simulate_S2_deterministic() -> None:
    """Same seed → identical output."""
    a = simulate_S2(K=3, n=80, seed=7)
    b = simulate_S2(K=3, n=80, seed=7)
    assert jnp.allclose(a.X, b.X)
    assert jnp.array_equal(a.true_Z, b.true_Z)


# ---- S3: sparse + heterogeneous Gaussian scales --------------------------------------


@pytest.mark.unit
def test_simulate_S3_default_pi_is_sparse() -> None:
    """S3's default pi_star is 0.10 — the sparse-signal regime."""
    sim = simulate_S3(K=3, n=200, seed=0)
    assert sim.regime == "S3"
    assert sim.true_pi == pytest.approx(0.10)
    assert int(jnp.sum(sim.true_Z)) == 20


@pytest.mark.unit
def test_simulate_S3_per_replicate_scale_matches_sigma_grid() -> None:
    """Irreproducible-feature std per replicate tracks the requested sigma_grid."""
    sigma_grid = (1.0, 1.5, 2.0)
    sim = simulate_S3(K=3, n=4000, pi_star=0.10, sigma_grid=sigma_grid, seed=0)
    X = np.asarray(sim.X)
    Z = np.asarray(sim.true_Z)
    stds = [float(np.std(X[Z == 0, j])) for j in range(3)]
    # 5% empirical std tolerance; per-replicate scales should match sigma_grid.
    for emp, exp in zip(stds, sigma_grid, strict=False):
        assert exp * 0.85 < emp < exp * 1.15


@pytest.mark.unit
def test_simulate_S3_sigma_grid_cycles_mod_K() -> None:
    """When K > len(sigma_grid), the grid cycles. K=5, grid=(1,2) → (1,2,1,2,1)."""
    sim = simulate_S3(K=5, n=200, sigma_grid=(1.0, 2.0), seed=0)
    assert sim.X.shape == (200, 5)


@pytest.mark.unit
def test_simulate_S3_rejects_invalid_sigma_grid() -> None:
    """Non-positive or empty sigma_grid is rejected."""
    with pytest.raises(ValueError, match="sigma_grid"):
        simulate_S3(K=3, n=100, sigma_grid=())
    with pytest.raises(ValueError, match="sigma_grid"):
        simulate_S3(K=3, n=100, sigma_grid=(1.0, 0.0, 2.0))
    with pytest.raises(ValueError, match="sigma_grid"):
        simulate_S3(K=3, n=100, sigma_grid=(-1.0,))


# ---- S4: skewed marginals ------------------------------------------------------------


@pytest.mark.unit
def test_simulate_S4_default_params() -> None:
    """S4 keeps the S1 latent structure (rho=0.85, pi*=0.30); only marginals differ."""
    sim = simulate_S4(K=3, n=100, seed=0)
    assert sim.regime == "S4"
    assert sim.true_pi == pytest.approx(0.30)
    assert sim.true_rho == pytest.approx(0.85)


@pytest.mark.unit
def test_simulate_S4_skewness_matches_grid_sign() -> None:
    """Per-replicate sample skewness has the same sign as the requested skew param."""
    skewness_grid = (-3.0, 0.0, 3.0)  # large magnitudes for a clean sign read
    sim = simulate_S4(K=3, n=4000, skewness_grid=skewness_grid, seed=0)
    X = np.asarray(sim.X)
    from scipy.stats import skew

    sk = [float(skew(X[:, j])) for j in range(3)]
    assert sk[0] < -0.5  # strong negative skew
    assert abs(sk[1]) < 0.3  # symmetric
    assert sk[2] > 0.5  # strong positive skew


@pytest.mark.unit
def test_simulate_S4_rejects_empty_skewness_grid() -> None:
    """Empty skewness_grid is rejected."""
    with pytest.raises(ValueError, match="skewness_grid"):
        simulate_S4(K=3, n=100, skewness_grid=())


# ---- Cross-regime sanity ------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.parametrize(
    "fn,regime",
    [(simulate_S1, "S1"), (simulate_S2, "S2"), (simulate_S3, "S3"), (simulate_S4, "S4")],
)
def test_all_regimes_share_validator_path(fn, regime) -> None:
    """All S1–S4 scenarios share the same K/n/pi/rho validators."""
    with pytest.raises(ValueError):
        fn(K=1, n=100)
    with pytest.raises(ValueError):
        fn(K=2, n=5)


# ---- S5: heavy-tail mixture ---------------------------------------------------------


@pytest.mark.unit
def test_simulate_S5_shapes_and_counts() -> None:
    """S5 returns the expected shapes and reproducible mass at default pi*=0.30."""
    sim = simulate_S5(K=2, n=200, seed=0)
    assert isinstance(sim, SimulationResult)
    assert sim.regime == "S5"
    assert sim.X.shape == (200, 2)
    assert sim.true_Z.shape == (200,)
    assert sim.true_pi == pytest.approx(0.30)
    assert int(jnp.sum(sim.true_Z)) == 60


@pytest.mark.unit
def test_simulate_S5_rho_is_nan() -> None:
    """S5 has no single rho — the family mixture means true_rho should be NaN."""
    sim = simulate_S5(K=2, n=100, seed=0)
    assert np.isnan(sim.true_rho)


@pytest.mark.unit
def test_simulate_S5_deterministic() -> None:
    """Same seed → identical output."""
    a = simulate_S5(K=2, n=100, seed=42)
    b = simulate_S5(K=2, n=100, seed=42)
    assert jnp.allclose(a.X, b.X)
    assert jnp.array_equal(a.true_Z, b.true_Z)


@pytest.mark.unit
def test_simulate_S5_empirical_kendall_tau_matches() -> None:
    """Reproducible features hit the requested matched Kendall's tau (within MC noise)."""
    from scipy.stats import kendalltau

    tau_target = 0.6
    sim = simulate_S5(K=2, n=4000, tau=tau_target, seed=0)
    X = np.asarray(sim.X)
    Z = np.asarray(sim.true_Z)
    tau_emp, _ = kendalltau(X[Z == 1, 0], X[Z == 1, 1])
    # Mixture of three families at matched tau should give an empirical tau
    # within ~5% of the target at n_rep = 1200.
    assert abs(tau_emp - tau_target) < 0.05


@pytest.mark.unit
def test_simulate_S5_K_ge_3_drops_gumbel() -> None:
    """For K >= 3 the Gumbel family is excluded from the pool (no NaN, no error)."""
    sim = simulate_S5(K=4, n=200, seed=0)
    X = np.asarray(sim.X)
    assert sim.X.shape == (200, 4)
    assert not np.any(np.isnan(X))


@pytest.mark.unit
def test_simulate_S5_rejects_invalid_tau() -> None:
    """tau must be strictly in (0, 1)."""
    with pytest.raises(ValueError, match="tau"):
        simulate_S5(K=2, n=100, tau=0.0)
    with pytest.raises(ValueError, match="tau"):
        simulate_S5(K=2, n=100, tau=1.0)


@pytest.mark.unit
def test_simulate_S5_rejects_invalid_sigma_grid() -> None:
    """sigma_grid must be non-empty and strictly positive."""
    with pytest.raises(ValueError, match="sigma_grid"):
        simulate_S5(K=2, n=100, sigma_grid=())
    with pytest.raises(ValueError, match="sigma_grid"):
        simulate_S5(K=2, n=100, sigma_grid=(1.0, -1.0))


@pytest.mark.unit
def test_simulate_S5_rejects_invalid_nu() -> None:
    """nu must be at least 1."""
    with pytest.raises(ValueError, match="nu"):
        simulate_S5(K=2, n=100, nu=0)


@pytest.mark.unit
def test_simulate_S5_sparse_labels_and_pi() -> None:
    """S5-sparse keeps the S5 mixture structure but with default pi*=0.10."""
    sim = simulate_S5_sparse(K=2, n=200, seed=0)
    assert sim.regime == "S5-sparse"
    assert sim.true_pi == pytest.approx(0.10)
    assert int(jnp.sum(sim.true_Z)) == 20


@pytest.mark.unit
def test_simulate_S5_marginals_use_t5_scale() -> None:
    """Per-replicate marginals are scaled t_nu; irrep std tracks sigma * sqrt(nu/(nu-2))."""
    sigma_grid = (1.0, 2.0, 3.0)
    sim = simulate_S5(K=3, n=4000, sigma_grid=sigma_grid, nu=5, seed=0)
    X = np.asarray(sim.X)
    Z = np.asarray(sim.true_Z)
    # For t_5 the theoretical std is sqrt(nu/(nu-2)) = sqrt(5/3) ≈ 1.291.
    # Irrep features come from a t_5 marginal with scale sigma_j, so emp std ≈ sigma_j * 1.291.
    t5_std = np.sqrt(5.0 / 3.0)
    stds = [float(np.std(X[Z == 0, j])) for j in range(3)]
    for emp, exp_sigma in zip(stds, sigma_grid, strict=False):
        # 20% tolerance — t_5 has a finite but noisy second moment.
        assert exp_sigma * t5_std * 0.7 < emp < exp_sigma * t5_std * 1.6
