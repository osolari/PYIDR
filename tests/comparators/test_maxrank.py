"""Tests for the MaxRank rank-based comparator."""

from __future__ import annotations

import numpy as np
import pytest

from py_idr.comparators.maxrank import fit_maxrank

# ---- validators ----------------------------------------------------------------------


@pytest.mark.unit
def test_fit_maxrank_rejects_1d() -> None:
    """X must be 2-D."""
    with pytest.raises(ValueError, match="X must be 2-D"):
        fit_maxrank(np.zeros(20))


@pytest.mark.unit
def test_fit_maxrank_rejects_single_replicate() -> None:
    """K must be at least 2."""
    with pytest.raises(ValueError, match="K must be >= 2"):
        fit_maxrank(np.zeros((20, 1)))


# ---- output contract -----------------------------------------------------------------


@pytest.mark.unit
def test_fit_maxrank_returns_correct_shape() -> None:
    """Output is length n."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(60, 3))
    idr = fit_maxrank(X)
    assert idr.shape == (60,)


@pytest.mark.unit
def test_fit_maxrank_in_unit_interval() -> None:
    """Values are in [1/n, 1]."""
    rng = np.random.default_rng(0)
    n = 50
    X = rng.normal(size=(n, 3))
    idr = fit_maxrank(X)
    assert idr.min() >= 1.0 / n - 1e-9
    assert idr.max() <= 1.0 + 1e-9


@pytest.mark.unit
def test_fit_maxrank_deterministic() -> None:
    """Same input → same output (no random component)."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(50, 3))
    a = fit_maxrank(X)
    b = fit_maxrank(X)
    assert np.array_equal(a, b)


# ---- correctness ---------------------------------------------------------------------


@pytest.mark.unit
def test_fit_maxrank_top_feature_in_all_replicates_gets_low_idr() -> None:
    """A feature that ranks #1 in every replicate has max rank = 1, idr = 1/n."""
    n, K = 10, 3
    X = np.zeros((n, K))
    # Make feature 0 the best in every replicate.
    X[0, :] = 100.0
    # Everyone else gets random scores.
    rng = np.random.default_rng(0)
    X[1:, :] = rng.normal(size=(n - 1, K))
    idr = fit_maxrank(X)
    assert idr[0] == pytest.approx(1.0 / n)


@pytest.mark.unit
def test_fit_maxrank_bottom_feature_in_some_replicate_gets_high_idr() -> None:
    """A feature that is last in even one replicate has max rank = n, idr = 1."""
    n, K = 10, 3
    rng = np.random.default_rng(0)
    X = rng.normal(size=(n, K))
    # Make feature 0 last in replicate 1 by giving it the smallest score there.
    X[0, 1] = -1000.0
    idr = fit_maxrank(X)
    assert idr[0] == pytest.approx(1.0)


@pytest.mark.unit
def test_fit_maxrank_S1_AUC_better_than_chance() -> None:
    """On clean S1 data the rank-based score discriminates better than random."""
    from py_idr.decision.roc_pr import auc, roc_curve
    from py_idr.simulation.scenarios import simulate_S1

    sim = simulate_S1(K=3, n=500, pi_star=0.30, rho=0.85, seed=0)
    Z = np.asarray(sim.true_Z)
    idr = fit_maxrank(np.asarray(sim.X))
    a = auc(roc_curve(idr, Z))
    assert a > 0.65  # comfortably above chance


@pytest.mark.unit
def test_fit_maxrank_high_rho_gives_higher_AUC_than_low_rho() -> None:
    """Stronger reproducible signal → better rank-based discrimination."""
    from py_idr.decision.roc_pr import auc, roc_curve
    from py_idr.simulation.scenarios import simulate_S1

    sim_high = simulate_S1(K=3, n=500, pi_star=0.30, rho=0.85, seed=0)
    sim_low = simulate_S1(K=3, n=500, pi_star=0.30, rho=0.20, seed=0)
    Z_high = np.asarray(sim_high.true_Z)
    Z_low = np.asarray(sim_low.true_Z)
    auc_high = auc(roc_curve(fit_maxrank(np.asarray(sim_high.X)), Z_high))
    auc_low = auc(roc_curve(fit_maxrank(np.asarray(sim_low.X)), Z_low))
    assert auc_high > auc_low


@pytest.mark.unit
def test_fit_maxrank_invariant_to_monotone_transform() -> None:
    """Rank-based score is invariant to per-replicate monotone transforms."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(40, 3))
    # Apply a monotone transform per replicate (square-then-sign-preserve).
    X_transformed = np.sign(X) * np.abs(X) ** 1.5
    a = fit_maxrank(X)
    b = fit_maxrank(X_transformed)
    assert np.array_equal(a, b)
