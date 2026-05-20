"""Unit tests for the ChIP-R comparator (W10.11)."""

from __future__ import annotations

import numpy as np
import pytest

from py_idr.comparators.chipr import fit_chipr


@pytest.mark.unit
def test_fit_chipr_returns_unit_interval() -> None:
    """ChIP-R output is in [1/n, 1] by construction."""
    rng = np.random.default_rng(0)
    n, K = 50, 3
    X = rng.normal(size=(n, K))
    idr = fit_chipr(X)
    assert idr.shape == (n,)
    assert idr.min() >= 1.0 / n - 1e-12
    assert idr.max() <= 1.0 + 1e-12


@pytest.mark.unit
def test_fit_chipr_top_feature_is_minimal() -> None:
    """A feature with the best (highest) score on every replicate gets the
    geometric-mean rank-product 1/n."""
    n, K = 20, 4
    rng = np.random.default_rng(1)
    X = rng.normal(size=(n, K))
    # Inject a feature whose score is +∞ on every replicate.
    X[5] = 100.0
    idr = fit_chipr(X)
    assert np.isclose(idr[5], 1.0 / n, atol=1e-12)


@pytest.mark.unit
def test_fit_chipr_separates_reproducible_from_random() -> None:
    """Reproducible features (consistently high ranks) score below random ones."""
    rng = np.random.default_rng(2)
    n_rep, n_noise, K = 20, 30, 4

    # Reproducible: shared high scores across replicates.
    base = rng.uniform(2.0, 5.0, size=n_rep)
    X_rep = np.column_stack([base + 0.01 * rng.normal(size=n_rep) for _ in range(K)])

    # Irreproducible: independent noise.
    X_noise = rng.normal(size=(n_noise, K))

    X = np.vstack([X_rep, X_noise])
    idr = fit_chipr(X)
    assert idr[:n_rep].mean() < idr[n_rep:].mean()


@pytest.mark.unit
def test_fit_chipr_invariant_to_monotone_per_replicate_transform() -> None:
    """Rank-product is built on per-replicate ranks → invariant to monotone transforms."""
    rng = np.random.default_rng(3)
    X = rng.normal(size=(40, 5))
    idr_raw = fit_chipr(X)
    X_xformed = np.column_stack(
        [
            np.expm1(X[:, 0]),
            2.0 * X[:, 1] + 7.0,
            np.cbrt(X[:, 2]),
            -np.exp(-X[:, 3]),  # also monotone
            X[:, 4] ** 3,
        ]
    )
    idr_xformed = fit_chipr(X_xformed)
    # Within numerical noise of identical (ties broken the same way).
    assert np.allclose(idr_raw, idr_xformed, atol=1e-9)


@pytest.mark.unit
def test_fit_chipr_rejects_1d_input() -> None:
    """X must be 2-D (n, K)."""
    with pytest.raises(ValueError, match="2-D"):
        fit_chipr(np.zeros(10))


@pytest.mark.unit
def test_fit_chipr_rejects_K_lt_2() -> None:
    """A single replicate has no across-replicate variance."""
    with pytest.raises(ValueError, match="K must be >= 2"):
        fit_chipr(np.zeros((10, 1)))


@pytest.mark.unit
def test_fit_chipr_matches_geometric_mean_formula() -> None:
    """Spot-check against the closed-form geometric mean of descending ranks."""
    # Construct a tiny example where the descending ranks are easy to read.
    X = np.array(
        [
            [5.0, 5.0, 5.0],  # all top → rank 1 each → RP = 1/n
            [4.0, 4.0, 4.0],  # all second → rank 2 each
            [3.0, 3.0, 3.0],
            [2.0, 2.0, 2.0],
            [1.0, 1.0, 1.0],  # all bottom → rank 5 each
        ]
    )
    n = 5
    idr = fit_chipr(X)
    # Expected: rank^(1/K) / n for K = 3 — but with all-same scores across
    # replicates, the descending rank is identical per feature so RP =
    # rank itself.
    expected = np.array([1, 2, 3, 4, 5]) / n
    assert np.allclose(idr, expected, atol=1e-12)
