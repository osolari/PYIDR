"""Unit tests for the eCV comparator (W10.5)."""

from __future__ import annotations

import numpy as np
import pytest

from py_idr.comparators.ecv import fit_ecv


@pytest.mark.unit
def test_fit_ecv_returns_unit_interval() -> None:
    """eCV output is in [0, 1] by construction (max-normalised)."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(50, 3))
    idr = fit_ecv(X)
    assert idr.shape == (50,)
    assert idr.min() >= 0.0
    assert idr.max() <= 1.0 + 1e-9
    # The argmax cell is exactly 1.0.
    assert np.isclose(idr.max(), 1.0)


@pytest.mark.unit
def test_fit_ecv_perfectly_concordant_replicates_score_zero() -> None:
    """If X[:, 0] == X[:, 1] == ... then ranks are identical and CV = 0."""
    n = 30
    rng = np.random.default_rng(1)
    x = rng.normal(size=n)
    X = np.column_stack([x, x, x])  # K = 3 identical replicates
    idr = fit_ecv(X)
    # Every feature has eCV = 0 (zero variance across replicates).
    assert np.allclose(idr, 0.0, atol=1e-12)


@pytest.mark.unit
def test_fit_ecv_separates_reproducible_from_random() -> None:
    """Features with concordant ranks score below features with random ranks."""
    rng = np.random.default_rng(2)
    n_rep, n_noise, K = 20, 30, 3

    # Reproducible: shared high ranks across replicates.
    base = rng.uniform(2.0, 5.0, size=n_rep)
    X_rep = np.column_stack([base + 0.01 * rng.normal(size=n_rep) for _ in range(K)])

    # Irreproducible: independent noise per replicate.
    X_noise = rng.normal(size=(n_noise, K))

    X = np.vstack([X_rep, X_noise])
    idr = fit_ecv(X)

    # Mean idr of the reproducible block should be strictly less than the noise block.
    assert idr[:n_rep].mean() < idr[n_rep:].mean()


@pytest.mark.unit
def test_fit_ecv_rejects_1d_input() -> None:
    """X must be 2-D (n, K)."""
    with pytest.raises(ValueError, match="2-D"):
        fit_ecv(np.zeros(10))


@pytest.mark.unit
def test_fit_ecv_rejects_K_lt_2() -> None:
    """A single replicate has no across-replicate variance — comparator undefined."""
    with pytest.raises(ValueError, match="K must be >= 2"):
        fit_ecv(np.zeros((10, 1)))


@pytest.mark.unit
def test_fit_ecv_invariant_to_monotone_per_replicate_transform() -> None:
    """eCV is built on per-replicate ranks → invariant to monotone transforms."""
    rng = np.random.default_rng(3)
    X = rng.normal(size=(40, 3))
    idr_raw = fit_ecv(X)
    # Apply per-replicate monotone transforms (different per column).
    X_xformed = np.column_stack(
        [
            np.expm1(X[:, 0]),
            2.0 * X[:, 1] + 7.0,
            np.cbrt(X[:, 2]),
        ]
    )
    idr_xformed = fit_ecv(X_xformed)
    # Ranks are preserved by monotone transforms, so eCV is identical.
    assert np.allclose(idr_raw, idr_xformed, atol=1e-12)
