"""Tests for the closed-form and MC Hellinger helpers."""

from __future__ import annotations

import pytest

from py_idr.eval.hellinger import (
    hellinger_gaussian_copula,
    hellinger_gaussian_copula_mc,
)

# ---- validation ----------------------------------------------------------------------


@pytest.mark.unit
def test_closed_form_rejects_K_lt_2() -> None:
    """K must be at least 2."""
    with pytest.raises(ValueError, match="K must be >= 2"):
        hellinger_gaussian_copula(0.5, 0.5, K=1)


@pytest.mark.unit
def test_closed_form_rejects_rho_out_of_range() -> None:
    """rho must be in (-1/(K-1), 1) so the matrix is positive definite."""
    with pytest.raises(ValueError, match="rho_1"):
        hellinger_gaussian_copula(1.0, 0.5, K=2)
    with pytest.raises(ValueError, match="rho_2"):
        hellinger_gaussian_copula(0.5, -1.0, K=2)


# ---- closed-form correctness ---------------------------------------------------------


@pytest.mark.unit
def test_closed_form_identical_correlations_zero() -> None:
    """H(rho, rho) == 0 up to floating-point noise (no MC variance)."""
    for rho in (0.0, 0.3, 0.5, 0.85):
        for K in (2, 3, 5):
            assert hellinger_gaussian_copula(rho, rho, K=K) == pytest.approx(0.0, abs=1e-6)


@pytest.mark.unit
def test_closed_form_symmetric() -> None:
    """H(rho_1, rho_2) == H(rho_2, rho_1)."""
    assert hellinger_gaussian_copula(0.2, 0.8, K=3) == pytest.approx(
        hellinger_gaussian_copula(0.8, 0.2, K=3)
    )


@pytest.mark.unit
def test_closed_form_in_unit_interval() -> None:
    """Output is in [0, 1]."""
    for rho1, rho2 in [(0.1, 0.9), (0.5, 0.55), (-0.3, 0.7)]:
        h = hellinger_gaussian_copula(rho1, rho2, K=3)
        assert 0.0 <= h <= 1.0


@pytest.mark.unit
def test_closed_form_grows_with_gap() -> None:
    """Wider correlation gap -> larger Hellinger distance."""
    small_gap = hellinger_gaussian_copula(0.5, 0.55, K=2)
    medium_gap = hellinger_gaussian_copula(0.3, 0.7, K=2)
    big_gap = hellinger_gaussian_copula(0.1, 0.9, K=2)
    assert small_gap < medium_gap < big_gap


@pytest.mark.unit
def test_closed_form_increases_with_K_at_fixed_gap() -> None:
    """Same correlation gap, higher K -> larger Hellinger (the gap is more visible
    in higher-dimensional joint behavior)."""
    fixed_gap = [hellinger_gaussian_copula(0.1, 0.9, K=K) for K in (2, 3, 5, 10)]
    assert fixed_gap == sorted(fixed_gap)


# ---- MC sanity vs closed form --------------------------------------------------------


@pytest.mark.unit
def test_mc_matches_closed_form_to_within_noise() -> None:
    """The MC estimator should match the closed form to within MC noise."""
    rho_1, rho_2, K = 0.5, 0.85, 2
    closed = hellinger_gaussian_copula(rho_1, rho_2, K=K)
    mc = hellinger_gaussian_copula_mc(rho_1, rho_2, K=K, n_mc=50_000, rng_seed=0)
    # MC SE on H roughly sqrt(SE_B) ≈ 0.02 at n_mc=50k for moderate rhos.
    assert abs(mc - closed) < 0.05


@pytest.mark.unit
def test_mc_is_seeded() -> None:
    """Same rng_seed reproduces the MC estimate exactly."""
    a = hellinger_gaussian_copula_mc(0.1, 0.9, K=3, n_mc=2000, rng_seed=42)
    b = hellinger_gaussian_copula_mc(0.1, 0.9, K=3, n_mc=2000, rng_seed=42)
    assert a == b


@pytest.mark.unit
def test_mc_rejects_bad_n_mc() -> None:
    """n_mc must be at least 1."""
    with pytest.raises(ValueError, match="n_mc"):
        hellinger_gaussian_copula_mc(0.5, 0.5, K=2, n_mc=0)
