"""Bayes-mFDR tests (Eq. 3.10 of the revised report; max(1, E[R]) denominator)."""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from py_idr.decision.bayes_mfdr import bayes_mfdr_curve, posterior_decomposition


@pytest.mark.decision
def test_bayes_mfdr_basic_values() -> None:
    """Worked example: idr = [0.01, 0.02, 0.10, 0.30, 0.50] with γ = 0.20.

    Below 0.20: {0.01, 0.02, 0.10} -> R = 3, V = 0.13. mFDR = 0.13 / max(1, 3) = 0.0433.
    """
    idr = jnp.array([0.01, 0.02, 0.10, 0.30, 0.50])
    curve = bayes_mfdr_curve(idr, jnp.array([0.20]))
    assert pytest.approx(float(curve[0]), abs=1e-6) == 0.13 / 3.0


@pytest.mark.decision
def test_bayes_mfdr_max_one_handles_empty_rejection() -> None:
    """At γ smaller than every idr, rejection set is empty — denominator = max(1, 0) = 1.

    The numerator is also 0 in that case, so the ratio is 0 (not NaN).
    """
    idr = jnp.array([0.10, 0.20, 0.30])
    curve = bayes_mfdr_curve(idr, jnp.array([0.05]))
    assert float(curve[0]) == 0.0


@pytest.mark.decision
def test_bayes_mfdr_max_one_does_not_change_nonempty_values() -> None:
    """For γ that yield R ≥ 1, the max(1, …) denominator equals E[R] exactly.

    The fix is invisible on any non-empty rejection set; this test pins that contract.
    """
    idr = jnp.array([0.05, 0.10, 0.20, 0.40])
    gammas = jnp.array([0.07, 0.15, 0.25, 0.50])  # R = 1, 2, 3, 4
    curve = bayes_mfdr_curve(idr, gammas)
    # Direct computation:
    expected = jnp.array(
        [
            0.05 / 1,  # R = 1
            (0.05 + 0.10) / 2,  # R = 2
            (0.05 + 0.10 + 0.20) / 3,  # R = 3
            (0.05 + 0.10 + 0.20 + 0.40) / 4,  # R = 4
        ]
    )
    assert jnp.allclose(curve, expected, atol=1e-6)


@pytest.mark.decision
def test_bayes_mfdr_monotone_non_decreasing() -> None:
    """Bayes-mFDR is non-decreasing in γ on a fine grid (operational sanity)."""
    rng = jnp.linspace(0.001, 0.999, 50)
    # Synthetic idr — uniformly distributed in [0, 1] suffices.
    idr = jnp.linspace(0.01, 0.99, 100)
    curve = bayes_mfdr_curve(idr, rng)
    diffs = curve[1:] - curve[:-1]
    assert bool(jnp.all(diffs >= -1e-12))


@pytest.mark.decision
def test_posterior_decomposition_matches_curve() -> None:
    """V/max(1, R) reconstructed from posterior_decomposition matches the curve."""
    idr = jnp.array([0.01, 0.10, 0.30, 0.50, 0.80])
    gamma = 0.20
    V, R = posterior_decomposition(idr, gamma)
    expected = V / max(1.0, R)
    curve = bayes_mfdr_curve(idr, jnp.array([gamma]))
    assert pytest.approx(float(curve[0]), abs=1e-12) == expected


@pytest.mark.decision
def test_bayes_mfdr_full_rejection_at_large_gamma() -> None:
    """At γ = 1, every feature is rejected — denominator = n, numerator = sum(idr)."""
    idr = jnp.array([0.05, 0.10, 0.20, 0.40, 0.60])
    curve = bayes_mfdr_curve(idr, jnp.array([1.0]))
    assert pytest.approx(float(curve[0]), abs=1e-6) == float(jnp.mean(idr))
