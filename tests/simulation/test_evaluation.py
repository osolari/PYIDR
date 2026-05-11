"""Recovery-evaluation tests."""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from py_idr.simulation.evaluation import RecoveryResult, evaluate_recovery


@pytest.mark.unit
def test_evaluate_recovery_near_perfect_idr() -> None:
    """Strong separation: low-idr Z=1 + high-idr Z=0 → power = 1 and realised FDR ≤ α.

    Sun--Cai's running-average rule can pull one or two high-idr features into
    the rejection set as long as the running mean stays ≤ α — this is the
    controlled-FDR property of the step-up procedure. The test therefore
    asserts that power is perfect and realised FDR is bounded above by α,
    not strictly zero.
    """
    true_Z = jnp.concatenate([jnp.ones(40, dtype=jnp.int32), jnp.zeros(60, dtype=jnp.int32)])
    idr = jnp.concatenate([jnp.full(40, 0.01), jnp.full(60, 0.99)])
    result = evaluate_recovery(idr, true_Z, alpha=0.05)
    assert isinstance(result, RecoveryResult)
    assert result.power == 1.0
    assert result.realized_fdr <= 0.05
    assert result.k_alpha >= 40


@pytest.mark.unit
def test_evaluate_recovery_returns_zero_on_empty_rejection_set() -> None:
    """When no idr crosses the Sun--Cai threshold, realised FDR + power = 0."""
    n = 50
    # Every idr above α — Sun--Cai returns an empty set.
    idr = jnp.full(n, 0.5)
    true_Z = jnp.concatenate([jnp.ones(20, dtype=jnp.int32), jnp.zeros(30, dtype=jnp.int32)])
    result = evaluate_recovery(idr, true_Z, alpha=0.05)
    assert result.k_alpha == 0
    assert result.realized_fdr == 0.0
    assert result.power == 0.0


@pytest.mark.unit
def test_evaluate_recovery_realized_fdr_when_mixed() -> None:
    """With a mix of correct and incorrect rejections, realised FDR equals V/R exactly."""
    # idr ranking (after sort) places features 0–3 in the rejection set.
    # Among those 4: 2 are truly reproducible (Z=1), 2 are not (Z=0).
    n = 20
    idr = jnp.concatenate(
        [
            jnp.array([0.01, 0.02, 0.03, 0.04]),  # 4 lowest idr — selected at small alpha
            jnp.full(n - 4, 0.5),  # rest stay above the threshold
        ]
    )
    true_Z = jnp.array(
        [1, 1, 0, 0] + [0] * (n - 4),  # 2 of the first 4 are true
        dtype=jnp.int32,
    )
    result = evaluate_recovery(idr, true_Z, alpha=0.05)
    # All 4 lowest idrs sum-mean to 0.025 < 0.05 → all 4 rejected.
    assert result.k_alpha == 4
    # Realised FDR = 2 / 4 = 0.5.
    assert result.realized_fdr == 0.5
    # Power = 2 / 2 = 1.0 (both reproducible features were rejected).
    assert result.power == 1.0


@pytest.mark.unit
def test_evaluate_recovery_handles_no_reproducible_features() -> None:
    """If true_Z is all zeros, power is 0 even when rejections happen."""
    idr = jnp.concatenate([jnp.array([0.01, 0.02]), jnp.full(8, 0.5)])
    true_Z = jnp.zeros(10, dtype=jnp.int32)
    result = evaluate_recovery(idr, true_Z, alpha=0.05)
    assert result.power == 0.0


@pytest.mark.unit
def test_evaluate_recovery_rejects_mismatched_shapes() -> None:
    """idr and true_Z must share length."""
    idr = jnp.array([0.1, 0.2, 0.3])
    true_Z = jnp.array([1, 0], dtype=jnp.int32)
    with pytest.raises(ValueError, match="share shape"):
        evaluate_recovery(idr, true_Z)


@pytest.mark.unit
def test_evaluate_recovery_rejects_empty_input() -> None:
    """Empty idr is a caller error."""
    with pytest.raises(ValueError, match="non-empty"):
        evaluate_recovery(jnp.array([]), jnp.array([], dtype=jnp.int32))
