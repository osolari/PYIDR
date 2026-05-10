"""Sun-Cai step-up rule tests (Eq. 3.11 of the revised report)."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from py_idr.decision.sun_cai import StepUpResult, discoveries, step_up_threshold


@pytest.mark.decision
def test_step_up_basic() -> None:
    """A small worked example: idr = [0.01, 0.02, 0.10, 0.30, 0.50, 0.80] at α = 0.05."""
    idr = jnp.array([0.01, 0.02, 0.10, 0.30, 0.50, 0.80])
    # Running means:    0.010, 0.015, 0.043, 0.108, 0.186, 0.288
    # Qualifying ≤ 0.05: T, T, T, F, F, F → k = 3
    res = step_up_threshold(idr, alpha=0.05)
    assert res.k_alpha == 3
    assert pytest.approx(res.gamma_alpha, abs=1e-7) == 0.10
    assert sorted(res.selected.tolist()) == [0, 1, 2]


@pytest.mark.decision
def test_step_up_returns_dataclass() -> None:
    """The return type is the public StepUpResult dataclass."""
    idr = jnp.array([0.05, 0.10, 0.15])
    res = step_up_threshold(idr, alpha=0.10)
    assert isinstance(res, StepUpResult)


@pytest.mark.decision
def test_step_up_empty_rejection() -> None:
    """When every idr exceeds α, the result is empty — not NaN, not an exception.

    This is the contract that callers rely on when iterating over an α-grid;
    earlier drafts of Eq. (3.10) would 0/0 here.
    """
    idr = jnp.array([0.6, 0.7, 0.8, 0.9])
    res = step_up_threshold(idr, alpha=0.05)
    assert res.k_alpha == 0
    assert np.isnan(res.gamma_alpha)
    assert res.selected.shape == (0,)


@pytest.mark.decision
def test_step_up_all_zero_idr() -> None:
    """If every feature is confidently reproducible, every feature is selected."""
    idr = jnp.zeros(5)
    res = step_up_threshold(idr, alpha=0.05)
    assert res.k_alpha == 5
    assert res.gamma_alpha == 0.0
    assert sorted(res.selected.tolist()) == [0, 1, 2, 3, 4]


@pytest.mark.decision
def test_step_up_returns_indices_in_input_order() -> None:
    """The selected indices reference the *original* idr ordering, not the sorted one."""
    # idrs sorted descending in input order — selection should hit indices 4, 3, 2.
    idr = jnp.array([0.9, 0.8, 0.05, 0.02, 0.01])
    res = step_up_threshold(idr, alpha=0.05)
    assert res.k_alpha == 3
    assert sorted(res.selected.tolist()) == [2, 3, 4]


@pytest.mark.decision
def test_step_up_invalid_alpha_raises() -> None:
    """alpha must be in (0, 1)."""
    idr = jnp.array([0.1, 0.2, 0.3])
    with pytest.raises(ValueError, match="alpha must be in"):
        step_up_threshold(idr, alpha=0.0)
    with pytest.raises(ValueError, match="alpha must be in"):
        step_up_threshold(idr, alpha=1.0)


@pytest.mark.decision
def test_step_up_nan_input_raises() -> None:
    """NaN inputs are not silently accepted."""
    idr = jnp.array([0.1, float("nan"), 0.3])
    with pytest.raises(ValueError, match="NaN"):
        step_up_threshold(idr, alpha=0.05)


@pytest.mark.decision
def test_step_up_empty_input_raises() -> None:
    """An empty idr vector is a programmer error, not a valid call."""
    idr = jnp.array([])
    with pytest.raises(ValueError, match="empty"):
        step_up_threshold(idr, alpha=0.05)


@pytest.mark.decision
def test_discoveries_simple() -> None:
    """discoveries returns indices of features below the threshold, in ascending order."""
    idr = jnp.array([0.1, 0.4, 0.05, 0.6, 0.02])
    out = discoveries(idr, gamma=0.15)
    assert sorted(out.tolist()) == [0, 2, 4]


@pytest.mark.decision
def test_discoveries_strict_inequality() -> None:
    """The threshold is *strict* — features with idr exactly == γ are not selected.

    Matches Eq. (3.10) of the report: V(γ) = Σ 1{idr_i < γ}.
    """
    idr = jnp.array([0.1, 0.2, 0.2])
    out = discoveries(idr, gamma=0.2)
    assert out.tolist() == [0]
