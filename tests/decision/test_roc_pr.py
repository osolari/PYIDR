"""Tests for the ROC and PR helpers."""

from __future__ import annotations

import numpy as np
import pytest

from py_idr.decision.roc_pr import (
    PRCurve,
    ROCCurve,
    auc,
    average_precision,
    pr_curve,
    roc_curve,
)

# ---- validators ----------------------------------------------------------------------


@pytest.mark.unit
def test_roc_curve_rejects_shape_mismatch() -> None:
    """idr and true_Z must share shape."""
    with pytest.raises(ValueError, match="matching shape"):
        roc_curve(np.array([0.1, 0.2]), np.array([1]))


@pytest.mark.unit
def test_roc_curve_rejects_degenerate_class() -> None:
    """All-positive or all-negative Z is rejected."""
    idr = np.array([0.1, 0.2, 0.3])
    with pytest.raises(ValueError, match="undefined"):
        roc_curve(idr, np.array([1, 1, 1]))
    with pytest.raises(ValueError, match="undefined"):
        roc_curve(idr, np.array([0, 0, 0]))


@pytest.mark.unit
def test_pr_curve_rejects_no_positives() -> None:
    """PR is undefined when there are no positives."""
    idr = np.array([0.1, 0.2, 0.3])
    with pytest.raises(ValueError, match="zero positive"):
        pr_curve(idr, np.array([0, 0, 0]))


# ---- perfect / random classifiers ----------------------------------------------------


@pytest.mark.unit
def test_roc_perfect_classifier_auc_1() -> None:
    """When positives are strictly ranked above negatives, AUC = 1."""
    # Positives: idr near 0. Negatives: idr near 1.
    rng = np.random.default_rng(0)
    idr = np.concatenate([rng.uniform(0, 0.1, 50), rng.uniform(0.9, 1.0, 50)])
    Z = np.concatenate([np.ones(50, int), np.zeros(50, int)])
    assert auc(roc_curve(idr, Z)) == pytest.approx(1.0)


@pytest.mark.unit
def test_roc_inverted_classifier_auc_0() -> None:
    """A perfectly inverted classifier (positives have largest idr) has AUC = 0."""
    rng = np.random.default_rng(0)
    idr = np.concatenate([rng.uniform(0.9, 1.0, 50), rng.uniform(0, 0.1, 50)])
    Z = np.concatenate([np.ones(50, int), np.zeros(50, int)])
    assert auc(roc_curve(idr, Z)) == pytest.approx(0.0)


@pytest.mark.unit
def test_roc_random_classifier_auc_near_half() -> None:
    """A random idr has AUC ≈ 0.5 (loose tolerance on 200 samples)."""
    rng = np.random.default_rng(0)
    idr = rng.uniform(size=200)
    Z = rng.integers(0, 2, size=200)
    assert abs(auc(roc_curve(idr, Z)) - 0.5) < 0.15


# ---- endpoints & monotonicity --------------------------------------------------------


@pytest.mark.unit
def test_roc_endpoints_are_origin_and_one() -> None:
    """ROC starts at (0, 0) and ends at (1, 1)."""
    rng = np.random.default_rng(0)
    idr = rng.uniform(size=50)
    Z = rng.integers(0, 2, size=50)
    roc = roc_curve(idr, Z)
    assert float(roc.fpr[0]) == 0.0
    assert float(roc.tpr[0]) == 0.0
    assert float(roc.fpr[-1]) == 1.0
    assert float(roc.tpr[-1]) == 1.0


@pytest.mark.unit
def test_roc_is_monotone() -> None:
    """FPR and TPR are both monotone non-decreasing."""
    rng = np.random.default_rng(0)
    idr = rng.uniform(size=80)
    Z = rng.integers(0, 2, size=80)
    roc = roc_curve(idr, Z)
    assert np.all(np.diff(roc.fpr) >= -1e-12)
    assert np.all(np.diff(roc.tpr) >= -1e-12)


@pytest.mark.unit
def test_pr_curve_recall_is_monotone_ascending() -> None:
    """Recall is sorted ascending."""
    rng = np.random.default_rng(0)
    idr = rng.uniform(size=80)
    Z = rng.integers(0, 2, size=80)
    pr = pr_curve(idr, Z)
    assert np.all(np.diff(pr.recall) >= -1e-12)


@pytest.mark.unit
def test_pr_starts_at_recall_zero_precision_one() -> None:
    """First point of the PR curve is the (0, 1) origin (sklearn convention)."""
    rng = np.random.default_rng(0)
    idr = rng.uniform(size=50)
    Z = np.concatenate([np.ones(25, int), np.zeros(25, int)])
    pr = pr_curve(idr, Z)
    assert float(pr.recall[0]) == 0.0
    assert float(pr.precision[0]) == 1.0


@pytest.mark.unit
def test_pr_average_precision_perfect_is_one() -> None:
    """AP = 1 for a perfect classifier."""
    rng = np.random.default_rng(0)
    idr = np.concatenate([rng.uniform(0, 0.1, 50), rng.uniform(0.9, 1.0, 50)])
    Z = np.concatenate([np.ones(50, int), np.zeros(50, int)])
    assert average_precision(pr_curve(idr, Z)) == pytest.approx(1.0)


# ---- structural ----------------------------------------------------------------------


@pytest.mark.unit
def test_roc_curve_returns_named_dataclass() -> None:
    """The returned object is an ROCCurve with the expected fields."""
    idr = np.array([0.1, 0.4, 0.6, 0.9])
    Z = np.array([1, 1, 0, 0])
    roc = roc_curve(idr, Z)
    assert isinstance(roc, ROCCurve)
    assert roc.fpr.shape == roc.tpr.shape == roc.thresholds.shape


@pytest.mark.unit
def test_pr_curve_returns_named_dataclass() -> None:
    """The returned object is a PRCurve with the expected fields."""
    idr = np.array([0.1, 0.4, 0.6, 0.9])
    Z = np.array([1, 1, 0, 0])
    pr = pr_curve(idr, Z)
    assert isinstance(pr, PRCurve)
    assert pr.precision.shape == pr.recall.shape == pr.thresholds.shape


# ---- known cell-by-cell --------------------------------------------------------------


@pytest.mark.unit
def test_roc_curve_small_known_example() -> None:
    """4-feature hand-computable example.

    idr = [0.1, 0.4, 0.6, 0.9], Z = [1, 1, 0, 0]:
    scores (1-idr) = [0.9, 0.6, 0.4, 0.1]. Sorted descending order is
    already (feat 0 [pos], feat 1 [pos], feat 2 [neg], feat 3 [neg]).
    Each feature is a distinct threshold, so the curve steps:
    (0,0) [prepended] → (0, .5) → (0, 1) → (.5, 1) → (1, 1) → (1, 1) [appended].
    """
    roc = roc_curve(np.array([0.1, 0.4, 0.6, 0.9]), np.array([1, 1, 0, 0]))
    assert auc(roc) == pytest.approx(1.0)
    assert tuple(float(v) for v in roc.fpr) == (0.0, 0.0, 0.0, 0.5, 1.0, 1.0)
    assert tuple(float(v) for v in roc.tpr) == (0.0, 0.5, 1.0, 1.0, 1.0, 1.0)
