r"""ROC and precision-recall curves from a posterior local idr.

The Sun-Cai step-up rule in :mod:`py_idr.decision.sun_cai` picks **one**
threshold $\gamma_\alpha$. ROC and PR curves sweep the threshold over
its full range and ask "how does discrimination trade off with
operating point?". They are the natural summary for figures like
``fig:roc-pr`` (F4) and for comparing PY-IDR against thresholding
methods that don't have a built-in calibrated rule.

Design notes
------------

We do not depend on scikit-learn (not in our pinned deps). The two
helpers here are short re-implementations of the standard
"sort-and-cumulate" algorithm:

- :func:`roc_curve` — returns FPR and TPR at every distinct decision
  threshold (the local-idr value at each rejected feature). The
  endpoints ``(0, 0)`` and ``(1, 1)`` are prepended / appended.
- :func:`pr_curve` — returns precision and recall at every distinct
  threshold. Precision is undefined at recall = 0; we follow the
  scikit-learn convention of dropping that point.

Both helpers expect **local-idr** input (small = more likely
reproducible), matching the rest of the pipeline. Internally we
convert to a "score" ``s = 1 - idr`` so the curves match the
standard "high score = positive class" convention.

The corresponding scalar summaries are :func:`auc` (area under the
ROC curve) and :func:`average_precision` (the standard PR-AUC).

See plans/06_decision_theory.md and §4.2 of the report.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from jaxtyping import Array, Float, Integer


@dataclass(frozen=True)
class ROCCurve:
    """One ROC curve: false-positive rate vs true-positive rate.

    Attributes
    ----------
    fpr
        Sorted false-positive rates, length $L$, starting at 0 and
        ending at 1.
    tpr
        True-positive rates at the matching threshold, same length.
        Sorted to be monotone non-decreasing.
    thresholds
        The local-idr threshold above which a feature is rejected
        (i.e., the cutoff that produced each ``(fpr, tpr)`` point).
        Length $L$. Synthetic endpoints have threshold $\\pm\\infty$
        which we encode as ``-inf`` / ``+inf``.
    """

    fpr: np.ndarray
    tpr: np.ndarray
    thresholds: np.ndarray


@dataclass(frozen=True)
class PRCurve:
    """One precision-recall curve.

    Attributes
    ----------
    precision, recall
        Same-length arrays of precision and recall at each threshold.
        Recall is sorted ascending.
    thresholds
        The local-idr cutoff for each point. Same length as
        ``precision`` and ``recall``.
    """

    precision: np.ndarray
    recall: np.ndarray
    thresholds: np.ndarray


def roc_curve(
    idr: Float[Array, "n"],
    true_Z: Integer[Array, "n"],
) -> ROCCurve:
    r"""Compute the ROC curve from a per-feature local idr and ground truth.

    Parameters
    ----------
    idr
        Length-$n$ local-idr vector with values in $[0, 1]$. Smaller =
        more likely reproducible.
    true_Z
        Length-$n$ ground-truth class indicators in $\{0, 1\}$. ``1``
        means reproducible (the positive class).

    Returns
    -------
    A :class:`ROCCurve`. The first point is ``(0, 0)`` and the last is
    ``(1, 1)``; the array is monotone non-decreasing in both
    coordinates.

    Raises
    ------
    ValueError
        If the inputs have mismatched shapes or ``true_Z`` is degenerate
        (all 0 or all 1 — ROC is undefined).
    """
    idr = np.asarray(idr, dtype=float)
    true_Z = np.asarray(true_Z, dtype=int)
    if idr.shape != true_Z.shape:
        raise ValueError(
            f"idr and true_Z must have matching shape; got {idr.shape} vs {true_Z.shape}"
        )
    n_pos = int(np.sum(true_Z == 1))
    n_neg = int(np.sum(true_Z == 0))
    if n_pos == 0 or n_neg == 0:
        raise ValueError(f"ROC undefined when one class is empty; got n_pos={n_pos}, n_neg={n_neg}")

    # Convert local idr to score (larger = more positive).
    score = 1.0 - idr
    # Sort descending by score so we sweep the threshold from high to low.
    order = np.argsort(-score, kind="mergesort")
    Z_sorted = true_Z[order]
    score_sorted = score[order]

    tps = np.cumsum(Z_sorted == 1).astype(float)
    fps = np.cumsum(Z_sorted == 0).astype(float)
    tpr = tps / n_pos
    fpr = fps / n_neg

    # Deduplicate equal-score runs (only the last point of each run is
    # operationally meaningful).
    distinct = np.r_[np.diff(score_sorted) != 0, True]
    tpr = tpr[distinct]
    fpr = fpr[distinct]
    thresholds = idr[order][distinct]  # local-idr cutoff per point

    # Prepend the (0, 0) origin and append (1, 1) finish.
    fpr = np.concatenate([[0.0], fpr, [1.0]])
    tpr = np.concatenate([[0.0], tpr, [1.0]])
    thresholds = np.concatenate([[-np.inf], thresholds, [np.inf]])

    return ROCCurve(fpr=fpr, tpr=tpr, thresholds=thresholds)


def pr_curve(
    idr: Float[Array, "n"],
    true_Z: Integer[Array, "n"],
) -> PRCurve:
    r"""Compute the precision-recall curve.

    Parameters mirror :func:`roc_curve`. Returns a :class:`PRCurve`
    sorted by ascending recall. Endpoints follow scikit-learn:
    precision = 1, recall = 0 at the start (no rejections), and
    precision = n_pos / n at the end (rejecting everything).

    Raises
    ------
    ValueError
        If shapes mismatch or there are no positive ground-truth
        features.
    """
    idr = np.asarray(idr, dtype=float)
    true_Z = np.asarray(true_Z, dtype=int)
    if idr.shape != true_Z.shape:
        raise ValueError(
            f"idr and true_Z must have matching shape; got {idr.shape} vs {true_Z.shape}"
        )
    n_pos = int(np.sum(true_Z == 1))
    if n_pos == 0:
        raise ValueError("PR curve undefined when there are zero positive features")

    score = 1.0 - idr
    order = np.argsort(-score, kind="mergesort")
    Z_sorted = true_Z[order]
    score_sorted = score[order]

    tps = np.cumsum(Z_sorted == 1).astype(float)
    k = np.arange(1, idr.size + 1, dtype=float)
    precision = tps / k
    recall = tps / n_pos

    distinct = np.r_[np.diff(score_sorted) != 0, True]
    precision = precision[distinct]
    recall = recall[distinct]
    thresholds = idr[order][distinct]

    # Prepend a (recall=0, precision=1) origin so the curve starts in
    # the top-left corner. (Matches the sklearn convention.)
    precision = np.concatenate([[1.0], precision])
    recall = np.concatenate([[0.0], recall])
    thresholds = np.concatenate([[-np.inf], thresholds])

    return PRCurve(precision=precision, recall=recall, thresholds=thresholds)


def auc(roc: ROCCurve) -> float:
    r"""Area under the ROC curve via the trapezoidal rule.

    Returns a float in $[0, 1]$. A perfect classifier scores 1.0; a
    random classifier scores 0.5.
    """
    # NumPy renamed `trapz` to `trapezoid` in 2.0; use the new name when
    # present and fall back without touching the deprecated symbol.
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(roc.tpr, roc.fpr))
    return float(np.trapz(roc.tpr, roc.fpr))  # type: ignore[attr-defined]


def average_precision(pr: PRCurve) -> float:
    r"""Area under the precision-recall curve (AP).

    Computed as the sum of precision-weighted recall increments —
    matches scikit-learn's :func:`sklearn.metrics.average_precision_score`
    convention for the discrete operating points.
    """
    # Sum of P(k) * (R(k) - R(k-1)) excluding the (0, 1) origin.
    delta_recall = np.diff(pr.recall)
    return float(np.sum(pr.precision[1:] * delta_recall))
