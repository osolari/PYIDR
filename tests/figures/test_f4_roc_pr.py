"""F4 ROC/PR figure reproducer tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from py_idr.figures.f4_roc_pr import (
    ROCPRSummary,
    compute_roc_pr_curves,
    make_roc_pr_figure,
)


def _synthetic_replicate(
    rng: np.random.Generator,
    n_pos: int = 50,
    n_neg: int = 50,
    separation: float = 0.7,
) -> tuple[np.ndarray, np.ndarray]:
    """Build a (idr, Z) pair where positives have idr drawn from a left-shifted Beta.

    ``separation`` controls how easy the discrimination is: 0 = random, 1 = perfect.
    """
    # Positives: Beta(2, 2+s); negatives: Beta(2+s, 2). At s=10 the two
    # distributions barely overlap.
    s = 10.0 * separation
    pos_idr = rng.beta(2.0, 2.0 + s, size=n_pos)
    neg_idr = rng.beta(2.0 + s, 2.0, size=n_neg)
    idr = np.concatenate([pos_idr, neg_idr])
    Z = np.concatenate([np.ones(n_pos, int), np.zeros(n_neg, int)])
    # Shuffle.
    perm = rng.permutation(idr.size)
    return idr[perm], Z[perm]


def _toy_dict(
    regimes: tuple[str, ...] = ("S5", "S5-sparse"),
    n_replicates: int = 8,
    separation_by_regime: dict[str, float] | None = None,
    rng_seed: int = 0,
) -> dict[str, list[tuple[np.ndarray, np.ndarray]]]:
    """Build per-regime list of (idr, Z) pairs."""
    rng = np.random.default_rng(rng_seed)
    if separation_by_regime is None:
        separation_by_regime = dict.fromkeys(regimes, 0.7)
    out: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {}
    for regime in regimes:
        out[regime] = [
            _synthetic_replicate(rng, separation=separation_by_regime[regime])
            for _ in range(n_replicates)
        ]
    return out


# ---- compute_roc_pr_curves: validation -----------------------------------------------


@pytest.mark.unit
def test_compute_roc_pr_rejects_bad_grid_size() -> None:
    """grid_size must be >= 2."""
    with pytest.raises(ValueError, match="grid_size"):
        compute_roc_pr_curves(_toy_dict(), grid_size=1)


@pytest.mark.unit
def test_compute_roc_pr_rejects_bad_ci_level() -> None:
    """ci_level must be in (0, 1)."""
    with pytest.raises(ValueError, match="ci_level"):
        compute_roc_pr_curves(_toy_dict(), ci_level=0.0)


# ---- compute_roc_pr_curves: structural -----------------------------------------------


@pytest.mark.unit
def test_compute_roc_pr_one_summary_per_regime() -> None:
    """Two regimes in the dict -> two summaries."""
    summaries = compute_roc_pr_curves(_toy_dict(regimes=("S5", "S5-sparse")))
    assert set(summaries.keys()) == {"S5", "S5-sparse"}
    assert all(isinstance(s, ROCPRSummary) for s in summaries.values())


@pytest.mark.unit
def test_compute_roc_pr_skips_empty_regimes() -> None:
    """Regimes with no replicates are omitted."""
    summaries = compute_roc_pr_curves({"S5": [], "S5-sparse": _toy_dict()["S5-sparse"]})
    assert list(summaries.keys()) == ["S5-sparse"]


@pytest.mark.unit
def test_compute_roc_pr_curves_have_correct_grid_size() -> None:
    """Both fpr_grid and recall_grid match the requested grid_size."""
    summaries = compute_roc_pr_curves(_toy_dict(), grid_size=51)
    s = next(iter(summaries.values()))
    assert s.fpr_grid.shape == (51,)
    assert s.recall_grid.shape == (51,)
    assert s.tpr_mean.shape == (51,)
    assert s.precision_mean.shape == (51,)


@pytest.mark.unit
def test_compute_roc_pr_aucs_in_unit_interval() -> None:
    """Per-replicate AUCs and APs lie in [0, 1]."""
    summaries = compute_roc_pr_curves(_toy_dict())
    s = next(iter(summaries.values()))
    assert np.all((s.aucs >= 0.0) & (s.aucs <= 1.0))
    assert np.all((s.aps >= 0.0) & (s.aps <= 1.0))


@pytest.mark.unit
def test_compute_roc_pr_ci_band_brackets_mean() -> None:
    """Wherever the band is defined, lower <= mean <= upper."""
    summaries = compute_roc_pr_curves(_toy_dict(n_replicates=10))
    s = next(iter(summaries.values()))
    finite = ~np.isnan(s.tpr_lo)
    assert np.all(s.tpr_lo[finite] <= s.tpr_mean[finite] + 1e-9)
    assert np.all(s.tpr_mean[finite] - 1e-9 <= s.tpr_hi[finite])


@pytest.mark.unit
def test_compute_roc_pr_single_replicate_band_is_nan() -> None:
    """With one replicate, the CI band is NaN everywhere (no variability)."""
    d = _toy_dict(n_replicates=1)
    summaries = compute_roc_pr_curves(d)
    s = next(iter(summaries.values()))
    assert np.all(np.isnan(s.tpr_lo))
    assert np.all(np.isnan(s.precision_lo))


@pytest.mark.unit
def test_compute_roc_pr_higher_separation_means_higher_mean_AUC() -> None:
    """Sanity: a regime built with separation=1.0 has higher mean AUC than separation=0.2."""
    summaries = compute_roc_pr_curves(
        _toy_dict(
            regimes=("hard", "easy"),
            separation_by_regime={"hard": 0.2, "easy": 1.0},
            n_replicates=10,
        )
    )
    assert summaries["easy"].aucs.mean() > summaries["hard"].aucs.mean()


@pytest.mark.unit
def test_compute_roc_pr_skips_degenerate_replicate() -> None:
    """Replicates with no positives (or no negatives) are silently dropped."""
    rng = np.random.default_rng(0)
    good = _synthetic_replicate(rng)
    degenerate = (np.array([0.3, 0.4, 0.5]), np.array([0, 0, 0]))
    summaries = compute_roc_pr_curves({"S5": [good, degenerate]})
    s = summaries["S5"]
    # Only one replicate contributed AUC.
    assert s.aucs.size == 1


# ---- make_roc_pr_figure: end-to-end --------------------------------------------------


@pytest.mark.integration
def test_make_roc_pr_figure_writes_pdf(tmp_path: Path) -> None:
    """End-to-end: dict of (idr, Z) -> non-empty PDF."""
    pytest.importorskip("matplotlib")
    pdf = make_roc_pr_figure(_toy_dict(), tmp_path / "f4.pdf")
    assert pdf.exists()
    assert pdf.stat().st_size > 1000


@pytest.mark.integration
def test_make_roc_pr_figure_creates_parent_dir(tmp_path: Path) -> None:
    """Parent dir is created if absent."""
    pytest.importorskip("matplotlib")
    pdf = make_roc_pr_figure(_toy_dict(regimes=("S5",)), tmp_path / "a" / "b" / "f4.pdf")
    assert pdf.exists()


@pytest.mark.integration
def test_make_roc_pr_figure_raises_when_all_regimes_empty(tmp_path: Path) -> None:
    """Empty input dict raises with a helpful message."""
    pytest.importorskip("matplotlib")
    with pytest.raises(ValueError, match="non-empty"):
        make_roc_pr_figure({}, tmp_path / "f4.pdf")
