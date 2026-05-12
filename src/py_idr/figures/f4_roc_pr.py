r"""F4 reproducer: ROC and precision-recall curves.

Reproduces ``fig:roc-pr`` (docs/report/figures/fig_roc_pr.pdf) — the
discrimination diagnostic for the S5-sparse stress test where the
prior expectation is that PY-IDR's multi-cluster atoms beat
Gaussian-only baselines under heavy-tail dependence.

The figure consumes per-replicate ``(idr, true_Z)`` arrays directly
in memory rather than via the sweep CSV, because ROC / PR are curves
(not scalars) and the sweep CSV's column-per-row layout cannot carry
the full per-feature posterior. A CLI / persistence integration is
left to a follow-on (it will store the arrays as a per-run NPZ
sidecar alongside ``results.csv``).

Two-tier surface, matching F1 / F2 / F3:

- :func:`compute_roc_pr_curves` — pure data helper. Takes a dict
  ``{regime: [(idr_array, Z_array), ...]}``, computes per-replicate
  ROC and PR curves via :mod:`py_idr.decision.roc_pr`, interpolates
  onto a shared FPR grid (for ROC) and recall grid (for PR), returns
  mean curves + bootstrap CI bands.
- :func:`make_roc_pr_figure` — renders a 2-row × N-regime PDF (ROC
  on top, PR on bottom) with the diagonal for ROC and the
  trivial-classifier line for PR overlaid as reference.

See plans/09_figures_and_tables.md.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from py_idr.decision.roc_pr import auc, average_precision, pr_curve, roc_curve


@dataclass(frozen=True)
class ROCPRSummary:
    """Per-regime ROC + PR summary across replicates.

    Attributes
    ----------
    regime
        Regime label.
    fpr_grid
        Common FPR grid used to align replicate ROC curves before
        averaging. Length $G$.
    tpr_mean, tpr_lo, tpr_hi
        Mean TPR and 95% bootstrap-percentile band evaluated on the
        FPR grid. Length $G$.
    recall_grid
        Common recall grid for the PR curves. Length $G$.
    precision_mean, precision_lo, precision_hi
        Mean precision and 95% band on the recall grid. Length $G$.
    aucs, aps
        Per-replicate AUC and AP scalars. Length = number of
        replicates contributing to this regime.
    """

    regime: str
    fpr_grid: np.ndarray
    tpr_mean: np.ndarray
    tpr_lo: np.ndarray
    tpr_hi: np.ndarray
    recall_grid: np.ndarray
    precision_mean: np.ndarray
    precision_lo: np.ndarray
    precision_hi: np.ndarray
    aucs: np.ndarray
    aps: np.ndarray


def _interp_tpr(roc, fpr_grid: np.ndarray) -> np.ndarray:  # type: ignore[no-untyped-def]
    """Step-function interpolate TPR onto fpr_grid.

    The ROC curve is piecewise-constant in TPR between sample points;
    we use ``np.interp`` on the (fpr, tpr) pairs which gives
    linear interpolation — close enough for the band overlay at
    research-grade granularity.
    """
    return np.interp(fpr_grid, roc.fpr, roc.tpr)


def _interp_precision(pr, recall_grid: np.ndarray) -> np.ndarray:  # type: ignore[no-untyped-def]
    """Interpolate precision onto recall_grid via linear interpolation.

    PR curves are non-monotone in precision; we use the standard
    "interpolated precision" (max precision at recall >= each grid
    point) for the band, matching scikit-learn's
    ``precision_recall_curve``-based plots.
    """
    # Sort recall ascending then take running max of precision from the
    # right so the interpolated precision is monotone non-increasing.
    order = np.argsort(pr.recall)
    r_sorted = pr.recall[order]
    p_sorted = pr.precision[order]
    p_monotone = np.maximum.accumulate(p_sorted[::-1])[::-1]
    return np.interp(recall_grid, r_sorted, p_monotone)


def compute_roc_pr_curves(
    idr_z_by_regime: Mapping[str, Sequence[tuple[np.ndarray, np.ndarray]]],
    *,
    regimes: tuple[str, ...] | None = None,
    grid_size: int = 101,
    ci_level: float = 0.95,
    bootstrap_b: int = 500,
    rng_seed: int = 0,
) -> dict[str, ROCPRSummary]:
    r"""Aggregate per-replicate ROC and PR curves into mean + 95% band per regime.

    Parameters
    ----------
    idr_z_by_regime
        Mapping ``{regime: [(idr_1, Z_1), (idr_2, Z_2), ...]}`` of
        per-replicate (local idr, ground-truth Z) arrays. Each
        ``idr`` is length-$n$ in $[0, 1]$; each ``Z`` is length-$n$
        in $\\{0, 1\\}$.
    regimes
        Regime order to include. ``None`` keeps every key present.
    grid_size
        Number of points in the shared FPR / recall grid. Default
        101 = 1% resolution.
    ci_level, bootstrap_b, rng_seed
        Bootstrap configuration for the band.

    Returns
    -------
    Ordered dict ``{regime: ROCPRSummary}`` covering only the regimes
    with at least one replicate.

    Raises
    ------
    ValueError
        If ``grid_size`` is < 2 or ``ci_level`` is out of range.
    """
    if grid_size < 2:
        raise ValueError(f"grid_size must be >= 2; got {grid_size}")
    if not (0.0 < ci_level < 1.0):
        raise ValueError(f"ci_level must be in (0, 1); got {ci_level}")

    fpr_grid = np.linspace(0.0, 1.0, grid_size)
    recall_grid = np.linspace(0.0, 1.0, grid_size)
    half = (1.0 - ci_level) / 2.0
    rng = np.random.default_rng(rng_seed)

    if regimes is None:
        regimes = tuple(idr_z_by_regime.keys())

    out: dict[str, ROCPRSummary] = {}
    for regime in regimes:
        replicates = idr_z_by_regime.get(regime, [])
        if not replicates:
            continue
        tpr_mat = []
        prec_mat = []
        aucs = []
        aps = []
        for idr_arr, Z_arr in replicates:
            try:
                roc = roc_curve(idr_arr, Z_arr)
                pr = pr_curve(idr_arr, Z_arr)
            except ValueError:
                # Degenerate replicate (all-pos or all-neg). Skip.
                continue
            tpr_mat.append(_interp_tpr(roc, fpr_grid))
            prec_mat.append(_interp_precision(pr, recall_grid))
            aucs.append(auc(roc))
            aps.append(average_precision(pr))

        if not tpr_mat:
            continue

        tpr_arr = np.stack(tpr_mat, axis=0)
        prec_arr = np.stack(prec_mat, axis=0)

        tpr_mean = tpr_arr.mean(axis=0)
        prec_mean = prec_arr.mean(axis=0)

        if tpr_arr.shape[0] >= 2:
            # Bootstrap-percentile band over the replicate index.
            n_rep = tpr_arr.shape[0]
            sample_idx = rng.integers(0, n_rep, size=(bootstrap_b, n_rep))
            tpr_boot = tpr_arr[sample_idx].mean(axis=1)  # (B, G)
            prec_boot = prec_arr[sample_idx].mean(axis=1)
            tpr_lo = np.quantile(tpr_boot, half, axis=0)
            tpr_hi = np.quantile(tpr_boot, 1.0 - half, axis=0)
            prec_lo = np.quantile(prec_boot, half, axis=0)
            prec_hi = np.quantile(prec_boot, 1.0 - half, axis=0)
        else:
            tpr_lo = np.full_like(tpr_mean, np.nan)
            tpr_hi = np.full_like(tpr_mean, np.nan)
            prec_lo = np.full_like(prec_mean, np.nan)
            prec_hi = np.full_like(prec_mean, np.nan)

        out[regime] = ROCPRSummary(
            regime=regime,
            fpr_grid=fpr_grid,
            tpr_mean=tpr_mean,
            tpr_lo=tpr_lo,
            tpr_hi=tpr_hi,
            recall_grid=recall_grid,
            precision_mean=prec_mean,
            precision_lo=prec_lo,
            precision_hi=prec_hi,
            aucs=np.asarray(aucs),
            aps=np.asarray(aps),
        )

    return out


def make_roc_pr_figure(
    idr_z_by_regime: Mapping[str, Sequence[tuple[np.ndarray, np.ndarray]]],
    out_pdf: str | Path,
    *,
    regimes: tuple[str, ...] | None = None,
    grid_size: int = 101,
    ci_level: float = 0.95,
    bootstrap_b: int = 500,
    rng_seed: int = 0,
) -> Path:
    """Render F4 (ROC + PR) to ``out_pdf``.

    A 2-row × N-regime PDF: ROC on top, PR on bottom. Each panel shows
    the mean curve, the bootstrap CI band, and a reference line (the
    diagonal for ROC, the trivial-classifier precision = prevalence
    for PR). Panel titles include mean AUC / AP across replicates.

    Parameters
    ----------
    idr_z_by_regime
        Per-replicate ``(idr, Z)`` arrays per regime; same shape as
        :func:`compute_roc_pr_curves`.
    out_pdf
        Output PDF path. Parent directory is created if missing.
    regimes
        Regime order. ``None`` keeps every key present.
    grid_size, ci_level, bootstrap_b, rng_seed
        Forwarded to :func:`compute_roc_pr_curves`.

    Returns
    -------
    The resolved output path.

    Raises
    ------
    ImportError
        If matplotlib is not installed.
    ValueError
        If no regimes survive the filter.
    """
    try:
        import matplotlib.pyplot as plt  # type: ignore[import-not-found]
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "make_roc_pr_figure requires matplotlib. " 'Install with: pip install -e ".[viz]"'
        ) from e

    summaries = compute_roc_pr_curves(
        idr_z_by_regime,
        regimes=regimes,
        grid_size=grid_size,
        ci_level=ci_level,
        bootstrap_b=bootstrap_b,
        rng_seed=rng_seed,
    )
    if not summaries:
        raise ValueError(
            "No regimes with at least one valid replicate. "
            "Did the input dict contain non-empty (idr, Z) pairs?"
        )

    out_pdf = Path(out_pdf)
    out_pdf.parent.mkdir(parents=True, exist_ok=True)

    plotted = list(summaries.keys())
    n_panels = len(plotted)
    fig, axes = plt.subplots(
        2,
        n_panels,
        figsize=(3.0 * n_panels + 0.5, 5.5),
        sharex="row",
        sharey="row",
    )
    if n_panels == 1:
        axes = axes.reshape(2, 1)

    color = "#1b4d3e"

    for j, regime in enumerate(plotted):
        s = summaries[regime]
        # ROC.
        ax_roc = axes[0, j]
        ax_roc.plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.6, label="chance")
        ax_roc.plot(
            s.fpr_grid,
            s.tpr_mean,
            "-",
            color=color,
            lw=1.6,
            label=f"mean (AUC = {s.aucs.mean():.3f})",
        )
        if not np.isnan(s.tpr_lo).all():
            ax_roc.fill_between(
                s.fpr_grid,
                s.tpr_lo,
                s.tpr_hi,
                color=color,
                alpha=0.20,
                lw=0,
                label=f"{int(ci_level * 100)}% CI",
            )
        ax_roc.set_xlim(0, 1)
        ax_roc.set_ylim(0, 1)
        ax_roc.set_xlabel("FPR")
        ax_roc.set_title(f"{regime} — ROC", fontsize=9)
        ax_roc.grid(True, alpha=0.25, lw=0.4)
        if j == 0:
            ax_roc.set_ylabel("TPR")
        ax_roc.legend(loc="lower right", frameon=False, fontsize=7.5)

        # PR.
        ax_pr = axes[1, j]
        ax_pr.plot(
            s.recall_grid,
            s.precision_mean,
            "-",
            color=color,
            lw=1.6,
            label=f"mean (AP = {s.aps.mean():.3f})",
        )
        if not np.isnan(s.precision_lo).all():
            ax_pr.fill_between(
                s.recall_grid,
                s.precision_lo,
                s.precision_hi,
                color=color,
                alpha=0.20,
                lw=0,
                label=f"{int(ci_level * 100)}% CI",
            )
        ax_pr.set_xlim(0, 1)
        ax_pr.set_ylim(0, 1)
        ax_pr.set_xlabel("recall")
        ax_pr.set_title(f"{regime} — PR", fontsize=9)
        ax_pr.grid(True, alpha=0.25, lw=0.4)
        if j == 0:
            ax_pr.set_ylabel("precision")
        ax_pr.legend(loc="lower left", frameon=False, fontsize=7.5)

    fig.suptitle(
        "ROC and Precision-Recall across regimes (mean over replicates)",
        fontsize=9,
        y=1.01,
    )
    fig.tight_layout()
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    return out_pdf
