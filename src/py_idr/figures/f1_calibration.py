r"""F1 reproducer: realised FDR vs nominal $\alpha$ calibration.

Reproduces ``fig:calibration`` (docs/report/figures/fig_idr_calibration.pdf)
using a long-format sweep CSV (one row per (regime, seed, $\alpha$))
produced by :func:`py_idr.simulation.sweep.write_sweep_csv`.

Module surface:

- :func:`compute_calibration_curves` — pure data helper. Groups a sweep
  DataFrame by (regime, alpha) and returns mean realised FDR with
  bootstrap-percentile 95% CI per regime. Testable without matplotlib.
- :func:`make_calibration_figure` — wraps the data helper and renders a
  1-row × N-regime panel figure to PDF.

Both functions take the long-format DataFrame layout used everywhere
in this repo: columns match :meth:`ReplicateRow.to_dict`.

See plans/09_figures_and_tables.md.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    pass


@dataclass(frozen=True)
class CalibrationCurve:
    """One regime's calibration data (mean realised FDR + CI band over alpha grid).

    Attributes
    ----------
    regime
        Regime label.
    alphas
        Nominal $\\alpha$ grid, ascending. Length $A$.
    realized_fdr_mean
        Mean realised FDR across replicates at each $\\alpha$. Length $A$.
    realized_fdr_lo, realized_fdr_hi
        Lower and upper bounds of the 95% bootstrap-percentile band.
        Length $A$ each. ``NaN`` if only one replicate per $\\alpha$.
    n_replicates
        Number of distinct replicate seeds contributing at each $\\alpha$.
    """

    regime: str
    alphas: np.ndarray
    realized_fdr_mean: np.ndarray
    realized_fdr_lo: np.ndarray
    realized_fdr_hi: np.ndarray
    n_replicates: np.ndarray


def compute_calibration_curves(
    df: pd.DataFrame,
    *,
    regimes: tuple[str, ...] = ("S1", "S2", "S4", "S5"),
    method_filter: str | None = None,
    bootstrap_b: int = 1000,
    ci_level: float = 0.95,
    rng_seed: int = 0,
) -> dict[str, CalibrationCurve]:
    """Group a sweep DataFrame by (regime, alpha) and compute mean + 95% CI.

    Parameters
    ----------
    df
        Long-format DataFrame. Must contain columns ``regime``, ``alpha``,
        ``replicate_seed``, ``realized_fdr``, and ``method``.
    regimes
        Regime labels to include, in plotting order. Missing regimes
        are silently dropped (with a warning-free omission).
    method_filter
        If supplied, keep only rows whose ``method`` matches exactly.
        ``None`` keeps all methods (typically there is only one).
    bootstrap_b
        Number of bootstrap resamples for the CI band.
    ci_level
        Two-sided confidence level. Default 0.95 → 2.5th / 97.5th
        percentile band.
    rng_seed
        Anchor seed for the bootstrap RNG; the same seed reproduces the
        bands exactly.

    Returns
    -------
    Ordered dict ``{regime: CalibrationCurve}`` covering only the
    regimes actually present in ``df`` (intersection with the
    ``regimes`` argument).

    Raises
    ------
    ValueError
        If ``df`` is missing any required column or if ``bootstrap_b``
        is non-positive.
    """
    required = {"regime", "alpha", "replicate_seed", "realized_fdr", "method"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"df is missing required columns: {sorted(missing)}")
    if bootstrap_b <= 0:
        raise ValueError(f"bootstrap_b must be > 0; got {bootstrap_b}")
    if not (0.0 < ci_level < 1.0):
        raise ValueError(f"ci_level must be in (0, 1); got {ci_level}")

    if method_filter is not None:
        df = df[df["method"] == method_filter]

    out: dict[str, CalibrationCurve] = {}
    rng = np.random.default_rng(rng_seed)
    half = (1.0 - ci_level) / 2.0

    for regime in regimes:
        sub = df[df["regime"] == regime]
        if sub.empty:
            continue
        alpha_values = np.sort(sub["alpha"].unique())
        means = np.zeros_like(alpha_values, dtype=float)
        lo = np.full_like(alpha_values, np.nan, dtype=float)
        hi = np.full_like(alpha_values, np.nan, dtype=float)
        n_reps = np.zeros_like(alpha_values, dtype=int)

        for i, a in enumerate(alpha_values):
            cell = sub[sub["alpha"] == a]["realized_fdr"].to_numpy()
            n_reps[i] = cell.size
            means[i] = float(cell.mean())
            if cell.size >= 2:
                # Bootstrap-percentile CI band.
                draws = rng.choice(cell, size=(bootstrap_b, cell.size), replace=True)
                boot_means = draws.mean(axis=1)
                lo[i] = float(np.quantile(boot_means, half))
                hi[i] = float(np.quantile(boot_means, 1.0 - half))

        out[regime] = CalibrationCurve(
            regime=regime,
            alphas=alpha_values,
            realized_fdr_mean=means,
            realized_fdr_lo=lo,
            realized_fdr_hi=hi,
            n_replicates=n_reps,
        )

    return out


def make_calibration_figure(
    sweep_csv: str | Path,
    out_pdf: str | Path,
    *,
    regimes: tuple[str, ...] = ("S1", "S2", "S4", "S5"),
    method_filter: str | None = None,
    method_label: str = "PY-IDR (MCMC, T=1)",
    ci_level: float = 0.95,
    bootstrap_b: int = 1000,
    rng_seed: int = 0,
) -> Path:
    """Render the F1 calibration figure to ``out_pdf``.

    Reads the sweep CSV at ``sweep_csv``, computes per-regime
    calibration curves via :func:`compute_calibration_curves`, and
    saves a 1-row × N-regime PDF with realised FDR vs nominal $\\alpha$,
    overlaid on the perfect-calibration diagonal.

    Parameters
    ----------
    sweep_csv
        Path to the long-format CSV.
    out_pdf
        Output PDF path. Parent directory is created if missing.
    regimes
        Regime labels to plot, in left-to-right order.
    method_filter
        If supplied, filter rows by ``method`` before aggregating.
    method_label
        Legend label for the plotted curve. Pure cosmetic.
    ci_level, bootstrap_b, rng_seed
        Bootstrap CI configuration.

    Returns
    -------
    The resolved output path.

    Raises
    ------
    ImportError
        If matplotlib is not installed (it is an optional extra).
    """
    # Lazy import — matplotlib is in the [viz] extra, not the default deps.
    try:
        import matplotlib.pyplot as plt  # type: ignore[import-not-found]
    except ImportError as e:  # pragma: no cover — exercised only when viz missing
        raise ImportError(
            "make_calibration_figure requires matplotlib. " 'Install with: pip install -e ".[viz]"'
        ) from e

    sweep_csv = Path(sweep_csv)
    if not sweep_csv.exists():
        raise FileNotFoundError(f"sweep_csv not found: {sweep_csv}")

    df = pd.read_csv(sweep_csv)
    curves = compute_calibration_curves(
        df,
        regimes=regimes,
        method_filter=method_filter,
        bootstrap_b=bootstrap_b,
        ci_level=ci_level,
        rng_seed=rng_seed,
    )
    if not curves:
        raise ValueError(
            f"No matching regimes in {sweep_csv} for regimes={regimes} "
            f"and method_filter={method_filter!r}."
        )

    out_pdf = Path(out_pdf)
    out_pdf.parent.mkdir(parents=True, exist_ok=True)

    # Plotting --------------------------------------------------------------
    plotted_regimes = list(curves.keys())
    n_panels = len(plotted_regimes)
    fig, axes = plt.subplots(1, n_panels, figsize=(2.7 * n_panels + 0.5, 2.8), sharey=True)
    if n_panels == 1:
        axes = [axes]

    alpha_max = max(float(c.alphas.max()) for c in curves.values())
    for ax, regime in zip(axes, plotted_regimes, strict=True):
        curve = curves[regime]
        # Diagonal = perfect calibration.
        ax.plot(
            [0, alpha_max],
            [0, alpha_max],
            color="black",
            lw=0.8,
            alpha=0.6,
            label="perfect calibration",
        )
        # Mean curve.
        ax.plot(
            curve.alphas,
            curve.realized_fdr_mean,
            "o-",
            color="#1b4d3e",
            lw=1.6,
            ms=4,
            label=method_label,
        )
        # CI band where available.
        finite = ~np.isnan(curve.realized_fdr_lo)
        if finite.any():
            ax.fill_between(
                curve.alphas[finite],
                curve.realized_fdr_lo[finite],
                curve.realized_fdr_hi[finite],
                color="#1b4d3e",
                alpha=0.20,
                lw=0,
                label=f"{int(ci_level * 100)}% bootstrap CI",
            )
        ax.set_xlim(0, alpha_max)
        ax.set_ylim(0, alpha_max * 1.5)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.25, lw=0.4)
        ax.set_xlabel(r"nominal $\alpha$")
        ax.set_title(regime)

    axes[0].set_ylabel("realised FDR")
    # Legend on the rightmost panel only.
    axes[-1].legend(loc="upper left", frameon=False, fontsize=7.5)

    n_seeds = max(int(c.n_replicates.max()) for c in curves.values())
    fig.suptitle(
        f"Realised FDR vs nominal $\\alpha$ (mean over {n_seeds} replicate seeds)",
        fontsize=9,
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    return out_pdf
