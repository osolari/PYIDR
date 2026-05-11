r"""F3 reproducer: chain walltime scaling vs problem size.

Reproduces ``fig:runtime`` (docs/report/figures/fig_runtime.pdf) — the
empirical walltime scaling of the PY-IDR chain. We plot
``walltime_s`` against ``n`` on log-log axes, with one panel per
``(regime, method)`` combination. Slope of the fitted line is the
empirical exponent — useful for sanity-checking against the analytic
$O(nKT + K^3 T)$ per-sweep cost.

Two-tier surface, matching F1 / F2:

- :func:`compute_runtime_curves` — pure data helper. Groups by
  ``(regime, method, n)``, returns per-cell mean walltime + log-linear
  slope. Testable without matplotlib.
- :func:`make_runtime_figure` — reads a CSV, renders a 1-row × N-cell
  log-log scatter with a fitted line per cell.

See plans/09_figures_and_tables.md.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class RuntimeCurve:
    """Walltime-vs-$n$ curve for one (regime, method) cell.

    Attributes
    ----------
    regime, method
        Identifying labels.
    n_values
        Sorted distinct $n$ values that appear in the cell.
    walltime_mean_s, walltime_std_s
        Per-$n$ mean and standard deviation of ``walltime_s`` across
        replicates. Same length as ``n_values``.
    slope, intercept
        Coefficients of ``log10(walltime) = slope * log10(n) + intercept``
        fitted across the cell. Use ``slope`` as an empirical
        complexity exponent: 1.0 ≈ linear, 2.0 ≈ quadratic, etc.
        ``NaN`` if fewer than 2 distinct $n$ values.
    """

    regime: str
    method: str
    n_values: np.ndarray
    walltime_mean_s: np.ndarray
    walltime_std_s: np.ndarray
    slope: float
    intercept: float


def compute_runtime_curves(
    df: pd.DataFrame,
    *,
    alpha: float | None = 0.05,
    regimes: tuple[str, ...] | None = None,
    methods: tuple[str, ...] | None = None,
) -> dict[tuple[str, str], RuntimeCurve]:
    """Aggregate ``walltime_s`` over replicates per ``(regime, method, n)``.

    Parameters
    ----------
    df
        Long-format sweep DataFrame. Must contain columns ``regime``,
        ``method``, ``n``, ``alpha``, ``walltime_s``, ``replicate_seed``.
    alpha
        Optional $\\alpha$ filter — passing a single value deduplicates
        the per-replicate row when the CSV carries multiple $\\alpha$
        rows per chain run. ``None`` keeps all rows.
    regimes, methods
        Optional regime / method allow-lists. ``None`` (default for
        each) keeps every value present in the DataFrame.

    Returns
    -------
    Ordered dict ``{(regime, method): RuntimeCurve}``. Iteration order
    matches the DataFrame's row order.

    Raises
    ------
    ValueError
        If ``df`` is missing required columns.
    """
    required = {"regime", "method", "n", "alpha", "walltime_s", "replicate_seed"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"df is missing required columns: {sorted(missing)}")

    sub = df
    if alpha is not None:
        sub = sub[sub["alpha"] == alpha]
    if regimes is not None:
        sub = sub[sub["regime"].isin(regimes)]
    if methods is not None:
        sub = sub[sub["method"].isin(methods)]

    out: dict[tuple[str, str], RuntimeCurve] = {}
    for (regime, method), cell in sub.groupby(["regime", "method"], sort=False):
        agg = cell.groupby("n")["walltime_s"].agg(["mean", "std"])
        n_values = np.asarray(agg.index.to_numpy(), dtype=float)
        means = np.asarray(agg["mean"].to_numpy(), dtype=float)
        stds = np.asarray(agg["std"].to_numpy(), dtype=float)
        # std is NaN when only one replicate at that n; fill 0 to make plotting
        # error bars trivial.
        stds = np.where(np.isnan(stds), 0.0, stds)

        # Log-linear fit; NaN slope when <2 distinct n values.
        if n_values.size >= 2:
            log_n = np.log10(n_values)
            log_w = np.log10(np.maximum(means, 1e-12))
            slope_, intercept_ = np.polyfit(log_n, log_w, 1)
            slope = float(slope_)
            intercept = float(intercept_)
        else:
            slope = float("nan")
            intercept = float("nan")

        out[(regime, method)] = RuntimeCurve(
            regime=regime,
            method=method,
            n_values=n_values,
            walltime_mean_s=means,
            walltime_std_s=stds,
            slope=slope,
            intercept=intercept,
        )

    return out


def make_runtime_figure(
    sweep_csv: str | Path,
    out_pdf: str | Path,
    *,
    alpha: float | None = 0.05,
    regimes: tuple[str, ...] | None = None,
    methods: tuple[str, ...] | None = None,
) -> Path:
    """Render the F3 runtime figure to ``out_pdf``.

    Reads the sweep CSV at ``sweep_csv``, computes per-(regime, method)
    walltime curves via :func:`compute_runtime_curves`, and writes a
    log-log scatter PDF with one panel per cell.

    Parameters
    ----------
    sweep_csv
        Path to a long-format CSV produced by
        :func:`py_idr.simulation.sweep.write_sweep_csv`.
    out_pdf
        Output PDF path. Parent directory is created if missing.
    alpha
        Filter to one $\\alpha$. Default 0.05.
    regimes, methods
        Optional allow-lists; ``None`` keeps every regime / method.

    Returns
    -------
    The resolved output path.

    Raises
    ------
    ImportError
        If matplotlib is not installed.
    FileNotFoundError
        If ``sweep_csv`` does not exist.
    ValueError
        If no rows survive the filters.
    """
    try:
        import matplotlib.pyplot as plt  # type: ignore[import-not-found]
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "make_runtime_figure requires matplotlib. " 'Install with: pip install -e ".[viz]"'
        ) from e

    sweep_csv = Path(sweep_csv)
    if not sweep_csv.exists():
        raise FileNotFoundError(f"sweep_csv not found: {sweep_csv}")

    df = pd.read_csv(sweep_csv)
    curves = compute_runtime_curves(
        df,
        alpha=alpha,
        regimes=regimes,
        methods=methods,
    )
    if not curves:
        raise ValueError(
            f"No matching rows in {sweep_csv} for alpha={alpha}, "
            f"regimes={regimes}, methods={methods}."
        )

    out_pdf = Path(out_pdf)
    out_pdf.parent.mkdir(parents=True, exist_ok=True)

    cells = list(curves.keys())
    n_panels = len(cells)
    fig, axes = plt.subplots(1, n_panels, figsize=(3.0 * n_panels + 0.5, 3.0), sharey=True)
    if n_panels == 1:
        axes = [axes]

    for ax, key in zip(axes, cells, strict=True):
        c = curves[key]
        ax.errorbar(
            c.n_values,
            c.walltime_mean_s,
            yerr=c.walltime_std_s,
            fmt="o-",
            color="#1b4d3e",
            ecolor="#1b4d3e80",
            capsize=3,
            lw=1.5,
            label="empirical",
        )
        if np.isfinite(c.slope):
            n_grid = np.logspace(
                np.log10(c.n_values.min()),
                np.log10(c.n_values.max()),
                32,
            )
            fit = 10 ** (c.slope * np.log10(n_grid) + c.intercept)
            ax.plot(
                n_grid,
                fit,
                "--",
                color="black",
                lw=0.8,
                alpha=0.6,
                label=f"fit (slope = {c.slope:.2f})",
            )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("$n$ (feature count)")
        ax.set_title(f"{c.regime} / {c.method}", fontsize=8.5)
        ax.grid(True, which="both", alpha=0.25, lw=0.4)
        ax.legend(loc="upper left", frameon=False, fontsize=7.5)

    axes[0].set_ylabel("chain walltime (s)")
    fig.suptitle("Walltime vs $n$ — empirical scaling (log-log)", fontsize=9, y=1.02)
    fig.tight_layout()
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    return out_pdf
