r"""F5 reproducer: posterior-contraction diagnostic.

Reproduces ``fig:contract`` (docs/report/figures/fig_contraction.pdf) —
the empirical posterior-contraction rate of PY-IDR's Gaussian-copula
posterior, plotted as Hellinger distance to the truth vs sample size
$n$ on log-log axes.

Scope: meaningful **only for S1** today, because S1 is the only
regime whose truth is exactly a single exchangeable Gaussian copula
($\rho_{\text{true}}$ known and finite, $\rho_{\text{posterior\_mean}}$
already in the sweep CSV). For S2 the same diagnostic would apply
mechanically but the report's planned figure uses S1.

Two-tier surface, matching F1 / F2 / F3 / F4:

- :func:`compute_contraction_curves` — pure data helper. Reads the
  sweep DataFrame, computes per-row Hellinger via the closed-form
  exchangeable-Gaussian helper, groups by $n$ (and optionally $K$),
  returns mean Hellinger + bootstrap CI per group. Testable without
  matplotlib.
- :func:`make_contraction_figure` — log-log Hellinger vs $n$ with a
  fitted decay-rate line.

See plans/09_figures_and_tables.md and §3.5 of the report.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from py_idr.eval.hellinger import hellinger_gaussian_copula


@dataclass(frozen=True)
class ContractionCurve:
    r"""One regime's Hellinger-vs-$n$ contraction curve.

    Attributes
    ----------
    regime
        Regime label (typically ``"S1"``).
    n_values
        Sorted distinct feature counts $n$ observed in the CSV.
    hellinger_mean
        Mean Hellinger distance across replicates at each $n$.
    hellinger_lo, hellinger_hi
        95% bootstrap-percentile band; NaN with only one replicate.
    slope, intercept
        Log-log fit
        $\log_{10}\!\bar H = \mathrm{slope}\,\log_{10} n + \mathrm{intercept}$.
        The parametric contraction rate for a single scalar $\rho$ is
        $n^{-1/2}$, so the expected slope is around $-0.5$ when the
        chain has converged.
    n_replicates
        Number of replicates contributing at each $n$.
    """

    regime: str
    n_values: np.ndarray
    hellinger_mean: np.ndarray
    hellinger_lo: np.ndarray
    hellinger_hi: np.ndarray
    slope: float
    intercept: float
    n_replicates: np.ndarray


def compute_contraction_curves(
    df: pd.DataFrame,
    *,
    regimes: tuple[str, ...] = ("S1",),
    K: int | None = None,
    alpha: float | None = 0.05,
    method_filter: str | None = "PY-IDR (MCMC, T=1)",
    ci_level: float = 0.95,
    bootstrap_b: int = 1000,
    rng_seed: int = 0,
) -> dict[str, ContractionCurve]:
    r"""Compute per-regime Hellinger contraction curves from a sweep DataFrame.

    Parameters
    ----------
    df
        Long-format sweep DataFrame. Required columns: ``regime``, ``n``,
        ``K``, ``rho_true``, ``rho_posterior_mean``, ``alpha``,
        ``replicate_seed``, ``method``.
    regimes
        Regime labels to keep. Default ``("S1",)`` since the closed-form
        Hellinger only applies to single-Gaussian regimes.
    K
        Optional $K$ filter. ``None`` keeps every $K$ present in the
        DataFrame (and treats them all as one curve per regime).
        Passing a specific $K$ produces a single-$K$ curve.
    alpha
        Filter to one Sun-Cai operating level. Deduplicates per-replicate
        rows when the CSV carries multiple alphas per chain run.
    method_filter
        Drop rows with a different ``method``. Default keeps only the
        T = 1 fast path, which is what F5 is about; pass ``None`` to
        keep every method present.
    ci_level
        Two-sided confidence level for the bootstrap band.
    bootstrap_b
        Bootstrap resample count.
    rng_seed
        Anchor seed for the bootstrap.

    Returns
    -------
    Ordered dict ``{regime: ContractionCurve}``.

    Raises
    ------
    ValueError
        If ``df`` is missing required columns or no rows survive the filters.
    """
    required = {
        "regime",
        "n",
        "K",
        "rho_true",
        "rho_posterior_mean",
        "alpha",
        "replicate_seed",
        "method",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"df is missing required columns: {sorted(missing)}")

    sub = df.copy()
    if alpha is not None:
        sub = sub[sub["alpha"] == alpha]
    if method_filter is not None:
        sub = sub[sub["method"] == method_filter]
    if K is not None:
        sub = sub[sub["K"] == K]
    # Drop rows where the truth is undefined (S5's NaN rho_true).
    sub = sub[~sub["rho_true"].isna()]
    sub = sub[~sub["rho_posterior_mean"].isna()]

    if sub.empty:
        raise ValueError(
            "No rows survived the filters. Check alpha, method_filter, K, "
            "and that the regime has a finite rho_true."
        )

    rng = np.random.default_rng(rng_seed)
    half = (1.0 - ci_level) / 2.0
    out: dict[str, ContractionCurve] = {}

    for regime in regimes:
        regime_rows = sub[sub["regime"] == regime]
        if regime_rows.empty:
            continue
        # Compute Hellinger per row using the closed-form helper.
        h_values = []
        for _, row in regime_rows.iterrows():
            try:
                h = hellinger_gaussian_copula(
                    float(row["rho_true"]),
                    float(row["rho_posterior_mean"]),
                    K=int(row["K"]),
                )
            except ValueError:
                # rho_posterior_mean drifted outside admissible range
                # at small n — treat as a missing observation.
                h = float("nan")
            h_values.append(h)
        regime_rows = regime_rows.assign(hellinger=h_values)
        regime_rows = regime_rows[~regime_rows["hellinger"].isna()]
        if regime_rows.empty:
            continue

        n_values = np.sort(regime_rows["n"].unique())
        means = np.zeros_like(n_values, dtype=float)
        lo = np.full_like(n_values, np.nan, dtype=float)
        hi = np.full_like(n_values, np.nan, dtype=float)
        n_reps = np.zeros_like(n_values, dtype=int)

        for i, n_val in enumerate(n_values):
            cell = regime_rows[regime_rows["n"] == n_val]["hellinger"].to_numpy()
            means[i] = float(cell.mean())
            n_reps[i] = cell.size
            if cell.size >= 2:
                idx = rng.integers(0, cell.size, size=(bootstrap_b, cell.size))
                boot = cell[idx].mean(axis=1)
                lo[i] = float(np.quantile(boot, half))
                hi[i] = float(np.quantile(boot, 1.0 - half))

        # Log-log fit; only meaningful when we have >= 2 distinct n values
        # AND positive means (Hellinger=0 breaks the log).
        positive = means > 0
        if positive.sum() >= 2:
            log_n = np.log10(n_values[positive].astype(float))
            log_h = np.log10(means[positive])
            slope_, intercept_ = np.polyfit(log_n, log_h, 1)
            slope = float(slope_)
            intercept = float(intercept_)
        else:
            slope = float("nan")
            intercept = float("nan")

        out[regime] = ContractionCurve(
            regime=regime,
            n_values=n_values.astype(float),
            hellinger_mean=means,
            hellinger_lo=lo,
            hellinger_hi=hi,
            slope=slope,
            intercept=intercept,
            n_replicates=n_reps,
        )

    return out


def make_contraction_figure(
    sweep_csv: str | Path,
    out_pdf: str | Path,
    *,
    regimes: tuple[str, ...] = ("S1",),
    K: int | None = None,
    alpha: float | None = 0.05,
    method_filter: str | None = "PY-IDR (MCMC, T=1)",
    ci_level: float = 0.95,
    bootstrap_b: int = 1000,
    rng_seed: int = 0,
) -> Path:
    r"""Render the F5 posterior-contraction figure to ``out_pdf``.

    A 1-row × N-regime log-log PDF showing mean Hellinger vs $n$,
    bootstrap CI band, and the fitted slope.

    Parameters mirror :func:`compute_contraction_curves`. The figure
    is meaningful only for regimes with a finite ``rho_true`` (S1-S4);
    S5 / S5-sparse have NaN and are dropped silently.

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
            "make_contraction_figure requires matplotlib. " 'Install with: pip install -e ".[viz]"'
        ) from e

    sweep_csv = Path(sweep_csv)
    if not sweep_csv.exists():
        raise FileNotFoundError(f"sweep_csv not found: {sweep_csv}")
    df = pd.read_csv(sweep_csv)
    curves = compute_contraction_curves(
        df,
        regimes=regimes,
        K=K,
        alpha=alpha,
        method_filter=method_filter,
        ci_level=ci_level,
        bootstrap_b=bootstrap_b,
        rng_seed=rng_seed,
    )
    if not curves:
        raise ValueError(
            f"No regimes survived the filter in {sweep_csv} for "
            f"regimes={regimes}, K={K}, alpha={alpha}, method_filter={method_filter!r}."
        )

    out_pdf = Path(out_pdf)
    out_pdf.parent.mkdir(parents=True, exist_ok=True)

    plotted = list(curves.keys())
    n_panels = len(plotted)
    fig, axes = plt.subplots(1, n_panels, figsize=(3.2 * n_panels + 0.5, 3.0), sharey=True)
    if n_panels == 1:
        axes = [axes]

    color = "#1b4d3e"
    for ax, regime in zip(axes, plotted, strict=True):
        c = curves[regime]
        ax.plot(
            c.n_values,
            c.hellinger_mean,
            "o-",
            color=color,
            lw=1.6,
            label=f"mean (over {int(c.n_replicates.max())} reps max)",
        )
        finite = ~np.isnan(c.hellinger_lo)
        if finite.any():
            ax.fill_between(
                c.n_values[finite],
                c.hellinger_lo[finite],
                c.hellinger_hi[finite],
                color=color,
                alpha=0.20,
                lw=0,
                label=f"{int(ci_level * 100)}% bootstrap CI",
            )
        if np.isfinite(c.slope):
            n_grid = np.logspace(np.log10(c.n_values.min()), np.log10(c.n_values.max()), 32)
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
        ax.set_title(f"{regime} — contraction", fontsize=9)
        ax.grid(True, which="both", alpha=0.25, lw=0.4)
        ax.legend(loc="lower left", frameon=False, fontsize=7.5)

    axes[0].set_ylabel(r"Hellinger to truth ($\rho$)")
    fig.suptitle(
        r"Empirical posterior contraction: $H(c_{\hat\rho}, c_{\rho^*})$ vs $n$",
        fontsize=9,
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    return out_pdf
