r"""F2 reproducer: posterior cluster-count distribution per regime.

Reproduces ``fig:py-vs-dp`` (docs/report/figures/fig_py_vs_dp.pdf) — the
cluster-occupancy diagnostic that distinguishes PY ($\sigma > 0$) from
the DP special case. We plot the per-regime histogram of the
**posterior mean** of $T$ (active cluster count) across the replicates
in a sweep, taking the ``T_posterior_mean`` column from the long-format
sweep CSV.

Two-tier surface (matches F1):

- :func:`compute_T_histograms` — pure data helper. Filters the
  DataFrame to a single $\alpha$, optionally to one method, and groups
  by regime. Returns a dict ``{regime: TClusterCountHistogram}``.
  Testable without matplotlib.
- :func:`make_cluster_count_figure` — reads a CSV, calls the helper,
  renders a 1-row × N-regime histogram PDF.

The figure is meaningful only for the multi-cluster path. Rows from
the T=1 path have ``T_posterior_mean = NaN`` and are dropped.

See plans/09_figures_and_tables.md.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class TClusterCountHistogram:
    """One regime's posterior-mean-$T$ summary across replicates.

    Attributes
    ----------
    regime
        Regime label.
    T_means
        1-D array of per-replicate ``T_posterior_mean`` values
        (with NaN entries dropped). Length = number of valid
        replicates.
    bin_edges, bin_counts
        Histogram with integer-aligned bins covering the observed
        range. ``len(bin_edges) == len(bin_counts) + 1``.
    overall_mean
        Mean of ``T_means`` — a single scalar summary of how many
        clusters the chain typically uses for this regime.
    """

    regime: str
    T_means: np.ndarray
    bin_edges: np.ndarray
    bin_counts: np.ndarray
    overall_mean: float


def compute_T_histograms(
    df: pd.DataFrame,
    *,
    regimes: tuple[str, ...] = ("S5", "S5-sparse"),
    alpha: float | None = None,
    method_filter: str | None = "PY-IDR (MCMC, T>1)",
    bin_width: float = 0.5,
) -> dict[str, TClusterCountHistogram]:
    """Group sweep rows by regime and summarise ``T_posterior_mean``.

    Parameters
    ----------
    df
        Long-format sweep DataFrame. Must contain columns ``regime``,
        ``replicate_seed``, ``T_posterior_mean``, and (if
        ``method_filter`` is supplied) ``method``. If ``alpha`` is
        supplied the rows are filtered to that operating level so each
        replicate contributes exactly one value.
    regimes
        Regime labels to include, in plotting order. Missing regimes
        are silently omitted.
    alpha
        Optional alpha filter. When the sweep CSV has multiple
        ($\\alpha$) rows per replicate, pass one to deduplicate to
        per-replicate values. ``None`` keeps all rows (will
        double-count if more than one alpha per replicate).
    method_filter
        If supplied, keep only rows whose ``method`` matches exactly.
        Default ``"PY-IDR (MCMC, T>1)"`` selects only the
        multi-cluster path (the only one for which T is meaningful).
    bin_width
        Histogram bin width on the continuous T_posterior_mean axis.
        Default 0.5 gives a smooth-looking PMF-style chart.

    Returns
    -------
    Ordered dict ``{regime: TClusterCountHistogram}``. Regimes with
    zero valid rows after filtering are omitted.

    Raises
    ------
    ValueError
        If ``df`` is missing required columns or ``bin_width`` is
        non-positive.
    """
    required = {"regime", "replicate_seed", "T_posterior_mean"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"df is missing required columns: {sorted(missing)}")
    if bin_width <= 0:
        raise ValueError(f"bin_width must be > 0; got {bin_width}")

    df_filt = df
    if method_filter is not None:
        if "method" not in df_filt.columns:
            raise ValueError("df has no 'method' column; cannot apply method_filter")
        df_filt = df_filt[df_filt["method"] == method_filter]
    if alpha is not None:
        df_filt = df_filt[df_filt["alpha"] == alpha]

    out: dict[str, TClusterCountHistogram] = {}
    for regime in regimes:
        sub = df_filt[df_filt["regime"] == regime]
        if sub.empty:
            continue
        T_means = np.asarray(sub["T_posterior_mean"].to_numpy(), dtype=float)
        T_means = T_means[~np.isnan(T_means)]
        if T_means.size == 0:
            continue
        lo = max(1.0, float(np.floor(T_means.min())))
        hi = float(np.ceil(T_means.max())) + 1.0
        edges = np.arange(lo, hi + bin_width, bin_width)
        counts, _ = np.histogram(T_means, bins=edges)
        out[regime] = TClusterCountHistogram(
            regime=regime,
            T_means=T_means,
            bin_edges=edges,
            bin_counts=counts,
            overall_mean=float(T_means.mean()),
        )
    return out


def make_cluster_count_figure(
    sweep_csv: str | Path,
    out_pdf: str | Path,
    *,
    regimes: tuple[str, ...] = ("S5", "S5-sparse"),
    alpha: float | None = 0.05,
    method_filter: str | None = "PY-IDR (MCMC, T>1)",
    bin_width: float = 0.5,
) -> Path:
    """Render the F2 cluster-count figure to ``out_pdf``.

    Reads the sweep CSV at ``sweep_csv``, computes per-regime
    ``T_posterior_mean`` histograms via :func:`compute_T_histograms`,
    and writes a 1-row × N-regime PDF.

    Parameters
    ----------
    sweep_csv
        Path to a long-format CSV produced by
        :func:`py_idr.simulation.sweep.write_sweep_csv`.
    out_pdf
        Output PDF path. Parent directory is created if missing.
    regimes
        Regime labels to plot, in left-to-right order. Default
        ``("S5", "S5-sparse")`` — the two regimes whose data actually
        exercises the multi-cluster path.
    alpha
        Filter to one $\\alpha$ to avoid double-counting replicates.
        Default 0.05.
    method_filter
        Default keeps only the multi-cluster method rows.
    bin_width
        Histogram bin width on the continuous T axis.

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
        If no regimes survive the filter.
    """
    try:
        import matplotlib.pyplot as plt  # type: ignore[import-not-found]
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "make_cluster_count_figure requires matplotlib. "
            'Install with: pip install -e ".[viz]"'
        ) from e

    sweep_csv = Path(sweep_csv)
    if not sweep_csv.exists():
        raise FileNotFoundError(f"sweep_csv not found: {sweep_csv}")

    df = pd.read_csv(sweep_csv)
    hists = compute_T_histograms(
        df,
        regimes=regimes,
        alpha=alpha,
        method_filter=method_filter,
        bin_width=bin_width,
    )
    if not hists:
        raise ValueError(
            f"No matching regimes in {sweep_csv} for regimes={regimes}, "
            f"alpha={alpha}, method_filter={method_filter!r} after "
            "dropping NaN T_posterior_mean rows. Did the sweep route "
            "through the multi-cluster chain?"
        )

    out_pdf = Path(out_pdf)
    out_pdf.parent.mkdir(parents=True, exist_ok=True)

    plotted = list(hists.keys())
    n_panels = len(plotted)
    fig, axes = plt.subplots(1, n_panels, figsize=(2.8 * n_panels + 0.5, 2.8), sharey=True)
    if n_panels == 1:
        axes = [axes]

    for ax, regime in zip(axes, plotted, strict=True):
        h = hists[regime]
        centers = 0.5 * (h.bin_edges[:-1] + h.bin_edges[1:])
        widths = np.diff(h.bin_edges)
        ax.bar(
            centers,
            h.bin_counts,
            width=widths,
            align="center",
            color="#1b4d3e",
            alpha=0.7,
            edgecolor="black",
            linewidth=0.4,
        )
        ax.axvline(
            h.overall_mean,
            color="red",
            lw=1.5,
            linestyle="--",
            label=f"mean = {h.overall_mean:.2f}",
        )
        ax.set_xlabel(r"posterior mean $T$ (active clusters)")
        ax.set_title(f"{regime}  (n_rep = {len(h.T_means)})")
        ax.grid(True, alpha=0.25, lw=0.4)
        ax.legend(loc="upper right", frameon=False, fontsize=8)

    axes[0].set_ylabel("# replicates")
    fig.suptitle(
        "Posterior cluster-count distribution across replicates " f"(method = {method_filter})",
        fontsize=9,
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    return out_pdf
