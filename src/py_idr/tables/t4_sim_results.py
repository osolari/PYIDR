r"""T4 reproducer: realised FDR / power at $\alpha=0.05$ across regimes.

Reproduces ``tab:sim-fdr`` (and the structurally identical
``tab:sim-power``) from docs/report/tables/table_sim_results.tex.

Two-tier surface, mirroring the F1 reproducer:

- :func:`compute_summary_table` — pure data helper. Filters the sweep
  DataFrame to one $\alpha$, groups by (method, regime), and returns
  per-cell mean + standard error. Testable without LaTeX.
- :func:`format_latex_table` — emit a LaTeX tabular string for a given
  summary DataFrame. Methods we haven't implemented yet (Vanilla IDR,
  eCV, nestedIDR, MaRR, ChIP-R, GMCM-AD, BNP-Archimedean, PY-IDR (VI))
  appear as placeholder rows so the structure matches the report.
- :func:`make_simulation_results_table` — convenience wrapper:
  ``sweep_csv → summary → LaTeX string → .tex file``.

See plans/09_figures_and_tables.md.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# Method label as it appears in our ReplicateRow.method field.
_PY_IDR_MCMC = "PY-IDR (MCMC, T=1)"

# Full ordered method list as it appears in the report's tab:sim-fdr.
# Labels match exactly what the sweep CSV's `method` column produces (see
# `py_idr.simulation.replicates`); methods not yet integrated in the package
# (and therefore absent from any sweep CSV) become placeholder rows.
_REPORT_METHODS_ORDER: tuple[str, ...] = (
    "Vanilla IDR (pairwise)",
    "eCV",
    "nestedIDR",
    "MaxRank",
    "ChIP-R",
    "GMCM-AD",
    "BNP-Archimedean",
    "PY-IDR (MCMC, T=1)",
    "PY-IDR (MCMC, T>1)",
    "PY-IDR (VI)",
)


def compute_summary_table(
    df: pd.DataFrame,
    *,
    metric: str = "realized_fdr",
    alpha: float = 0.05,
    regimes: tuple[str, ...] = ("S1", "S2", "S3", "S4", "S5"),
) -> pd.DataFrame:
    """Summarise ``metric`` (mean + SE) per (method, regime) at one $\\alpha$.

    Parameters
    ----------
    df
        Long-format DataFrame (one row per replicate × $\\alpha$). Must
        contain columns ``regime``, ``alpha``, ``method``, ``replicate_seed``,
        and the requested ``metric``.
    metric
        Column to summarise. Default ``"realized_fdr"`` matches
        ``tab:sim-fdr``; pass ``"power"`` for ``tab:sim-power``.
    alpha
        Sun-Cai operating level to filter on. Rows with a different
        ``alpha`` value are dropped.
    regimes
        Regime labels to include, in column order. Missing regimes
        produce all-NaN columns.

    Returns
    -------
    A DataFrame indexed by ``method`` with columns
    ``[("mean", regime), ("se", regime), ("n", regime)]`` for each
    regime — i.e. a MultiIndex column layout suitable for
    :func:`format_latex_table`.

    Raises
    ------
    ValueError
        If ``df`` is missing required columns or ``metric`` is absent.
    """
    required = {"regime", "alpha", "method", "replicate_seed", metric}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"df is missing required columns: {sorted(missing)}")

    sub = df[df["alpha"] == alpha].copy()
    if sub.empty:
        raise ValueError(f"No rows in df with alpha == {alpha}")

    # Group per (method, regime) — robust to "S5-sparse" containing a hyphen.
    grouped = sub.groupby(["method", "regime"], dropna=False)[metric].agg(["mean", "std", "count"])
    grouped["se"] = grouped["std"] / np.sqrt(grouped["count"])
    grouped = grouped.drop(columns="std").rename(columns={"count": "n"})

    # Pivot to (method × regime) with mean / se / n columns.
    pivoted = grouped.unstack("regime")  # columns: (mean | se | n) × regime

    # Reindex columns so regimes appear in user-specified order; missing → NaN.
    cols = pd.MultiIndex.from_product([["mean", "se", "n"], list(regimes)])
    pivoted = pivoted.reindex(columns=cols)
    return pivoted


def format_latex_table(
    summary: pd.DataFrame,
    *,
    metric_label: str = "realised FDR",
    alpha: float = 0.05,
    caption: str | None = None,
    label: str = "tab:sim-fdr",
    bold_method: str | None = _PY_IDR_MCMC,
    methods_order: tuple[str, ...] = _REPORT_METHODS_ORDER,
    placeholder: str = "---",
    precision: int = 3,
) -> str:
    r"""Emit a LaTeX ``tabular`` string for the per-method summary.

    Cell layout: ``mean (se)`` with both formatted to ``precision``
    decimal places. Methods not present in ``summary`` get a row of
    ``placeholder`` strings. ``bold_method`` is wrapped in
    ``\textbf{}``.

    Parameters
    ----------
    summary
        Output of :func:`compute_summary_table`.
    metric_label
        Human-readable metric for the caption.
    alpha
        Operating level — appears in the caption.
    caption
        Optional override for the auto-generated caption.
    label
        LaTeX ``\\label``.
    bold_method
        Method label to render in ``\\textbf{}`` (typically the one
        we want to highlight). ``None`` disables.
    methods_order
        Row ordering. Methods absent from ``summary`` emit placeholders.
    placeholder
        Cell content for absent methods.
    precision
        Decimal places.

    Returns
    -------
    A LaTeX-ready string starting with ``\\begin{table}`` and ending
    with ``\\end{table}``.
    """
    regimes = list(summary.columns.get_level_values(1).unique())
    if caption is None:
        caption = (
            f"{metric_label} at nominal $\\alpha = {alpha}$ across simulation "
            f"regimes {' / '.join(regimes)}, mean over replicate seeds with "
            f"Monte Carlo standard error in parentheses."
        )

    n_cols = len(regimes)
    col_spec = "l" + "c" * n_cols

    lines: list[str] = []
    lines.append(r"\begin{table}[H]")
    lines.append(r"\centering")
    lines.append(rf"\caption{{{caption}}}")
    lines.append(rf"\label{{{label}}}")
    lines.append(r"\scriptsize")
    lines.append(rf"\begin{{tabular}}{{{col_spec}}}")
    lines.append(r"\toprule")
    header = "Method & " + " & ".join(regimes) + r" \\"
    lines.append(header)
    lines.append(r"\midrule")

    for method in methods_order:
        row_label = (
            rf"\textbf{{{method}}}"
            if (bold_method is not None and method == bold_method)
            else method
        )
        if method not in summary.index:
            cells = [placeholder] * n_cols
        else:
            cells = []
            for r in regimes:
                m = summary.loc[method, ("mean", r)]
                s = summary.loc[method, ("se", r)]
                if pd.isna(m):
                    cells.append(placeholder)
                elif pd.isna(s):
                    cells.append(f"{m:.{precision}f}")
                else:
                    cells.append(f"{m:.{precision}f} ({s:.{precision}f})")
        lines.append(f"{row_label} & " + " & ".join(cells) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines) + "\n"


def make_simulation_results_table(
    sweep_csv: str | Path,
    out_tex: str | Path,
    *,
    metric: str = "realized_fdr",
    alpha: float = 0.05,
    regimes: tuple[str, ...] = ("S1", "S2", "S3", "S4", "S5"),
    bold_method: str | None = _PY_IDR_MCMC,
    caption: str | None = None,
    label: str = "tab:sim-fdr",
    precision: int = 3,
) -> Path:
    """End-to-end: read sweep CSV, build summary, write LaTeX ``.tex`` file.

    Parameters mirror :func:`compute_summary_table` +
    :func:`format_latex_table`. Returns the resolved output path.

    Raises
    ------
    FileNotFoundError
        If ``sweep_csv`` does not exist.
    """
    sweep_csv = Path(sweep_csv)
    if not sweep_csv.exists():
        raise FileNotFoundError(f"sweep_csv not found: {sweep_csv}")
    df = pd.read_csv(sweep_csv)
    summary = compute_summary_table(df, metric=metric, alpha=alpha, regimes=regimes)
    tex = format_latex_table(
        summary,
        metric_label="realised FDR" if metric == "realized_fdr" else metric,
        alpha=alpha,
        caption=caption,
        label=label,
        bold_method=bold_method,
        precision=precision,
    )
    out_tex = Path(out_tex)
    out_tex.parent.mkdir(parents=True, exist_ok=True)
    out_tex.write_text(tex)
    return out_tex
