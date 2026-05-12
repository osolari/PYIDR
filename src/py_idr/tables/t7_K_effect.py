r"""T7 reproducer: effect of replicate count $K$ on power and realised FDR.

Reproduces ``tab:sim-K`` from docs/report/tables/table_sim_results.tex.
Restricted to **S5** (the heavy-tail mixture regime) — that is the
column where the report's planned hypothesis pins down "more
replicates lets the multi-cluster atom better identify the mixture".

Each cell is ``power / realised_FDR`` (both as fractions, formatted to
the requested precision). One row per method, one column per $K$.

Two-tier surface, matching T4:

- :func:`compute_K_effect_table` — pure data helper. Filters the
  sweep DataFrame to a single regime (default ``"S5"``) and a single
  $\alpha$ (default ``0.05``), groups by (method, K), returns per-cell
  mean ``power`` and mean ``realized_fdr``.
- :func:`format_latex_table` — emit a LaTeX tabular string.
- :func:`make_K_effect_table` — end-to-end CSV → ``.tex`` wrapper.

See plans/09_figures_and_tables.md.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

_PY_IDR_MCMC = "PY-IDR (MCMC, T>1)"  # T7 is about S5, which routes through T>1

# Row order matches the report's tab:sim-K. Methods we don't yet have
# implementations for are emitted as placeholder rows.
_REPORT_METHODS_ORDER: tuple[str, ...] = (
    "Vanilla IDR (pairwise)",
    "eCV",
    "nestedIDR",
    "PY-IDR (MCMC, T=1)",
    "PY-IDR (MCMC, T>1)",
    "PY-IDR (VI)",
)


def compute_K_effect_table(
    df: pd.DataFrame,
    *,
    regime: str = "S5",
    alpha: float = 0.05,
    K_values: tuple[int, ...] = (2, 5, 10, 50),
) -> pd.DataFrame:
    """Summarise mean (power, realised FDR) per (method, K) at one $\\alpha$.

    Parameters
    ----------
    df
        Long-format sweep DataFrame. Required columns: ``regime``,
        ``K``, ``method``, ``alpha``, ``replicate_seed``, ``power``,
        ``realized_fdr``.
    regime
        Regime to restrict to. Default ``"S5"`` matches the report's
        tab:sim-K.
    alpha
        Sun-Cai operating level.
    K_values
        $K$ column order. Missing values give NaN.

    Returns
    -------
    A DataFrame indexed by ``method`` with MultiIndex columns
    ``[("power", K), ("realized_fdr", K)]`` for each $K$.

    Raises
    ------
    ValueError
        If required columns are missing or no rows survive the
        regime/alpha filter.
    """
    required = {"regime", "K", "method", "alpha", "replicate_seed", "power", "realized_fdr"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"df is missing required columns: {sorted(missing)}")

    sub = df[(df["regime"] == regime) & (df["alpha"] == alpha)].copy()
    if sub.empty:
        raise ValueError(f"No rows in df with regime == {regime!r} and alpha == {alpha}")

    grouped = sub.groupby(["method", "K"])[["power", "realized_fdr"]].mean()
    pivoted = grouped.unstack("K")
    cols = pd.MultiIndex.from_product([["power", "realized_fdr"], list(K_values)])
    pivoted = pivoted.reindex(columns=cols)
    return pivoted


def format_latex_table(
    summary: pd.DataFrame,
    *,
    regime: str = "S5",
    alpha: float = 0.05,
    caption: str | None = None,
    label: str = "tab:sim-K",
    bold_methods: tuple[str, ...] = ("PY-IDR (MCMC, T>1)", "PY-IDR (VI)"),
    methods_order: tuple[str, ...] = _REPORT_METHODS_ORDER,
    placeholder: str = "---",
    precision: int = 3,
) -> str:
    r"""Emit a LaTeX ``tabular`` string for the K-effect summary.

    Cell layout: ``power / realised_FDR`` to the requested precision.
    Methods not present in ``summary`` emit a placeholder row.
    """
    K_values = list(summary.columns.get_level_values(1).unique())
    if caption is None:
        caption = (
            f"Effect of replicate count $K$ on power and realised FDR for "
            f"regime {regime} at nominal $\\alpha = {alpha}$. Each cell shows "
            f"mean power / realised FDR across replicate seeds."
        )

    n_cols = len(K_values)
    col_spec = "l" + "c" * n_cols
    lines: list[str] = []
    lines.append(r"\begin{table}[H]")
    lines.append(r"\centering")
    lines.append(rf"\caption{{{caption}}}")
    lines.append(rf"\label{{{label}}}")
    lines.append(r"\small")
    lines.append(r"\resizebox{\textwidth}{!}{%")
    lines.append(rf"\begin{{tabular}}{{{col_spec}}}")
    lines.append(r"\toprule")
    header = "Method & " + " & ".join(f"$K={k}$" for k in K_values) + r" \\"
    lines.append(header)
    lines.append(r"\midrule")

    bold_set = set(bold_methods)
    for method in methods_order:
        row_label = rf"\textbf{{{method}}}" if method in bold_set else method
        if method not in summary.index:
            cells = [placeholder] * n_cols
        else:
            cells = []
            for k in K_values:
                p = summary.loc[method, ("power", k)]
                f = summary.loc[method, ("realized_fdr", k)]
                if pd.isna(p) or pd.isna(f):
                    cells.append(placeholder)
                else:
                    cells.append(f"{p:.{precision}f} / {f:.{precision}f}")
        lines.append(f"{row_label} & " + " & ".join(cells) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}}")
    lines.append(r"\end{table}")
    return "\n".join(lines) + "\n"


def make_K_effect_table(
    sweep_csv: str | Path,
    out_tex: str | Path,
    *,
    regime: str = "S5",
    alpha: float = 0.05,
    K_values: tuple[int, ...] = (2, 5, 10, 50),
    caption: str | None = None,
    label: str = "tab:sim-K",
    precision: int = 3,
) -> Path:
    """End-to-end CSV → ``.tex`` wrapper for T7.

    Raises
    ------
    FileNotFoundError
        If ``sweep_csv`` does not exist.
    """
    sweep_csv = Path(sweep_csv)
    if not sweep_csv.exists():
        raise FileNotFoundError(f"sweep_csv not found: {sweep_csv}")
    df = pd.read_csv(sweep_csv)
    summary = compute_K_effect_table(df, regime=regime, alpha=alpha, K_values=K_values)
    tex = format_latex_table(
        summary,
        regime=regime,
        alpha=alpha,
        caption=caption,
        label=label,
        precision=precision,
    )
    out_tex = Path(out_tex)
    out_tex.parent.mkdir(parents=True, exist_ok=True)
    out_tex.write_text(tex)
    return out_tex
