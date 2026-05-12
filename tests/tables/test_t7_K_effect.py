"""T7 K-effect LaTeX table reproducer tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from py_idr.tables.t7_K_effect import (
    _PY_IDR_MCMC,
    compute_K_effect_table,
    format_latex_table,
    make_K_effect_table,
)


def _toy_df(
    regimes: tuple[str, ...] = ("S5",),
    K_values: tuple[int, ...] = (2, 5, 10),
    n_replicates: int = 5,
    method: str = _PY_IDR_MCMC,
    alphas: tuple[float, ...] = (0.05,),
    rng_seed: int = 0,
) -> pd.DataFrame:
    """Build a synthetic sweep DataFrame whose power grows with K (synthetic law)."""
    rng = np.random.default_rng(rng_seed)
    rows = []
    for regime in regimes:
        for K in K_values:
            for seed in range(n_replicates):
                # Power tied to K so the table cells are non-trivial.
                power_mean = min(0.95, 0.20 + 0.05 * K + 0.02 * rng.standard_normal())
                fdr_mean = max(0.0, 0.04 + 0.01 * rng.standard_normal())
                for alpha in alphas:
                    rows.append(
                        {
                            "regime": regime,
                            "K": K,
                            "n": 100,
                            "replicate_seed": seed,
                            "method": method,
                            "alpha": alpha,
                            "pi_true": 0.30,
                            "pi_posterior_mean": 0.32,
                            "rho_true": float("nan"),
                            "rho_posterior_mean": float("nan"),
                            "T_posterior_mean": 2.5,
                            "k_alpha": 30,
                            "realized_fdr": fdr_mean,
                            "power": power_mean,
                            "num_chains": 2,
                            "num_samples_per_chain": 50,
                            "walltime_s": 1.0,
                        }
                    )
    return pd.DataFrame(rows)


# ---- compute_K_effect_table: validation ----------------------------------------------


@pytest.mark.unit
def test_compute_K_effect_rejects_missing_columns() -> None:
    """Missing one of the required columns raises ValueError."""
    df = pd.DataFrame({"regime": ["S5"], "K": [5]})
    with pytest.raises(ValueError, match="missing required columns"):
        compute_K_effect_table(df)


@pytest.mark.unit
def test_compute_K_effect_rejects_empty_after_filter() -> None:
    """Filtering to a non-matching regime/alpha raises."""
    df = _toy_df()
    with pytest.raises(ValueError, match="No rows"):
        compute_K_effect_table(df, regime="S99")


# ---- compute_K_effect_table: structural ----------------------------------------------


@pytest.mark.unit
def test_compute_K_effect_shape_and_columns() -> None:
    """Output has MultiIndex columns (power | realized_fdr) × K."""
    df = _toy_df(K_values=(2, 5, 10))
    summary = compute_K_effect_table(df, K_values=(2, 5, 10))
    assert isinstance(summary.columns, pd.MultiIndex)
    assert set(summary.columns.get_level_values(0)) == {"power", "realized_fdr"}
    assert list(summary.columns.get_level_values(1).unique()) == [2, 5, 10]
    assert _PY_IDR_MCMC in summary.index


@pytest.mark.unit
def test_compute_K_effect_power_increases_with_K() -> None:
    """Synthetic power-grows-with-K law shows up in the summary."""
    df = _toy_df(K_values=(2, 5, 10), n_replicates=20, rng_seed=0)
    summary = compute_K_effect_table(df, K_values=(2, 5, 10))
    p2 = summary.loc[_PY_IDR_MCMC, ("power", 2)]
    p10 = summary.loc[_PY_IDR_MCMC, ("power", 10)]
    assert p10 > p2


@pytest.mark.unit
def test_compute_K_effect_missing_K_is_nan() -> None:
    """K values not present in the DataFrame yield NaN."""
    df = _toy_df(K_values=(2, 5))
    summary = compute_K_effect_table(df, K_values=(2, 5, 50))
    assert np.isnan(summary.loc[_PY_IDR_MCMC, ("power", 50)])


# ---- format_latex_table -------------------------------------------------------------


@pytest.mark.unit
def test_format_latex_emits_valid_skeleton() -> None:
    """LaTeX skeleton is well-formed."""
    df = _toy_df()
    summary = compute_K_effect_table(df)
    tex = format_latex_table(summary)
    assert tex.startswith("\\begin{table}")
    assert tex.rstrip().endswith("\\end{table}")
    assert "tab:sim-K" in tex


@pytest.mark.unit
def test_format_latex_bolds_target_method() -> None:
    """bold_methods entries get \\textbf{} wrapping."""
    df = _toy_df()
    summary = compute_K_effect_table(df)
    tex = format_latex_table(summary, bold_methods=(_PY_IDR_MCMC,))
    assert f"\\textbf{{{_PY_IDR_MCMC}}}" in tex


@pytest.mark.unit
def test_format_latex_emits_power_slash_fdr_cells() -> None:
    """Cell content uses 'power / realised_FDR' layout."""
    df = _toy_df()
    summary = compute_K_effect_table(df)
    tex = format_latex_table(summary, precision=3)
    import re

    # Look for a cell that matches NNN / NNN with 3 decimals each.
    matches = re.findall(r"0\.\d{3}\s*/\s*0\.\d{3}", tex)
    assert matches


@pytest.mark.unit
def test_format_latex_placeholder_for_absent_methods() -> None:
    """Methods absent from the summary emit a placeholder row."""
    df = _toy_df()
    summary = compute_K_effect_table(df)
    tex = format_latex_table(summary, placeholder="—")
    assert "Vanilla IDR (pairwise)" in tex  # listed in the report's row order
    assert "—" in tex


# ---- make_K_effect_table: end-to-end -----------------------------------------------


@pytest.mark.unit
def test_make_K_effect_table_round_trip(tmp_path: Path) -> None:
    """CSV → .tex round-trip."""
    df = _toy_df(K_values=(2, 5))
    csv = tmp_path / "sweep.csv"
    df.to_csv(csv, index=False)
    out = make_K_effect_table(csv, tmp_path / "t7.tex", K_values=(2, 5))
    assert out.exists()
    content = out.read_text()
    assert "tab:sim-K" in content
    assert "$K=2$" in content


@pytest.mark.unit
def test_make_K_effect_table_raises_on_missing_csv(tmp_path: Path) -> None:
    """Missing CSV raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        make_K_effect_table(tmp_path / "absent.csv", tmp_path / "t7.tex")


@pytest.mark.unit
def test_make_K_effect_table_creates_parent_dir(tmp_path: Path) -> None:
    """Parent directory is created if missing."""
    df = _toy_df()
    csv = tmp_path / "sweep.csv"
    df.to_csv(csv, index=False)
    out = make_K_effect_table(csv, tmp_path / "a" / "b" / "t7.tex")
    assert out.exists()
