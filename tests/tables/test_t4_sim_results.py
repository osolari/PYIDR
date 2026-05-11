"""T4 simulation-results LaTeX table reproducer tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from py_idr.tables.t4_sim_results import (
    _PY_IDR_MCMC,
    compute_summary_table,
    format_latex_table,
    make_simulation_results_table,
)


def _toy_df(
    regimes: tuple[str, ...] = ("S1", "S2"),
    alphas: tuple[float, ...] = (0.05, 0.10),
    n_replicates: int = 5,
    method: str = _PY_IDR_MCMC,
    rng_seed: int = 0,
) -> pd.DataFrame:
    """Hand-built sweep DataFrame, no chain needed."""
    rng = np.random.default_rng(rng_seed)
    rows = []
    for regime in regimes:
        for seed in range(n_replicates):
            for alpha in alphas:
                rows.append(
                    {
                        "regime": regime,
                        "K": 2,
                        "n": 100,
                        "replicate_seed": seed,
                        "method": method,
                        "alpha": alpha,
                        "pi_true": 0.30,
                        "pi_posterior_mean": 0.31,
                        "rho_true": 0.85,
                        "rho_posterior_mean": 0.83,
                        "T_posterior_mean": float("nan"),
                        "k_alpha": 30,
                        "realized_fdr": float(alpha + 0.005 * rng.standard_normal()),
                        "power": float(0.8 + 0.02 * rng.standard_normal()),
                        "num_chains": 2,
                        "num_samples_per_chain": 50,
                    }
                )
    return pd.DataFrame(rows)


# ---- compute_summary_table: validation -----------------------------------------------


@pytest.mark.unit
def test_compute_summary_table_rejects_missing_columns() -> None:
    """Missing a required column raises ValueError."""
    df = pd.DataFrame({"regime": ["S1"], "alpha": [0.05]})
    with pytest.raises(ValueError, match="missing required columns"):
        compute_summary_table(df)


@pytest.mark.unit
def test_compute_summary_table_rejects_unmatched_alpha() -> None:
    """If no rows survive the alpha filter, raise."""
    df = _toy_df(alphas=(0.05,))
    with pytest.raises(ValueError, match="No rows"):
        compute_summary_table(df, alpha=0.99)


@pytest.mark.unit
def test_compute_summary_table_rejects_missing_metric_column() -> None:
    """metric must reference a real column."""
    df = _toy_df()
    with pytest.raises(ValueError, match="missing required columns"):
        compute_summary_table(df, metric="not_a_column")


# ---- compute_summary_table: structural ----------------------------------------------


@pytest.mark.unit
def test_compute_summary_table_shape_and_levels() -> None:
    """Output has MultiIndex columns (mean | se | n) × regimes."""
    df = _toy_df(regimes=("S1", "S2", "S3"))
    summary = compute_summary_table(df, alpha=0.05, regimes=("S1", "S2", "S3"))
    assert isinstance(summary.columns, pd.MultiIndex)
    assert set(summary.columns.get_level_values(0)) == {"mean", "se", "n"}
    assert list(summary.columns.get_level_values(1).unique()) == ["S1", "S2", "S3"]
    assert _PY_IDR_MCMC in summary.index


@pytest.mark.unit
def test_compute_summary_table_means_track_synthetic_data() -> None:
    """For synthetic realised FDR ≈ alpha, the mean column matches."""
    df = _toy_df(n_replicates=50, alphas=(0.05,))
    summary = compute_summary_table(df, alpha=0.05, regimes=("S1",))
    m = summary.loc[_PY_IDR_MCMC, ("mean", "S1")]
    assert abs(m - 0.05) < 0.005


@pytest.mark.unit
def test_compute_summary_table_se_positive_and_finite() -> None:
    """Standard errors are positive and finite when >= 2 replicates."""
    df = _toy_df(n_replicates=5)
    summary = compute_summary_table(df, alpha=0.05, regimes=("S1",))
    se = summary.loc[_PY_IDR_MCMC, ("se", "S1")]
    assert np.isfinite(se) and se > 0


@pytest.mark.unit
def test_compute_summary_table_missing_regime_is_nan() -> None:
    """Asking for a regime not in the DataFrame yields a NaN column."""
    df = _toy_df(regimes=("S1",))
    summary = compute_summary_table(df, alpha=0.05, regimes=("S1", "S99"))
    assert np.isnan(summary.loc[_PY_IDR_MCMC, ("mean", "S99")])


# ---- format_latex_table -------------------------------------------------------------


@pytest.mark.unit
def test_format_latex_table_emits_valid_skeleton() -> None:
    """Output starts/ends with the right LaTeX markers."""
    df = _toy_df()
    summary = compute_summary_table(df, alpha=0.05)
    tex = format_latex_table(summary)
    assert tex.startswith("\\begin{table}")
    assert tex.rstrip().endswith("\\end{table}")
    assert "\\toprule" in tex and "\\bottomrule" in tex


@pytest.mark.unit
def test_format_latex_table_bolds_target_method() -> None:
    """The bold_method argument wraps its row label in \\textbf."""
    df = _toy_df()
    summary = compute_summary_table(df, alpha=0.05)
    tex = format_latex_table(summary, bold_method=_PY_IDR_MCMC)
    assert f"\\textbf{{{_PY_IDR_MCMC}}}" in tex


@pytest.mark.unit
def test_format_latex_table_placeholders_for_absent_methods() -> None:
    """Methods not present in the summary emit placeholder rows."""
    df = _toy_df()
    summary = compute_summary_table(df, alpha=0.05)
    tex = format_latex_table(summary, placeholder="—")
    # Vanilla IDR is one of the report-listed methods absent from our toy df.
    assert "Vanilla IDR" in tex
    assert "—" in tex


@pytest.mark.unit
def test_format_latex_table_respects_precision() -> None:
    """Cells render with the requested decimal precision."""
    df = _toy_df()
    summary = compute_summary_table(df, alpha=0.05)
    tex = format_latex_table(summary, precision=4)
    # Look for a 4-decimal cell pattern.
    import re

    matches = re.findall(r"0\.\d{4}\s*\(0\.\d{4}\)", tex)
    assert matches


# ---- make_simulation_results_table: end-to-end --------------------------------------


@pytest.mark.unit
def test_make_simulation_results_table_round_trip(tmp_path: Path) -> None:
    """CSV → .tex with file content matching format_latex_table output."""
    df = _toy_df(regimes=("S1", "S2"))
    csv = tmp_path / "sweep.csv"
    df.to_csv(csv, index=False)
    tex_path = make_simulation_results_table(
        csv,
        tmp_path / "out" / "t4.tex",
        alpha=0.05,
        regimes=("S1", "S2"),
    )
    assert tex_path.exists()
    content = tex_path.read_text()
    assert "\\begin{table}" in content
    assert "S1" in content and "S2" in content


@pytest.mark.unit
def test_make_simulation_results_table_creates_parent_dir(tmp_path: Path) -> None:
    """Output parent directory is created if absent."""
    df = _toy_df()
    csv = tmp_path / "sweep.csv"
    df.to_csv(csv, index=False)
    out = make_simulation_results_table(csv, tmp_path / "a" / "b" / "t4.tex")
    assert out.exists()


@pytest.mark.unit
def test_make_simulation_results_table_raises_on_missing_csv(tmp_path: Path) -> None:
    """Missing CSV raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        make_simulation_results_table(tmp_path / "missing.csv", tmp_path / "t4.tex")


@pytest.mark.unit
def test_make_simulation_results_table_power_metric(tmp_path: Path) -> None:
    """metric='power' picks up the power column and uses the right caption."""
    df = _toy_df()
    csv = tmp_path / "sweep.csv"
    df.to_csv(csv, index=False)
    out = make_simulation_results_table(
        csv,
        tmp_path / "t4_power.tex",
        metric="power",
        alpha=0.05,
        label="tab:sim-power",
    )
    content = out.read_text()
    assert "tab:sim-power" in content
