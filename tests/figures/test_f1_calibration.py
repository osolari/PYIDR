"""F1 calibration figure reproducer tests.

The data helper :func:`compute_calibration_curves` is tested with hand-
crafted DataFrames so no chain runs are needed. The plotting wrapper
:func:`make_calibration_figure` is exercised in one integration test
that writes a tiny PDF to a tmp path.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from py_idr.figures.f1_calibration import (
    CalibrationCurve,
    compute_calibration_curves,
    make_calibration_figure,
)


def _toy_df(
    regimes: tuple[str, ...] = ("S1", "S2"),
    alphas: tuple[float, ...] = (0.01, 0.05, 0.10),
    n_replicates: int = 5,
    method: str = "PY-IDR (MCMC, T=1)",
    rng_seed: int = 0,
) -> pd.DataFrame:
    """Build a synthetic sweep DataFrame with structure-only realised FDR."""
    rng = np.random.default_rng(rng_seed)
    rows = []
    for regime in regimes:
        for seed in range(n_replicates):
            for alpha in alphas:
                # Realised FDR is alpha plus a small zero-mean noise.
                fdr = float(alpha + 0.005 * rng.standard_normal())
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
                        "realized_fdr": max(0.0, min(1.0, fdr)),
                        "power": 0.80,
                        "num_chains": 2,
                        "num_samples_per_chain": 50,
                    }
                )
    return pd.DataFrame(rows)


# ---- compute_calibration_curves: validation ------------------------------------------


@pytest.mark.unit
def test_compute_calibration_curves_rejects_missing_columns() -> None:
    """Missing one of the required columns triggers a ValueError listing the missing names."""
    df = pd.DataFrame({"regime": ["S1"], "alpha": [0.05]})
    with pytest.raises(ValueError, match="missing required columns"):
        compute_calibration_curves(df)


@pytest.mark.unit
def test_compute_calibration_curves_rejects_invalid_bootstrap_b() -> None:
    """bootstrap_b must be positive."""
    df = _toy_df()
    with pytest.raises(ValueError, match="bootstrap_b"):
        compute_calibration_curves(df, bootstrap_b=0)


@pytest.mark.unit
def test_compute_calibration_curves_rejects_invalid_ci_level() -> None:
    """ci_level must be in (0, 1)."""
    df = _toy_df()
    with pytest.raises(ValueError, match="ci_level"):
        compute_calibration_curves(df, ci_level=0.0)


# ---- compute_calibration_curves: structural -----------------------------------------


@pytest.mark.unit
def test_compute_calibration_curves_returns_one_curve_per_regime() -> None:
    """Two regimes in the DataFrame → two curves in the output."""
    df = _toy_df(regimes=("S1", "S2"))
    curves = compute_calibration_curves(df, regimes=("S1", "S2"))
    assert set(curves.keys()) == {"S1", "S2"}
    assert all(isinstance(c, CalibrationCurve) for c in curves.values())


@pytest.mark.unit
def test_compute_calibration_curves_silently_drops_missing_regimes() -> None:
    """Regimes not present in the DataFrame are silently omitted (no error)."""
    df = _toy_df(regimes=("S1",))
    curves = compute_calibration_curves(df, regimes=("S1", "S2", "S5"))
    assert list(curves.keys()) == ["S1"]


@pytest.mark.unit
def test_compute_calibration_curves_alpha_grid_is_sorted() -> None:
    """Per-regime alpha grid is returned in ascending order."""
    df = _toy_df(alphas=(0.20, 0.01, 0.10))
    curves = compute_calibration_curves(df, regimes=("S1",))
    assert list(curves["S1"].alphas) == sorted(curves["S1"].alphas)


@pytest.mark.unit
def test_compute_calibration_curves_mean_tracks_alpha() -> None:
    """For a well-calibrated synthetic DataFrame, mean realised FDR ≈ nominal alpha."""
    df = _toy_df(n_replicates=20, alphas=(0.01, 0.05, 0.10, 0.20), rng_seed=7)
    curves = compute_calibration_curves(df, regimes=("S1",))
    curve = curves["S1"]
    for a, m in zip(curve.alphas, curve.realized_fdr_mean, strict=True):
        # Synthetic data has mean fdr ≈ alpha; 5% tolerance handles MC noise.
        assert abs(m - a) < 0.01


@pytest.mark.unit
def test_compute_calibration_curves_ci_band_contains_mean() -> None:
    """The bootstrap-percentile band brackets the point estimate."""
    df = _toy_df(n_replicates=10)
    curves = compute_calibration_curves(df, regimes=("S1",))
    curve = curves["S1"]
    for m, lo, hi in zip(
        curve.realized_fdr_mean,
        curve.realized_fdr_lo,
        curve.realized_fdr_hi,
        strict=True,
    ):
        assert lo <= m <= hi


@pytest.mark.unit
def test_compute_calibration_curves_ci_nan_when_single_replicate() -> None:
    """With one replicate per alpha, the bootstrap band is NaN (no variability)."""
    df = _toy_df(n_replicates=1)
    curves = compute_calibration_curves(df, regimes=("S1",))
    curve = curves["S1"]
    assert np.all(np.isnan(curve.realized_fdr_lo))
    assert np.all(np.isnan(curve.realized_fdr_hi))


@pytest.mark.unit
def test_compute_calibration_curves_method_filter_keeps_only_match() -> None:
    """method_filter drops rows whose method does not match exactly."""
    df1 = _toy_df(method="PY-IDR (MCMC, T=1)")
    df2 = _toy_df(method="competitor-X", rng_seed=99)
    df = pd.concat([df1, df2], ignore_index=True)
    curves_all = compute_calibration_curves(df, regimes=("S1",))
    curves_pyidr = compute_calibration_curves(
        df, regimes=("S1",), method_filter="PY-IDR (MCMC, T=1)"
    )
    # n_replicates should halve when we filter to one of two methods.
    assert curves_pyidr["S1"].n_replicates[0] == curves_all["S1"].n_replicates[0] // 2


@pytest.mark.unit
def test_compute_calibration_curves_bootstrap_is_seeded() -> None:
    """Same rng_seed reproduces the bootstrap CI bands exactly."""
    df = _toy_df(n_replicates=5)
    c1 = compute_calibration_curves(df, regimes=("S1",), rng_seed=42)["S1"]
    c2 = compute_calibration_curves(df, regimes=("S1",), rng_seed=42)["S1"]
    assert np.array_equal(c1.realized_fdr_lo, c2.realized_fdr_lo)
    assert np.array_equal(c1.realized_fdr_hi, c2.realized_fdr_hi)


# ---- make_calibration_figure: end-to-end --------------------------------------------


@pytest.mark.integration
def test_make_calibration_figure_writes_pdf(tmp_path: Path) -> None:
    """The wrapper reads a CSV, renders the figure, and writes a non-empty PDF."""
    pytest.importorskip("matplotlib")
    df = _toy_df(regimes=("S1", "S2"))
    csv = tmp_path / "sweep.csv"
    df.to_csv(csv, index=False)
    pdf = make_calibration_figure(
        csv,
        tmp_path / "figs" / "f1.pdf",
        regimes=("S1", "S2"),
    )
    assert pdf.exists()
    assert pdf.stat().st_size > 1000  # PDFs are at least a kilobyte


@pytest.mark.integration
def test_make_calibration_figure_creates_parent_dir(tmp_path: Path) -> None:
    """Output PDF's parent dir is created if absent."""
    pytest.importorskip("matplotlib")
    df = _toy_df(regimes=("S1",))
    csv = tmp_path / "sweep.csv"
    df.to_csv(csv, index=False)
    pdf = make_calibration_figure(csv, tmp_path / "a" / "b" / "c" / "f1.pdf", regimes=("S1",))
    assert pdf.exists()


@pytest.mark.integration
def test_make_calibration_figure_raises_on_missing_csv(tmp_path: Path) -> None:
    """Missing CSV path raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        make_calibration_figure(tmp_path / "absent.csv", tmp_path / "f1.pdf")


@pytest.mark.integration
def test_make_calibration_figure_raises_when_no_regime_matches(tmp_path: Path) -> None:
    """Asking for regimes that aren't in the CSV is an explicit error, not a silent empty figure."""
    pytest.importorskip("matplotlib")
    df = _toy_df(regimes=("S1",))
    csv = tmp_path / "sweep.csv"
    df.to_csv(csv, index=False)
    with pytest.raises(ValueError, match="No matching regimes"):
        make_calibration_figure(csv, tmp_path / "f1.pdf", regimes=("S99",))
