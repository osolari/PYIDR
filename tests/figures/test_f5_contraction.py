"""F5 contraction figure reproducer tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from py_idr.figures.f5_contraction import (
    ContractionCurve,
    compute_contraction_curves,
    make_contraction_figure,
)


def _toy_df(
    regimes: tuple[str, ...] = ("S1",),
    n_values: tuple[int, ...] = (50, 100, 200, 400, 800),
    n_replicates: int = 6,
    rho_true: float = 0.85,
    # rho_hat - rho_true ~ N(0, noise_scale / sqrt(n)) so contraction is n^{-1/2}.
    noise_scale: float = 0.3,
    methods: tuple[str, ...] = ("PY-IDR (MCMC, T=1)",),
    K_values: tuple[int, ...] = (2,),
    alphas: tuple[float, ...] = (0.05,),
    rng_seed: int = 0,
) -> pd.DataFrame:
    """Build a synthetic sweep DataFrame whose rho_posterior_mean exhibits ~n^{-1/2}
    contraction, so the F5 fit recovers a known slope (~-0.5).
    """
    rng = np.random.default_rng(rng_seed)
    rows = []
    for regime in regimes:
        for method in methods:
            for K in K_values:
                for n in n_values:
                    for seed in range(n_replicates):
                        rho_hat = rho_true + (noise_scale / np.sqrt(n)) * rng.standard_normal()
                        # Clip into the admissible range so the closed-form
                        # Hellinger doesn't raise.
                        rho_hat = float(np.clip(rho_hat, -0.99 / (K - 1), 0.99))
                        for alpha in alphas:
                            rows.append(
                                {
                                    "regime": regime,
                                    "K": K,
                                    "n": n,
                                    "replicate_seed": seed,
                                    "method": method,
                                    "alpha": alpha,
                                    "pi_true": 0.30,
                                    "pi_posterior_mean": 0.32,
                                    "rho_true": rho_true,
                                    "rho_posterior_mean": rho_hat,
                                    "T_posterior_mean": float("nan"),
                                    "k_alpha": 30,
                                    "realized_fdr": 0.04,
                                    "power": 0.80,
                                    "num_chains": 2,
                                    "num_samples_per_chain": 50,
                                    "walltime_s": 1.0,
                                }
                            )
    return pd.DataFrame(rows)


# ---- compute_contraction_curves: validation ------------------------------------------


@pytest.mark.unit
def test_compute_contraction_rejects_missing_columns() -> None:
    """Missing one of the required columns raises ValueError."""
    df = pd.DataFrame({"regime": ["S1"], "n": [100]})
    with pytest.raises(ValueError, match="missing required columns"):
        compute_contraction_curves(df)


@pytest.mark.unit
def test_compute_contraction_rejects_when_filters_empty() -> None:
    """If every row is filtered out, raise."""
    df = _toy_df()
    with pytest.raises(ValueError, match="No rows survived"):
        compute_contraction_curves(df, alpha=0.99)


# ---- compute_contraction_curves: structural -----------------------------------------


@pytest.mark.unit
def test_compute_contraction_returns_one_curve_per_regime() -> None:
    """Two regimes → two curves."""
    df = _toy_df(regimes=("S1", "S2"))
    curves = compute_contraction_curves(df, regimes=("S1", "S2"))
    assert set(curves.keys()) == {"S1", "S2"}
    assert all(isinstance(c, ContractionCurve) for c in curves.values())


@pytest.mark.unit
def test_compute_contraction_n_sorted() -> None:
    """n_values within a curve are sorted ascending."""
    df = _toy_df(n_values=(400, 50, 200, 100))
    curves = compute_contraction_curves(df, regimes=("S1",))
    n = curves["S1"].n_values
    assert list(n) == sorted(n)


@pytest.mark.unit
def test_compute_contraction_zero_at_perfect_recovery() -> None:
    """rho_posterior_mean == rho_true => Hellinger == 0."""
    # Build a df where rho_hat = rho_true at every row.
    df = _toy_df(noise_scale=0.0, n_values=(100, 200), n_replicates=3)
    curves = compute_contraction_curves(df, regimes=("S1",))
    assert np.allclose(curves["S1"].hellinger_mean, 0.0)


@pytest.mark.unit
def test_compute_contraction_decreasing_with_n() -> None:
    """At noise_scale > 0, Hellinger should decrease (in expectation) with n."""
    df = _toy_df(
        regimes=("S1",),
        n_values=(50, 100, 200, 400, 800),
        n_replicates=20,
        noise_scale=0.3,
        rng_seed=42,
    )
    curves = compute_contraction_curves(df, regimes=("S1",))
    h = curves["S1"].hellinger_mean
    # At least the smallest-n Hellinger should exceed the largest-n one.
    assert h[0] > h[-1]


@pytest.mark.unit
def test_compute_contraction_recovers_half_slope() -> None:
    """Synthetic n^{-1/2} noise yields a fitted slope close to -0.5 in expectation."""
    df = _toy_df(
        regimes=("S1",),
        n_values=(50, 100, 200, 400, 800, 1600),
        n_replicates=30,
        noise_scale=0.3,
        rng_seed=0,
    )
    curves = compute_contraction_curves(df, regimes=("S1",))
    slope = curves["S1"].slope
    # Loose tolerance — MC noise + small replicate count.
    assert slope == pytest.approx(-0.5, abs=0.2)


@pytest.mark.unit
def test_compute_contraction_skips_nan_rho_true() -> None:
    """Rows with NaN rho_true (S5) are dropped silently."""
    df = _toy_df(regimes=("S1",))
    df_s5 = df.copy()
    df_s5["regime"] = "S5"
    df_s5["rho_true"] = float("nan")
    combined = pd.concat([df, df_s5], ignore_index=True)
    curves = compute_contraction_curves(combined, regimes=("S1", "S5"))
    assert list(curves.keys()) == ["S1"]


@pytest.mark.unit
def test_compute_contraction_K_filter() -> None:
    """K filter narrows to one K's contribution."""
    df = _toy_df(K_values=(2, 3))
    curves = compute_contraction_curves(df, regimes=("S1",), K=2)
    # K=2 only -> half the rows -> half the replicate counts.
    full = compute_contraction_curves(df, regimes=("S1",), K=None)
    assert curves["S1"].n_replicates[0] == full["S1"].n_replicates[0] // 2


# ---- make_contraction_figure: end-to-end --------------------------------------------


@pytest.mark.integration
def test_make_contraction_figure_writes_pdf(tmp_path: Path) -> None:
    """End-to-end: CSV -> PDF with non-zero size."""
    pytest.importorskip("matplotlib")
    df = _toy_df()
    csv = tmp_path / "sweep.csv"
    df.to_csv(csv, index=False)
    pdf = make_contraction_figure(csv, tmp_path / "f5.pdf")
    assert pdf.exists()
    assert pdf.stat().st_size > 1000


@pytest.mark.integration
def test_make_contraction_figure_raises_on_missing_csv(tmp_path: Path) -> None:
    """Missing CSV raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        make_contraction_figure(tmp_path / "absent.csv", tmp_path / "f5.pdf")


@pytest.mark.integration
def test_make_contraction_figure_raises_when_filter_empty(tmp_path: Path) -> None:
    """If filters drop everything, the figure helper raises with a clear message."""
    pytest.importorskip("matplotlib")
    df = _toy_df()
    csv = tmp_path / "sweep.csv"
    df.to_csv(csv, index=False)
    with pytest.raises(ValueError, match="No regimes survived"):
        make_contraction_figure(csv, tmp_path / "f5.pdf", regimes=("S99",))
