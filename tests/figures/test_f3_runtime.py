"""F3 runtime figure reproducer tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from py_idr.figures.f3_runtime import (
    RuntimeCurve,
    compute_runtime_curves,
    make_runtime_figure,
)


def _toy_df(
    regimes: tuple[str, ...] = ("S1",),
    methods: tuple[str, ...] = ("PY-IDR (MCMC, T=1)",),
    n_values: tuple[int, ...] = (50, 100, 200, 400),
    n_replicates: int = 4,
    alphas: tuple[float, ...] = (0.05,),
    # walltime(n) = scale * n**slope_true + noise.
    scale: float = 0.01,
    slope_true: float = 1.0,
    rng_seed: int = 0,
) -> pd.DataFrame:
    """Build a synthetic sweep DataFrame with a known walltime scaling law."""
    rng = np.random.default_rng(rng_seed)
    rows = []
    for regime in regimes:
        for method in methods:
            for n in n_values:
                for seed in range(n_replicates):
                    walltime = float(scale * (n**slope_true) * (1.0 + 0.05 * rng.standard_normal()))
                    for alpha in alphas:
                        rows.append(
                            {
                                "regime": regime,
                                "K": 2,
                                "n": n,
                                "replicate_seed": seed,
                                "method": method,
                                "alpha": alpha,
                                "pi_true": 0.30,
                                "pi_posterior_mean": 0.32,
                                "rho_true": 0.85,
                                "rho_posterior_mean": 0.83,
                                "T_posterior_mean": float("nan"),
                                "k_alpha": 30,
                                "realized_fdr": 0.04,
                                "power": 0.80,
                                "num_chains": 2,
                                "num_samples_per_chain": 50,
                                "walltime_s": max(walltime, 1e-6),
                            }
                        )
    return pd.DataFrame(rows)


# ---- compute_runtime_curves: validation ----------------------------------------------


@pytest.mark.unit
def test_compute_runtime_curves_rejects_missing_columns() -> None:
    """Missing one of the required columns raises ValueError."""
    df = pd.DataFrame({"regime": ["S1"], "alpha": [0.05]})
    with pytest.raises(ValueError, match="missing required columns"):
        compute_runtime_curves(df)


# ---- compute_runtime_curves: structural ---------------------------------------------


@pytest.mark.unit
def test_compute_runtime_curves_one_per_regime_method() -> None:
    """One RuntimeCurve per (regime, method) pair in the DataFrame."""
    df = _toy_df(
        regimes=("S1", "S5"),
        methods=("PY-IDR (MCMC, T=1)", "PY-IDR (MCMC, T>1)"),
    )
    curves = compute_runtime_curves(df)
    assert len(curves) == 4
    for (regime, method), c in curves.items():
        assert isinstance(c, RuntimeCurve)
        assert c.regime == regime
        assert c.method == method


@pytest.mark.unit
def test_compute_runtime_curves_n_values_sorted() -> None:
    """n_values are returned in ascending order."""
    df = _toy_df(n_values=(400, 50, 200, 100))
    curves = compute_runtime_curves(df)
    n = next(iter(curves.values())).n_values
    assert list(n) == sorted(n)


@pytest.mark.unit
def test_compute_runtime_curves_recovers_known_slope() -> None:
    """With slope_true = 1.0, the fitted slope is within ±0.1."""
    df = _toy_df(n_values=(100, 200, 400, 800), n_replicates=10, slope_true=1.0)
    curves = compute_runtime_curves(df)
    c = next(iter(curves.values()))
    assert abs(c.slope - 1.0) < 0.1


@pytest.mark.unit
def test_compute_runtime_curves_recovers_quadratic_slope() -> None:
    """With slope_true = 2.0, the fitted slope is within ±0.1."""
    df = _toy_df(n_values=(100, 200, 400, 800), n_replicates=10, slope_true=2.0)
    curves = compute_runtime_curves(df)
    c = next(iter(curves.values()))
    assert abs(c.slope - 2.0) < 0.1


@pytest.mark.unit
def test_compute_runtime_curves_single_n_gives_nan_slope() -> None:
    """One n value can't fit a slope; output is NaN."""
    df = _toy_df(n_values=(100,))
    curves = compute_runtime_curves(df)
    c = next(iter(curves.values()))
    assert np.isnan(c.slope) and np.isnan(c.intercept)


@pytest.mark.unit
def test_compute_runtime_curves_alpha_dedupe() -> None:
    """alpha filter deduplicates per-replicate rows when CSV has multiple alphas."""
    df = _toy_df(n_values=(100, 200), n_replicates=3, alphas=(0.01, 0.05, 0.10))
    curves = compute_runtime_curves(df, alpha=0.05)
    c = next(iter(curves.values()))
    # Mean walltime at n=100 should be the mean over 3 replicates at alpha=0.05,
    # which is the same as the per-replicate value once dedupe happens.
    assert c.walltime_mean_s.size == 2


@pytest.mark.unit
def test_compute_runtime_curves_regimes_and_methods_filter() -> None:
    """Allow-lists drop everything not in them."""
    df = _toy_df(
        regimes=("S1", "S5"),
        methods=("PY-IDR (MCMC, T=1)", "PY-IDR (MCMC, T>1)"),
    )
    curves = compute_runtime_curves(df, regimes=("S5",), methods=("PY-IDR (MCMC, T>1)",))
    assert set(curves.keys()) == {("S5", "PY-IDR (MCMC, T>1)")}


# ---- make_runtime_figure: end-to-end ------------------------------------------------


@pytest.mark.integration
def test_make_runtime_figure_writes_pdf(tmp_path: Path) -> None:
    """End-to-end: CSV → PDF with non-zero size."""
    pytest.importorskip("matplotlib")
    df = _toy_df(n_values=(50, 100, 200, 400))
    csv = tmp_path / "sweep.csv"
    df.to_csv(csv, index=False)
    pdf = make_runtime_figure(csv, tmp_path / "f3.pdf")
    assert pdf.exists()
    assert pdf.stat().st_size > 1000


@pytest.mark.integration
def test_make_runtime_figure_raises_on_missing_csv(tmp_path: Path) -> None:
    """Missing CSV raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        make_runtime_figure(tmp_path / "absent.csv", tmp_path / "f3.pdf")


@pytest.mark.integration
def test_make_runtime_figure_raises_when_filters_empty(tmp_path: Path) -> None:
    """If the filters leave no rows, the user gets an explicit error."""
    pytest.importorskip("matplotlib")
    df = _toy_df(regimes=("S1",))
    csv = tmp_path / "sweep.csv"
    df.to_csv(csv, index=False)
    with pytest.raises(ValueError, match="No matching rows"):
        make_runtime_figure(csv, tmp_path / "f3.pdf", regimes=("S99",))
