"""F2 cluster-count figure reproducer tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from py_idr.figures.f2_cluster_count import (
    TClusterCountHistogram,
    compute_T_histograms,
    make_cluster_count_figure,
)


def _toy_df(
    regimes: tuple[str, ...] = ("S5", "S5-sparse"),
    n_replicates: int = 10,
    T_loc_by_regime: dict[str, float] | None = None,
    method: str = "PY-IDR (MCMC, T>1)",
    extra_methods: tuple[str, ...] = (),
    alphas: tuple[float, ...] = (0.05,),
    rng_seed: int = 0,
) -> pd.DataFrame:
    """Build a synthetic sweep DataFrame.

    The ``T_posterior_mean`` column is drawn from a per-regime Gaussian
    centred at ``T_loc_by_regime[regime]`` so the histogram has a
    meaningful shape.
    """
    rng = np.random.default_rng(rng_seed)
    if T_loc_by_regime is None:
        T_loc_by_regime = {r: 2.0 + i * 0.5 for i, r in enumerate(regimes)}

    rows = []
    methods = (method, *extra_methods)
    for m in methods:
        for regime in regimes:
            for seed in range(n_replicates):
                # T_posterior_mean is meaningful only for the T>1 method.
                T_pm = (
                    float(T_loc_by_regime[regime] + 0.2 * rng.standard_normal())
                    if m == "PY-IDR (MCMC, T>1)"
                    else float("nan")
                )
                for alpha in alphas:
                    rows.append(
                        {
                            "regime": regime,
                            "K": 2,
                            "n": 100,
                            "replicate_seed": seed,
                            "method": m,
                            "alpha": alpha,
                            "pi_true": 0.30,
                            "pi_posterior_mean": 0.31,
                            "rho_true": 0.85,
                            "rho_posterior_mean": float("nan"),
                            "T_posterior_mean": T_pm,
                            "k_alpha": 30,
                            "realized_fdr": 0.04,
                            "power": 0.80,
                            "num_chains": 2,
                            "num_samples_per_chain": 50,
                            "walltime_s": 1.0,
                        }
                    )
    return pd.DataFrame(rows)


# ---- compute_T_histograms: validation ------------------------------------------------


@pytest.mark.unit
def test_compute_T_histograms_rejects_missing_columns() -> None:
    """Missing one of the required columns raises a clear ValueError."""
    df = pd.DataFrame({"regime": ["S5"], "alpha": [0.05]})
    with pytest.raises(ValueError, match="missing required columns"):
        compute_T_histograms(df)


@pytest.mark.unit
def test_compute_T_histograms_rejects_zero_bin_width() -> None:
    """bin_width must be positive."""
    df = _toy_df()
    with pytest.raises(ValueError, match="bin_width"):
        compute_T_histograms(df, bin_width=0.0)


@pytest.mark.unit
def test_compute_T_histograms_method_filter_requires_method_column() -> None:
    """If method_filter is given but df has no 'method' column, raise."""
    df = pd.DataFrame(
        {
            "regime": ["S5"],
            "replicate_seed": [0],
            "T_posterior_mean": [2.0],
        }
    )
    with pytest.raises(ValueError, match="method"):
        compute_T_histograms(df, method_filter="anything")


# ---- compute_T_histograms: structural -----------------------------------------------


@pytest.mark.unit
def test_compute_T_histograms_returns_one_per_regime() -> None:
    """Two regimes in the DataFrame → two histograms in the output."""
    df = _toy_df(regimes=("S5", "S5-sparse"))
    hists = compute_T_histograms(df, regimes=("S5", "S5-sparse"))
    assert set(hists.keys()) == {"S5", "S5-sparse"}
    assert all(isinstance(h, TClusterCountHistogram) for h in hists.values())


@pytest.mark.unit
def test_compute_T_histograms_drops_nan_rows() -> None:
    """Rows with NaN T_posterior_mean are dropped silently."""
    df = _toy_df(n_replicates=5)
    # Inject some NaN rows.
    df.loc[df.index[:3], "T_posterior_mean"] = float("nan")
    hists = compute_T_histograms(df, regimes=("S5",))
    # 5 replicates × 1 alpha = 5 rows for S5; 3 dropped → 2 valid.
    assert hists["S5"].T_means.size == 2


@pytest.mark.unit
def test_compute_T_histograms_alpha_filter_deduplicates_replicates() -> None:
    """With multiple alphas in the CSV, passing alpha=X keeps each replicate once."""
    df = _toy_df(n_replicates=4, alphas=(0.01, 0.05, 0.10))
    # Without filter: 4 reps × 3 alphas = 12 rows per regime.
    hists_unfiltered = compute_T_histograms(df, regimes=("S5",), alpha=None)
    assert hists_unfiltered["S5"].T_means.size == 12
    hists_filtered = compute_T_histograms(df, regimes=("S5",), alpha=0.05)
    assert hists_filtered["S5"].T_means.size == 4


@pytest.mark.unit
def test_compute_T_histograms_method_filter_drops_other_methods() -> None:
    """method_filter keeps only rows whose method matches exactly."""
    df = _toy_df(
        n_replicates=5,
        method="PY-IDR (MCMC, T>1)",
        extra_methods=("Other-Method",),
    )
    hists = compute_T_histograms(df, regimes=("S5",), method_filter="PY-IDR (MCMC, T>1)")
    # The other-method rows have NaN T (per _toy_df) so they would be dropped
    # anyway, but the count must be exactly the T>1 contribution.
    assert hists["S5"].T_means.size == 5


@pytest.mark.unit
def test_compute_T_histograms_omits_regime_with_no_valid_rows() -> None:
    """If every row for a regime is NaN, that regime is omitted from the output."""
    df = _toy_df(regimes=("S5", "S5-sparse"))
    df.loc[df["regime"] == "S5-sparse", "T_posterior_mean"] = float("nan")
    hists = compute_T_histograms(df, regimes=("S5", "S5-sparse"))
    assert list(hists.keys()) == ["S5"]


@pytest.mark.unit
def test_compute_T_histograms_histogram_consistency() -> None:
    """Bin counts sum to the number of valid replicates; edges align with bin_width."""
    df = _toy_df(n_replicates=8)
    hists = compute_T_histograms(df, regimes=("S5",), bin_width=0.5)
    h = hists["S5"]
    assert h.bin_counts.sum() == h.T_means.size
    assert h.bin_counts.size == h.bin_edges.size - 1
    spacing = np.diff(h.bin_edges)
    assert np.allclose(spacing, 0.5)


@pytest.mark.unit
def test_compute_T_histograms_overall_mean_matches_T_means() -> None:
    """overall_mean is exactly mean(T_means)."""
    df = _toy_df(n_replicates=12)
    hists = compute_T_histograms(df, regimes=("S5",))
    h = hists["S5"]
    assert h.overall_mean == pytest.approx(float(h.T_means.mean()))


# ---- make_cluster_count_figure: end-to-end ------------------------------------------


@pytest.mark.integration
def test_make_cluster_count_figure_writes_pdf(tmp_path: Path) -> None:
    """End-to-end: CSV → PDF with non-zero size."""
    pytest.importorskip("matplotlib")
    df = _toy_df(regimes=("S5", "S5-sparse"))
    csv = tmp_path / "sweep.csv"
    df.to_csv(csv, index=False)
    pdf = make_cluster_count_figure(csv, tmp_path / "f2.pdf", regimes=("S5", "S5-sparse"))
    assert pdf.exists()
    assert pdf.stat().st_size > 1000


@pytest.mark.integration
def test_make_cluster_count_figure_raises_on_missing_csv(tmp_path: Path) -> None:
    """Missing CSV raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        make_cluster_count_figure(tmp_path / "absent.csv", tmp_path / "f2.pdf")


@pytest.mark.integration
def test_make_cluster_count_figure_raises_when_no_T_data(tmp_path: Path) -> None:
    """If every relevant row has NaN T, the user gets an explicit error."""
    pytest.importorskip("matplotlib")
    df = _toy_df(regimes=("S5",))
    df["T_posterior_mean"] = float("nan")
    csv = tmp_path / "sweep.csv"
    df.to_csv(csv, index=False)
    with pytest.raises(ValueError, match="No matching regimes"):
        make_cluster_count_figure(csv, tmp_path / "f2.pdf", regimes=("S5",))
