"""Sweep driver + CSV writer + NPZ sidecar tests."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from py_idr.simulation.replicates import ReplicateRow
from py_idr.simulation.sweep import (
    IdrTrace,
    load_idr_sidecar,
    run_sweep,
    run_sweep_with_traces,
    to_dataframe,
    write_idr_sidecar,
    write_sweep_csv,
)

# ---- input validation -----------------------------------------------------------------


@pytest.mark.unit
def test_run_sweep_rejects_unknown_regime() -> None:
    """Unknown regime label raises ValueError."""
    with pytest.raises(ValueError, match="unknown regime"):
        run_sweep("S99", K=2, n=20, num_replicates=1)


@pytest.mark.unit
def test_run_sweep_rejects_zero_replicates() -> None:
    """num_replicates must be positive."""
    with pytest.raises(ValueError, match="num_replicates"):
        run_sweep("S1", K=2, n=20, num_replicates=0)


@pytest.mark.unit
def test_run_sweep_rejects_empty_alphas() -> None:
    """alphas must be a non-empty sequence."""
    with pytest.raises(ValueError, match="alphas"):
        run_sweep("S1", K=2, n=20, num_replicates=1, alphas=())


# ---- to_dataframe + write_sweep_csv (no chain needed) --------------------------------


def _toy_row(regime: str = "S1", seed: int = 0, alpha: float = 0.05) -> ReplicateRow:
    """Hand-built ReplicateRow for testing IO without paying chain cost."""
    import math

    return ReplicateRow(
        regime=regime,
        K=2,
        n=20,
        replicate_seed=seed,
        method="PY-IDR (MCMC, T=1)",
        alpha=alpha,
        pi_true=0.30,
        pi_posterior_mean=0.32,
        rho_true=0.85,
        rho_posterior_mean=0.83,
        T_posterior_mean=math.nan,
        k_alpha=5,
        realized_fdr=0.04,
        power=0.80,
        num_chains=2,
        num_samples_per_chain=100,
        walltime_s=2.5,
    )


@pytest.mark.unit
def test_to_dataframe_empty_returns_correct_columns() -> None:
    """An empty rows list yields a DataFrame with the ReplicateRow column set."""
    df = to_dataframe([])
    expected = set(ReplicateRow.__dataclass_fields__.keys())
    assert set(df.columns) == expected
    assert len(df) == 0


@pytest.mark.unit
def test_to_dataframe_round_trips_rows() -> None:
    """Three rows in → three rows out, with the expected column names."""
    rows = [_toy_row(seed=i) for i in range(3)]
    df = to_dataframe(rows)
    assert len(df) == 3
    assert set(df.columns) == set(ReplicateRow.__dataclass_fields__.keys())
    assert list(df["replicate_seed"]) == [0, 1, 2]


@pytest.mark.unit
def test_write_sweep_csv_round_trips(tmp_path: Path) -> None:
    """write_sweep_csv emits a CSV that re-reads to the same DataFrame."""
    rows = [_toy_row(seed=i, alpha=a) for i in range(2) for a in (0.05, 0.10)]
    out = write_sweep_csv(rows, tmp_path / "results.csv")
    assert out.exists()
    df_read = pd.read_csv(out)
    df_expected = to_dataframe(rows)
    pd.testing.assert_frame_equal(df_read, df_expected)


@pytest.mark.unit
def test_write_sweep_csv_creates_parent_directory(tmp_path: Path) -> None:
    """Parent directory is created if missing."""
    out = write_sweep_csv([_toy_row()], tmp_path / "nested" / "deeper" / "results.csv")
    assert out.exists()


# ---- end-to-end small sweep ----------------------------------------------------------


@pytest.mark.integration
@pytest.mark.slow
def test_run_sweep_S1_emits_alpha_x_replicate_rows() -> None:
    """A tiny S1 sweep with 2 replicates × 3 alphas emits 6 rows."""
    rows = run_sweep(
        "S1",
        K=2,
        n=40,
        num_replicates=2,
        base_seed=0,
        alphas=(0.01, 0.05, 0.10),
        num_chains=2,
        n_warmup=10,
        n_samples=15,
        nuts_warmup=10,
        M=5,
    )
    assert len(rows) == 6  # 2 replicates × 3 alphas
    seeds = sorted({r.replicate_seed for r in rows})
    assert seeds == [0, 1]
    alphas = sorted({r.alpha for r in rows})
    assert alphas == [0.01, 0.05, 0.10]
    # k_alpha is monotone non-decreasing in alpha within each replicate.
    for seed in seeds:
        rows_seed = sorted([r for r in rows if r.replicate_seed == seed], key=lambda r: r.alpha)
        ks = [r.k_alpha for r in rows_seed]
        assert ks == sorted(ks)


@pytest.mark.integration
@pytest.mark.slow
def test_run_sweep_reuses_chain_across_alphas() -> None:
    """Single chain per replicate: the posterior summaries are identical across alphas."""
    rows = run_sweep(
        "S1",
        K=2,
        n=40,
        num_replicates=1,
        alphas=(0.01, 0.05, 0.10, 0.20),
        num_chains=2,
        n_warmup=10,
        n_samples=15,
        nuts_warmup=10,
        M=5,
    )
    # All four rows came from one chain ⇒ pi_posterior_mean, rho_posterior_mean
    # must match exactly across them.
    pis = {r.pi_posterior_mean for r in rows}
    rhos = {r.rho_posterior_mean for r in rows}
    assert len(pis) == 1
    assert len(rhos) == 1


# ---- NPZ sidecar (write_idr_sidecar / load_idr_sidecar) -----------------------------


def _toy_trace(regime: str, seed: int, n: int = 40, rng_seed: int = 0) -> IdrTrace:
    rng = np.random.default_rng(rng_seed)
    return IdrTrace(
        regime=regime,
        replicate_seed=seed,
        idr=rng.uniform(size=n).astype(float),
        true_Z=rng.integers(0, 2, size=n).astype(np.int8),
    )


@pytest.mark.unit
def test_write_idr_sidecar_rejects_empty(tmp_path: Path) -> None:
    """Writing an empty trace list raises with a clear message."""
    with pytest.raises(ValueError, match="empty"):
        write_idr_sidecar([], tmp_path / "idr.npz")


@pytest.mark.unit
def test_write_idr_sidecar_creates_parent_dir(tmp_path: Path) -> None:
    """Parent directory is created if missing."""
    out = write_idr_sidecar([_toy_trace("S5", 0)], tmp_path / "a" / "b" / "idr.npz")
    assert out.exists()


@pytest.mark.unit
def test_idr_sidecar_round_trip(tmp_path: Path) -> None:
    """write -> load yields the same arrays grouped by regime."""
    traces = [
        _toy_trace("S5", 0, rng_seed=0),
        _toy_trace("S5", 1, rng_seed=1),
        _toy_trace("S5-sparse", 0, rng_seed=2),
    ]
    npz = write_idr_sidecar(traces, tmp_path / "idr.npz")
    loaded = load_idr_sidecar(npz)
    assert set(loaded.keys()) == {"S5", "S5-sparse"}
    assert len(loaded["S5"]) == 2
    assert len(loaded["S5-sparse"]) == 1
    # Seeds should be sorted ascending within each regime; check by comparing
    # the first idr array against the seed=0 trace.
    np.testing.assert_array_equal(loaded["S5"][0][0], traces[0].idr)
    np.testing.assert_array_equal(loaded["S5"][1][0], traces[1].idr)


@pytest.mark.unit
def test_load_idr_sidecar_raises_on_missing_path(tmp_path: Path) -> None:
    """Reading a non-existent NPZ raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        load_idr_sidecar(tmp_path / "absent.npz")


@pytest.mark.unit
def test_load_idr_sidecar_raises_on_corrupted_file(tmp_path: Path) -> None:
    """An NPZ with an idr key but no matching Z key is rejected."""
    # Hand-craft a corrupted NPZ.
    np.savez(
        tmp_path / "broken.npz",
        **{"idr__S5__0": np.array([0.1, 0.2, 0.3])},
    )
    with pytest.raises(ValueError, match="corrupted"):
        load_idr_sidecar(tmp_path / "broken.npz")


# ---- run_sweep_with_traces (slow, runs a chain) --------------------------------------


@pytest.mark.integration
@pytest.mark.slow
def test_run_sweep_with_traces_returns_rows_and_traces() -> None:
    """The dual-output sweep produces matching rows + one trace per replicate."""
    result = run_sweep_with_traces(
        "S1",
        K=2,
        n=40,
        num_replicates=2,
        base_seed=0,
        alphas=(0.05, 0.10),
        num_chains=2,
        n_warmup=10,
        n_samples=15,
        nuts_warmup=10,
        M=5,
    )
    # 2 replicates × 2 alphas = 4 rows.
    assert len(result.rows) == 4
    # 2 replicates → 2 idr traces (one per (regime, seed), shared across alphas).
    assert len(result.idr_traces) == 2
    for tr in result.idr_traces:
        assert isinstance(tr, IdrTrace)
        assert tr.regime == "S1"
        assert tr.idr.shape == (40,)
        assert tr.true_Z.shape == (40,)
        assert np.all((tr.idr >= 0.0) & (tr.idr <= 1.0))
        assert set(np.unique(tr.true_Z)).issubset({0, 1})


@pytest.mark.unit
def test_run_sweep_with_traces_rejects_unknown_regime() -> None:
    """Same validation as run_sweep."""
    with pytest.raises(ValueError, match="unknown regime"):
        run_sweep_with_traces("S99", K=2, n=20, num_replicates=1)


def _unused() -> None:  # keep import-only usage from being flagged
    _ = math.nan
