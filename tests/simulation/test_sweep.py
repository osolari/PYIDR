"""Sweep driver + CSV writer tests."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from py_idr.simulation.replicates import ReplicateRow
from py_idr.simulation.sweep import (
    run_sweep,
    to_dataframe,
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
