"""Tests for the ``py-idr evaluate`` CLI command."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from typer.testing import CliRunner

from py_idr.cli import app
from py_idr.simulation.sweep import IdrTrace, write_idr_sidecar

runner = CliRunner()


def _synthetic_npz(tmp_path: Path, regimes: tuple[str, ...] = ("S1", "S2")) -> Path:
    """Build a small synthetic idr_traces.npz fixture."""
    rng = np.random.default_rng(0)
    traces = []
    for regime in regimes:
        for seed in range(3):
            n_pos, n_neg = 30, 70
            idr = np.concatenate([rng.uniform(0, 0.1, n_pos), rng.uniform(0.6, 1.0, n_neg)])
            Z = np.concatenate([np.ones(n_pos, np.int8), np.zeros(n_neg, np.int8)])
            perm = rng.permutation(100)
            traces.append(
                IdrTrace(
                    regime=regime,
                    replicate_seed=seed,
                    idr=idr[perm],
                    true_Z=Z[perm],
                )
            )
    return write_idr_sidecar(traces, tmp_path / "idr.npz")


@pytest.mark.unit
def test_evaluate_runs_and_prints_summary(tmp_path: Path) -> None:
    """Happy path — produces a non-empty summary on stdout."""
    npz = _synthetic_npz(tmp_path)
    result = runner.invoke(
        app,
        ["evaluate", "--idr-npz", str(npz), "--alpha", "0.05"],
    )
    assert result.exit_code == 0, result.output
    assert "alpha = 0.05" in result.output
    assert "S1" in result.output
    assert "S2" in result.output


@pytest.mark.unit
def test_evaluate_writes_csv_when_out_supplied(tmp_path: Path) -> None:
    """`--out` writes a CSV with one row per (regime, replicate)."""
    npz = _synthetic_npz(tmp_path)
    csv_path = tmp_path / "out" / "eval.csv"
    result = runner.invoke(
        app,
        [
            "evaluate",
            "--idr-npz",
            str(npz),
            "--alpha",
            "0.10",
            "--out",
            str(csv_path),
        ],
    )
    assert result.exit_code == 0, result.output
    assert csv_path.exists()
    df = pd.read_csv(csv_path)
    # 2 regimes × 3 replicates = 6 rows.
    assert len(df) == 6
    assert set(df["regime"]) == {"S1", "S2"}
    assert (df["alpha"] == 0.10).all()


@pytest.mark.unit
def test_evaluate_regime_filter(tmp_path: Path) -> None:
    """`--regimes` restricts the output to a subset."""
    npz = _synthetic_npz(tmp_path, regimes=("S1", "S2", "S5"))
    csv_path = tmp_path / "out.csv"
    result = runner.invoke(
        app,
        [
            "evaluate",
            "--idr-npz",
            str(npz),
            "--alpha",
            "0.05",
            "--regimes",
            "S5",
            "--out",
            str(csv_path),
        ],
    )
    assert result.exit_code == 0
    df = pd.read_csv(csv_path)
    assert set(df["regime"]) == {"S5"}


@pytest.mark.unit
def test_evaluate_rejects_bad_alpha(tmp_path: Path) -> None:
    """`--alpha` must be in (0, 1)."""
    npz = _synthetic_npz(tmp_path)
    result = runner.invoke(
        app,
        ["evaluate", "--idr-npz", str(npz), "--alpha", "1.5"],
    )
    assert result.exit_code != 0
    assert "alpha" in result.output


@pytest.mark.unit
def test_evaluate_rejects_empty_filter(tmp_path: Path) -> None:
    """If `--regimes` filters out everything, the user gets a clear error."""
    npz = _synthetic_npz(tmp_path)
    result = runner.invoke(
        app,
        [
            "evaluate",
            "--idr-npz",
            str(npz),
            "--alpha",
            "0.05",
            "--regimes",
            "S99",
        ],
    )
    assert result.exit_code != 0
    assert "No matching regimes" in result.output


@pytest.mark.unit
def test_evaluate_higher_alpha_calls_more_features(tmp_path: Path) -> None:
    """Sanity: at higher alpha, Sun-Cai calls (k_alpha) and power should be at least as high."""
    npz = _synthetic_npz(tmp_path)
    csv_low = tmp_path / "low.csv"
    csv_high = tmp_path / "high.csv"
    for alpha, out in ((0.01, csv_low), (0.20, csv_high)):
        result = runner.invoke(
            app,
            [
                "evaluate",
                "--idr-npz",
                str(npz),
                "--alpha",
                str(alpha),
                "--out",
                str(out),
            ],
        )
        assert result.exit_code == 0
    df_low = pd.read_csv(csv_low)
    df_high = pd.read_csv(csv_high)
    assert df_high["k_alpha"].mean() >= df_low["k_alpha"].mean()
