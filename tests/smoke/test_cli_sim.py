"""End-to-end tests for the ``py-idr sim`` CLI command."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from typer.testing import CliRunner

from py_idr.cli import app
from py_idr.data.schema import RunManifest

runner = CliRunner()


@pytest.mark.integration
@pytest.mark.slow
def test_py_idr_sim_writes_csv_and_manifest(tmp_path: Path) -> None:
    """End-to-end ``py-idr sim`` writes a non-empty CSV and a valid RunManifest."""
    out_root = tmp_path / "sim_out"
    result = runner.invoke(
        app,
        [
            "sim",
            "--regime",
            "S1",
            "--K",
            "2",
            "--n",
            "60",
            "--reps",
            "2",
            "--alphas",
            "0.05,0.10",
            "--num-chains",
            "2",
            "--n-warmup",
            "10",
            "--n-samples",
            "15",
            "--nuts-warmup",
            "10",
            "--out",
            str(out_root),
        ],
    )
    assert result.exit_code == 0, result.output

    # Exactly one run subdir under out.
    run_dirs = [p for p in out_root.iterdir() if p.is_dir()]
    assert len(run_dirs) == 1
    run_dir = run_dirs[0]

    # CSV exists and has the right shape: 2 replicates × 2 alphas = 4 rows.
    csv = run_dir / "results.csv"
    assert csv.exists()
    df = pd.read_csv(csv)
    assert len(df) == 4
    assert set(df["regime"]) == {"S1"}
    assert set(df["alpha"]) == {0.05, 0.10}
    assert sorted(df["replicate_seed"].unique()) == [0, 1]

    # Manifest exists and round-trips through the pydantic schema.
    manifest_path = run_dir / "run_manifest.json"
    assert manifest_path.exists()
    manifest = RunManifest.model_validate_json(manifest_path.read_text())
    assert manifest.inference_method == "mcmc"
    assert manifest.seed == 0


@pytest.mark.unit
def test_py_idr_sim_rejects_unknown_regime() -> None:
    """An unknown regime label exits non-zero with a helpful message."""
    result = runner.invoke(
        app,
        [
            "sim",
            "--regime",
            "S99",
            "--K",
            "2",
            "--n",
            "20",
            "--reps",
            "1",
        ],
    )
    assert result.exit_code != 0
    assert "unknown regime" in result.output or "S99" in result.output


@pytest.mark.integration
@pytest.mark.slow
def test_py_idr_sim_seed_schedule_is_reproducible(tmp_path: Path) -> None:
    """Two runs with the same base_seed produce identical CSVs (modulo run_id)."""
    out_a = tmp_path / "a"
    out_b = tmp_path / "b"
    for out in (out_a, out_b):
        result = runner.invoke(
            app,
            [
                "sim",
                "--regime",
                "S1",
                "--K",
                "2",
                "--n",
                "40",
                "--reps",
                "2",
                "--alphas",
                "0.05",
                "--base-seed",
                "7",
                "--num-chains",
                "2",
                "--n-warmup",
                "10",
                "--n-samples",
                "15",
                "--nuts-warmup",
                "10",
                "--out",
                str(out),
            ],
        )
        assert result.exit_code == 0

    df_a = pd.read_csv(next(out_a.iterdir()) / "results.csv")
    df_b = pd.read_csv(next(out_b.iterdir()) / "results.csv")
    pd.testing.assert_frame_equal(df_a, df_b)
