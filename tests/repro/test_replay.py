"""Tests for the run-config persistence + ``scripts/replay_run.sh`` round-trip.

The replay machinery has two halves:

1. ``py-idr sim`` writes ``run_config.json`` alongside ``run_manifest.json``
   so a future invocation can reconstruct the original CLI arguments.
2. ``scripts/replay_run.sh`` reads that file and re-executes ``py-idr sim``.

We exercise (1) end-to-end via the typer CliRunner, then exercise the bash
script as a subprocess on the smallest sweep that still produces a useful
delta — replay-on-CPU of an S1 fit reproduces the original posterior summary
to within float tolerance, which we verify by comparing the two
``results.csv`` files.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import pandas as pd
import pytest
from typer.testing import CliRunner

from py_idr.cli import app

runner = CliRunner()


_SIM_ARGS = [
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
    "0.05,0.10",
    "--base-seed",
    "11",
    "--num-chains",
    "2",
    "--n-warmup",
    "10",
    "--n-samples",
    "15",
    "--nuts-warmup",
    "10",
]


@pytest.mark.integration
@pytest.mark.slow
def test_sim_writes_run_config_with_canonical_shape(tmp_path: Path) -> None:
    """``py-idr sim`` writes a ``run_config.json`` containing every CLI knob."""
    out_root = tmp_path / "sim_out"
    result = runner.invoke(app, _SIM_ARGS + ["--out", str(out_root)])
    assert result.exit_code == 0, result.output

    run_dir = next(p for p in out_root.iterdir() if p.is_dir())
    config_path = run_dir / "run_config.json"
    assert config_path.exists()
    cfg = json.loads(config_path.read_text())

    # The keys the replay script reads — if any of these disappear, replay
    # breaks silently. Pin them here.
    expected_keys = {
        "regime",
        "K",
        "n",
        "reps",
        "base_seed",
        "alphas",
        "num_chains",
        "n_warmup",
        "n_samples",
        "nuts_warmup",
        "M",
        "chain_type",
        "T_max",
        "H",
        "py_hyperparam_warmup",
        "save_idr",
        "method",
    }
    assert set(cfg) == expected_keys

    # Spot-check the values came from the CLI flags above.
    assert cfg["regime"] == "S1"
    assert cfg["K"] == 2
    assert cfg["n"] == 40
    assert cfg["reps"] == 2
    assert cfg["base_seed"] == 11
    assert cfg["alphas"] == [0.05, 0.10]
    assert cfg["method"] == "py-idr"


@pytest.mark.integration
@pytest.mark.slow
def test_replay_run_sh_reproduces_results_csv(tmp_path: Path) -> None:
    """``scripts/replay_run.sh`` reproduces the original results.csv on CPU."""
    script = Path(__file__).resolve().parents[2] / "scripts" / "replay_run.sh"
    py_idr_bin = Path(__file__).resolve().parents[2] / ".venv" / "bin" / "py-idr"
    if not py_idr_bin.exists() or not script.exists() or shutil.which("bash") is None:
        pytest.skip("replay_run.sh requires .venv/bin/py-idr and bash")

    out_root = tmp_path / "sim_out"
    result = runner.invoke(app, _SIM_ARGS + ["--out", str(out_root)])
    assert result.exit_code == 0, result.output
    run_dir = next(p for p in out_root.iterdir() if p.is_dir())
    original_csv = pd.read_csv(run_dir / "results.csv")

    replay_root = tmp_path / "replay_out"
    env = os.environ.copy()
    env["PYIDR"] = str(py_idr_bin)
    env.setdefault("JAX_PLATFORMS", "cpu")
    proc = subprocess.run(
        ["bash", str(script), str(run_dir), "--out", str(replay_root)],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
    assert proc.returncode == 0, f"stdout={proc.stdout}\nstderr={proc.stderr}"

    replay_run_dir = next(p for p in replay_root.iterdir() if p.is_dir())
    replay_csv = pd.read_csv(replay_run_dir / "results.csv")

    # Same row count and same (regime, replicate_seed, alpha) cells.
    assert len(replay_csv) == len(original_csv)
    join_cols = ["regime", "replicate_seed", "alpha"]
    merged = original_csv.merge(replay_csv, on=join_cols, suffixes=("_orig", "_replay"))
    assert len(merged) == len(original_csv)

    # Realized-FDR and power match within a tight float tolerance.
    # We allow modest slack because JAX's CPU reductions are deterministic
    # at the elementwise op level but mixed-precision accumulators can shift
    # the last few ulps under different machine state.
    for col in ("realized_fdr", "power"):
        diff = (merged[f"{col}_orig"] - merged[f"{col}_replay"]).abs().max()
        assert diff < 1e-6, f"replay {col} drifted by {diff}"
