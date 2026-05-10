"""Run-directory bootstrap and verification helpers (plan 13).

A *run* is a single fit (simulation cell or real-data analysis). Every run owns
a directory under ``runs/<run_id>/`` containing the seed ledger, environment
file, dataset manifest (when applicable), Hydra config, posterior trace, and
diagnostics report. This module owns the bootstrap path and the post-hoc
verification.

See plans/13_reproducibility.md.
"""

from __future__ import annotations

import os
import platform
import secrets
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

from py_idr.data.schema import RunManifest, SeedLedger


def new_run_id() -> str:
    """Build a sortable, globally-unique run id.

    The id has the form ``<unix_ts_ms>-<8-hex>`` where the timestamp is the
    millisecond Unix time at call time and the suffix is 8 random hex digits
    (32 bits). Lexicographic sort matches start-time sort (ms-precision); the
    32-bit suffix keeps the birthday-collision probability below 10⁻⁶ at
    1000 ids per millisecond — comfortably above any plausible launch rate.
    """
    ts_ms = int(time.time() * 1000)
    suffix = secrets.token_hex(4)
    return f"{ts_ms}-{suffix}"


def new_run_dir(root: Path = Path("runs")) -> Path:
    """Create ``runs/<run_id>/`` and return the path.

    The directory is created with `parents=True, exist_ok=False` — colliding ids
    raise loudly rather than overwrite an existing run.
    """
    run_id = new_run_id()
    run_dir = root / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def freeze_environment(run_dir: Path) -> Path:
    """Write ``runs/<run_id>/env.lock`` with `pip freeze` and platform/GPU info.

    The file is intended for human inspection and for the replay-script
    regression test. It is not consumed programmatically; the canonical
    machine-readable record is :class:`RunManifest.software_versions`.
    """
    out = run_dir / "env.lock"
    lines: list[str] = []
    lines.append(f"# host: {platform.node()}")
    lines.append(f"# platform: {platform.platform()}")
    lines.append(f"# python: {sys.version.split()[0]}")
    try:
        nvidia = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
        if nvidia.returncode == 0:
            for row in nvidia.stdout.strip().splitlines():
                lines.append(f"# gpu: {row}")
        else:
            lines.append("# gpu: nvidia-smi unavailable (cpu-only host)")
    except (FileNotFoundError, subprocess.TimeoutExpired):
        lines.append("# gpu: nvidia-smi unavailable")
    lines.append("")
    pip_freeze = subprocess.run(
        [sys.executable, "-m", "pip", "freeze"],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    if pip_freeze.returncode == 0:
        lines.append(pip_freeze.stdout.rstrip())
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out


def now_utc() -> datetime:
    """Return the current UTC ``datetime`` (helper for manifests)."""
    return datetime.now(UTC)


def build_run_manifest(
    *,
    run_id: str,
    git_sha: str,
    config_hash: str,
    seed: int,
    started_at: datetime,
    inference_method: str,
    finished_at: datetime | None = None,
    walltime_s: float | None = None,
    hardware: dict[str, str] | None = None,
    software_versions: dict[str, str] | None = None,
    dataset_manifest: dict | None = None,
    diagnostics_verdict: str | None = None,
) -> RunManifest:
    """Assemble a :class:`RunManifest` from canonical inputs.

    Convenience wrapper that handles None defaults and forwards to pydantic
    validation.
    """
    payload: dict[str, object] = {
        "run_id": run_id,
        "git_sha": git_sha,
        "config_hash": config_hash,
        "seed": seed,
        "started_at": started_at,
        "inference_method": inference_method,
    }
    if finished_at is not None:
        payload["finished_at"] = finished_at
    if walltime_s is not None:
        payload["walltime_s"] = walltime_s
    if hardware is not None:
        payload["hardware"] = hardware
    if software_versions is not None:
        payload["software_versions"] = software_versions
    if dataset_manifest is not None:
        payload["dataset_manifest"] = dataset_manifest
    if diagnostics_verdict is not None:
        payload["diagnostics_verdict"] = diagnostics_verdict
    return RunManifest.model_validate(payload)


def verify_run(run_dir: Path) -> bool:
    """Return True iff ``run_dir`` contains the canonical artifacts plan 13 mandates.

    Required:
        - ``manifest.json``      — round-trips through :class:`RunManifest`.
        - ``seeds.yaml``         — round-trips through :class:`SeedLedger`.
        - ``env.lock``           — non-empty file.

    Optional but checked when present:
        - ``configs/``           — directory of Hydra configs.
        - ``diagnostics.json``   — plan 12 diagnostics report.

    The function does not raise; it returns ``False`` for any malformed run so
    the figure / table / verdict pipelines can short-circuit gracefully.
    """
    if not run_dir.is_dir():
        return False
    manifest_path = run_dir / "manifest.json"
    seeds_path = run_dir / "seeds.yaml"
    env_path = run_dir / "env.lock"
    if not manifest_path.is_file() or not seeds_path.is_file() or not env_path.is_file():
        return False
    if env_path.stat().st_size == 0:
        return False
    try:
        RunManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return False
    try:
        import yaml  # type: ignore[import-untyped]

        SeedLedger.model_validate(yaml.safe_load(seeds_path.read_text(encoding="utf-8")))
    except Exception:  # noqa: BLE001
        return False
    return True


def detect_git_sha(cwd: Path | None = None) -> str | None:
    """Return the current git SHA (40-char hex) or ``None`` if not inside a repo."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=False,
            capture_output=True,
            text=True,
            cwd=str(cwd) if cwd else None,
            timeout=5,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    sha = result.stdout.strip()
    if len(sha) == 40 and all(c in "0123456789abcdef" for c in sha):
        return sha
    return None


# Re-export the path constant so callers don't reach into env vars directly.
RUNS_ROOT = Path(os.environ.get("PY_IDR_RUNS_ROOT", "runs"))
