# Plan 13 — Reproducibility Infrastructure

**Owner:** W6
**Depends on:** all implementation workstreams (W1–W5).
**Implements:** §4.9 of the revised report; handoff IMPL-10.

## Status snapshot (W8.1)

| Deliverable | Module | Status |
|-------------|--------|--------|
| `DatasetManifest` pydantic schema | `data/schema.DatasetManifest` (extra="forbid", round-trips JSON+YAML) | DONE |
| `RunManifest` pydantic schema | `data/schema.RunManifest` (run_id, git_sha, config_hash, seed, started_at, inference_method) | DONE |
| `SeedLedger` schema | `data/schema.SeedLedger` | DONE |
| run_id constructor | `utils/run.new_run_id` (`<unix_ms>-<8hex>`; 32-bit suffix to avoid birthday-paradox at sweep scale) | DONE |
| git SHA detection | `utils/run.detect_git_sha` | DONE |
| Per-run manifest writer | `py-idr sim` writes `<run_id>/run_manifest.json` with `config_hash = SHA-256(canonical_config_json)` | DONE |
| NPZ idr-trace sidecar | `simulation/sweep.{write,load}_idr_sidecar` | DONE (W7.3) |
| Dockerfile | — | TODO (plan 14) |
| Snakemake / replay script | — | TODO |
| Verdict ledger | — | TODO |

## Goal

Make every run a complete, frozen, replayable experiment. From a clean checkout +
container image, the user can rerun any simulation cell or real-data analysis and
regenerate manuscript-quality tables and figures with bit-identical artifacts (modulo
hardware-stochastic GPU operations).

## Required artifacts (revised §4.9)

The revised report mandates a reproducibility package containing:

| Artifact | Source of truth |
|---|---|
| **Seed ledger** | `runs/<run_id>/seeds.yaml` — base seed, per-step PRNGKey derivations |
| **Environment file** | `runs/<run_id>/env.lock` — `pip freeze` + `nvidia-smi` snapshot |
| **Frozen data manifests** | `runs/<run_id>/data_manifests/<dataset>.yaml` (per plan 08) |
| **Simulation configuration files** | `runs/<run_id>/configs/` — every Hydra config used |
| **Model-output schemas** | `src/py_idr/data/schema.py` — pydantic for SamplerState, DiagnosticsReport, RunManifest |
| **Figure / table generation scripts** | `src/py_idr/figures/`, `src/py_idr/tables/`; consume only `runs/<run_id>/` artifacts |
| **Container or workflow target** | plan 14 (Docker on a10-dev); also a Snakemake workflow target |

## Deliverables

### `src/py_idr/data/schema.py` (new)

Pydantic v2 models:

```python
class DatasetManifest(BaseModel):
    accession: str
    release: str
    download_url: HttpUrl
    sha256: str
    feature_universe_path: Path
    replicate_matching_path: Path
    score_dictionary_path: Path
    tie_policy: Literal["mid_rank", "random", "truncate"]
    missingness_policy: Literal["mean_rank", "drop", "nan"]
    qc_summary_path: Path
    provenance_notes: str

class RunManifest(BaseModel):
    run_id: str          # ULID-style; sortable by time
    git_sha: str
    config_hash: str     # sha256 of the resolved Hydra config
    dataset_manifest: DatasetManifest | None = None
    seed: int
    started_at: datetime
    finished_at: datetime | None = None
    walltime_s: float | None = None
    hardware: dict[str, str]      # gpu name, driver, host, cuda
    software_versions: dict[str, str]   # jax, numpyro, py_idr
    inference_method: Literal["mcmc", "vi"]
    diagnostics_verdict: Literal["pass", "warn", "fail"] | None = None

class SeedLedger(BaseModel):
    base_seed: int
    derivations: dict[str, list[int]]   # step -> ordered PRNGKey ints
```

### `src/py_idr/utils/run.py` (new)

- `new_run_dir() -> Path` — creates `runs/<ULID>/` and writes empty `seeds.yaml` + `env.lock`.
- `freeze_environment(run_dir)` — writes `pip freeze` and `nvidia-smi` (or platform info).
- `build_run_manifest(run_dir, **kwargs) -> RunManifest` — assembles the manifest at run end.
- `verify_run(run_dir) -> bool` — checks the manifest, seed ledger, env file, diagnostics report all exist and validate.

### `scripts/`

- `scripts/freeze_env.sh` — emits `constraints.txt` via `pip-compile`; the CI image and the Docker target both use this lockfile.
- `scripts/replay_run.sh <run_id>` — reads the run's `RunManifest` + `configs/` and re-executes the pipeline with the same seed. Asserts bit-identical posterior summaries on a deterministic CPU run; documents stochastic divergence on GPU.

### Snakemake target (`workflow/Snakefile`)

A single workflow that, given a Hydra config, runs:
1. Dataset manifest verification (plan 08).
2. Fit (plan 04 / 05).
3. Diagnostics report (plan 12).
4. Decision evaluation (plan 06).
5. Figure / table generation (plan 09).

Each step is a Snakemake rule with `input`, `output`, and `script`. Parallelism across
simulation cells uses Snakemake's wildcard expansion; parallelism across GPUs uses
plan 14's Docker harness.

### Tests

- `tests/repro/test_seed_ledger.py` — re-running a fit with the same seed produces the
  same posterior summary statistics (CPU, deterministic).
- `tests/repro/test_replay_smoke.py` — `scripts/replay_run.sh` on a tiny S1 fixture
  produces a verdict that matches the recorded one.
- `tests/repro/test_run_manifest_schema.py` — `RunManifest.model_validate(...)` round-trips.

## Acceptance criteria

- `runs/<run_id>/` contains every artifact in the table above for every fit.
- `replay_run.sh` reproduces a tiny S1 fit's posterior summary to within float tolerance
  on CPU.
- `verify_run(run_dir)` returns `False` if any artifact is missing, malformed, or
  references a missing dataset manifest.

## Notes

- The seed ledger is **per-step** because Pólya-urn cluster reassignment, NUTS, and the
  conjugate Bernstein update each consume a fresh PRNGKey via `jax.random.fold_in`.
  Reconstructing the per-step keys from a base seed is what makes a fit replayable.
- ULID-style run IDs are sortable by start time and globally unique without coordination.
- The `replay_run.sh` script is **not expected** to reproduce GPU traces bit-identically;
  it asserts that the *posterior summary* (means, R-hat, ESS, idr) matches within a
  small tolerance. Bit-identical replay is a CPU-only guarantee.
