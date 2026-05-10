# Plan 14 — Docker on a10-dev (4× A10 GPU runner)

**Owner:** W6
**Depends on:** plan 13 (reproducibility), plan 07 (simulations), plan 08 (real data).
**Implements:** the GPU sweep harness for the simulation factorial and the larger real-data fits.

## Goal

A Dockerised JAX/CUDA image that runs on the user's `a10-dev` host (4× A10 GPUs) and on
GitHub's GPU runners. The image is the canonical execution environment for:

- The full S1–S5 × $K \in \{2,5,10,50\}$ × $n \in \{2k, 10k, 50k\}$ + S5-sparse simulation sweep (≈ 6100 fits).
- Pan-UKBB and large ENCODE real-data runs.
- Nightly `gpu-ci.yml` correctness checks.

## Why Docker for this

- **Reproducibility.** Locked CUDA / cuDNN / JAX versions matter for any GPU result.
- **Multi-host portability.** Same image on a10-dev, on GitHub's GPU runners, on a
  future Slurm cluster.
- **Sharding.** `CUDA_VISIBLE_DEVICES=0..3` round-robin across 4 GPU processes is
  embarrassingly parallel for the simulation sweep.

## Image stack

Base: `nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04`

Layers:
1. System packages: `git`, `python3.11`, `python3.11-venv`, `python3-pip`, `build-essential`, `ca-certificates`.
2. `pip install uv` for fast wheel resolution.
3. Copy `pyproject.toml` + `constraints.txt`; `uv pip install -e ".[dev,cuda,data,viz]"`.
4. Copy the rest of the repo.
5. `ENV JAX_PLATFORMS=cuda`, `XLA_PYTHON_CLIENT_PREALLOCATE=false`, `JAX_ENABLE_X64=1`.
6. `ENTRYPOINT ["python", "-m", "py_idr.cli"]` — so `docker run pyidr fit ...` Just Works.

## Deliverables

### `Dockerfile` (root of repo)

A multi-stage build:
- Stage `builder`: install system deps + uv + project.
- Stage `runtime`: copy only `.venv` and the source tree; minimal layer.
Result image: ≤ 4 GB.

### `.dockerignore`

Excludes `data/`, `runs/`, `artifacts/`, `checkpoints/`, `logs/`, `.venv/` (we install
it inside the image), `docs/report/figures/*.pdf`, `notebooks/`.

### `scripts/run_on_a10dev.sh`

```bash
#!/usr/bin/env bash
# Run a single PY-IDR command inside the Docker image on a10-dev.
# Usage: scripts/run_on_a10dev.sh fit --config configs/data/encode_ctcf.yaml
set -euo pipefail
IMAGE="${PYIDR_IMAGE:-ghcr.io/osolari/pyidr:latest}"
docker run --rm --gpus all \
    -e XLA_PYTHON_CLIENT_PREALLOCATE=false \
    -e JAX_PLATFORMS=cuda \
    -v "$PWD/runs:/workspace/runs" \
    -v "$PWD/artifacts:/workspace/artifacts" \
    -v "$PWD/data:/workspace/data:ro" \
    -v "$PWD/configs:/workspace/configs:ro" \
    -w /workspace "$IMAGE" "$@"
```

### `scripts/sweep_simulations_a10dev.sh`

Round-robin sharding across the 4 A10s using GNU parallel:

```bash
#!/usr/bin/env bash
set -euo pipefail
# Generate the cell list once.
python -m py_idr.simulation.designs --emit-cells > /tmp/cells.txt
# Run 4 cells at a time, one per GPU.
parallel -j 4 --colsep ' ' \
    'CUDA_VISIBLE_DEVICES=$(({%} - 1)) bash scripts/run_on_a10dev.sh sim \
        --regime {1} --K {2} --n {3} --reps 100 \
        --out artifacts/sim/{1}_K{2}_n{3}/' :::: /tmp/cells.txt
```

`{%}` is the GNU parallel job slot (1–4 with `-j 4`); we map to `CUDA_VISIBLE_DEVICES`
in `0..3`.

### `.github/workflows/gpu-ci.yml`

Add a job that:
1. Logs into GHCR.
2. Pulls `ghcr.io/osolari/pyidr:latest`.
3. Runs `pytest -m "gpu and not slow"` inside the container on a self-hosted GPU runner
   (or skips if no GPU runner is online).

### `.github/workflows/build-image.yml`

On every push to `main` (or on manual dispatch), builds the image and pushes to GHCR
with tags `:latest` and `:sha-<git-sha>`.

## Operational notes (from the chat)

- **`XLA_PYTHON_CLIENT_PREALLOCATE=false`** is mandatory, or JAX grabs all 24 GB of one
  A10 on import and the other three GPUs' processes OOM at startup.
- **CUDA version pin.** `jax[cuda12]` ↔ CUDA 12.x. Mismatch = silent CPU fallback.
  The Dockerfile pins both.
- **Image size budget.** ≤ 4 GB after the multi-stage build. If it exceeds, audit the
  `mkdocs-jupyter` and `scanpy` extras (they pull in a lot).
- **Interactive debugging.** `docker run -it --rm --gpus all ... bash` for a shell
  inside the image. Mostly used to investigate a stuck JAX process; routine debugging
  happens locally on Mac CPU first.
- **Within-fit parallelism.** NumPyro can `pmap` 4 chains across the 4 GPUs for one big
  fit. Useful for Pan-UKBB; not used for the simulation sweep (where embarrassing
  parallelism wins).

## Acceptance criteria

- Building the image locally: `docker build -t pyidr:dev .` succeeds in < 10 min.
- `docker run pyidr:dev doctor` exits 0 inside the image.
- `scripts/run_on_a10dev.sh doctor` works on a10-dev and reports `backend=cuda`,
  `devices=4`.
- A 100-replicate S1 cell completes on one A10 in $\le$ 30 min wall.
- `scripts/sweep_simulations_a10dev.sh` runs 4 cells concurrently with one process per
  GPU, no OOM.

## Risks

| Risk | Mitigation |
|---|---|
| Image size balloons past 4 GB | Multi-stage build; audit extras. |
| `jax[cuda12]` wheel cadence changes | Pin to `jax==<exact>` in `constraints.txt`. |
| `nvidia-smi` not available in image | Surface this in `doctor()` so it's a clear failure, not a silent CPU fallback. |
| GHCR rate limits on pull | Cache the image on a10-dev with `docker pull` once per release. |
