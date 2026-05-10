#!/usr/bin/env bash
# Run a single PY-IDR command inside the Docker image on a10-dev (4× A10 GPUs).
#
# Usage:
#   scripts/run_on_a10dev.sh doctor
#   scripts/run_on_a10dev.sh sim --regime S1 --K 5 --n 10000 --reps 100
#   scripts/run_on_a10dev.sh fit --config configs/data/encode_ctcf.yaml --inference mcmc
#
# Environment overrides:
#   PYIDR_IMAGE          : image reference (default ghcr.io/osolari/pyidr:latest).
#   CUDA_VISIBLE_DEVICES : comma-separated GPU indices to expose (default all).
#                          The simulation sweep harness uses this to round-robin
#                          across the 4 A10s; one process per GPU.
#
# See plans/14_docker_a10dev.md.
set -euo pipefail

IMAGE="${PYIDR_IMAGE:-ghcr.io/osolari/pyidr:latest}"

# Resolve the host-side workspace root (the repo we're calling from).
WORKSPACE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Build the GPU spec. CUDA_VISIBLE_DEVICES comes through to the container intact.
GPU_FLAG="--gpus all"
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    GPU_FLAG="--gpus device=${CUDA_VISIBLE_DEVICES}"
fi

# Mounts:
#   runs/, artifacts/   — writable; the image writes traces and CSVs here
#   data/, configs/     — read-only; raw inputs and Hydra configs
#
# The -v mounts use the host paths so a fit on a10-dev appears under the user's
# workspace at runs/<run_id>/.
docker run --rm -it \
    ${GPU_FLAG} \
    -e XLA_PYTHON_CLIENT_PREALLOCATE=false \
    -e JAX_PLATFORMS=cuda \
    -e JAX_ENABLE_X64=1 \
    ${CUDA_VISIBLE_DEVICES:+-e CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}"} \
    -v "${WORKSPACE}/runs:/workspace/runs" \
    -v "${WORKSPACE}/artifacts:/workspace/artifacts" \
    -v "${WORKSPACE}/data:/workspace/data:ro" \
    -v "${WORKSPACE}/configs:/workspace/configs:ro" \
    -w /workspace \
    "${IMAGE}" \
    "$@"
