#!/usr/bin/env bash
# scripts/create_env.sh — bootstrap dev env for PY-IDR (PY-IDR).
#
# Idempotent. Detects accelerator (CPU / CUDA / Apple Metal), installs the matched
# JAX wheel, sets up pre-commit. CI calls this with INSTALL_EXTRAS=dev.
set -euo pipefail
IFS=$'\n\t'

PY_VER="${PY_VER:-3.11}"
JAX_BACKEND="${JAX_BACKEND:-auto}"      # auto | cpu | cuda12 | metal
ENV_MANAGER="${ENV_MANAGER:-venv}"      # venv | uv
INSTALL_EXTRAS="${INSTALL_EXTRAS:-dev,docs,viz}"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# 1. Detect platform.
OS="$(uname -s | tr '[:upper:]' '[:lower:]')"
ARCH="$(uname -m)"
echo "[create_env] platform: ${OS}/${ARCH}"

# 2. Detect accelerator if JAX_BACKEND=auto.
if [[ "$JAX_BACKEND" == "auto" ]]; then
    if [[ "$OS" == "darwin" && "$ARCH" == "arm64" ]]; then
        JAX_BACKEND=metal
    elif command -v nvidia-smi >/dev/null 2>&1; then
        JAX_BACKEND=cuda12
    else
        JAX_BACKEND=cpu
    fi
fi
echo "[create_env] jax backend: ${JAX_BACKEND}"

# 3. Create the venv.
if [[ "$ENV_MANAGER" == "uv" ]]; then
    if ! command -v uv >/dev/null 2>&1; then
        echo "[create_env] uv not found; falling back to venv."
        ENV_MANAGER=venv
    fi
fi

if [[ "$ENV_MANAGER" == "uv" ]]; then
    uv venv --python "$PY_VER" .venv
    # shellcheck disable=SC1091
    source .venv/bin/activate
    uv pip install -e ".[${INSTALL_EXTRAS}]"
else
    python -m venv .venv
    # shellcheck disable=SC1091
    source .venv/bin/activate
    python -m pip install --upgrade pip wheel
    python -m pip install -e ".[${INSTALL_EXTRAS}]"
fi

# 4. Install backend-specific JAX wheel.
case "$JAX_BACKEND" in
    cuda12)
        python -m pip install --upgrade "jax[cuda12]" \
            -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
        ;;
    metal)
        python -m pip install --upgrade jax-metal
        ;;
    cpu)
        # already installed by the editable install above
        ;;
    *)
        echo "[create_env] unknown JAX_BACKEND=${JAX_BACKEND}" >&2
        exit 2
        ;;
esac

# 5. Pre-commit hooks.
if [[ -f .pre-commit-config.yaml ]]; then
    pre-commit install --install-hooks
fi

# 6. Smoke-check.
python -c "import py_idr; raise SystemExit(py_idr.doctor())"

echo
echo "[create_env] done. next steps:"
echo "    source .venv/bin/activate"
echo "    pytest -m smoke"
