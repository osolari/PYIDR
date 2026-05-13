#!/usr/bin/env bash
# launch_remote.sh — drive a GPU instance over SSH for the PY-IDR
# production-scale sweep. Five subcommands cover the whole lifecycle:
#
#   probe     — report what's installed on the remote (CUDA, Python, Docker,
#               disk space). Run this first.
#   bootstrap — install missing dependencies on the remote (Python venv +
#               jax[cuda12]) for the native-venv path. Idempotent.
#   sync      — rsync the local repo to the remote, excluding .venv, runs,
#               artifacts, .git.
#   run       — launch a `py-idr sim` (or any other py-idr subcommand) on the
#               remote in the background. Returns the remote log path.
#   fetch     — rsync the remote run directory back to the local
#               artifacts/<basename> tree.
#
# Required env:
#   GPU_HOST     ssh target, e.g. "ubuntu@1.2.3.4" (no "-l" prefix).
#   REMOTE_REPO  default: ~/PYIDR (resolved on the remote).
#
# Optional env:
#   SSH_KEY      path to a private key file (passed via `ssh -i`).
#   SSH_OPTS     extra ssh flags (e.g. "-p 2222").
#   PYTHON       remote python binary; default `python3.11`.
#
# Usage examples:
#
#   GPU_HOST=ubuntu@1.2.3.4 scripts/launch_remote.sh probe
#   GPU_HOST=ubuntu@1.2.3.4 scripts/launch_remote.sh bootstrap
#   GPU_HOST=ubuntu@1.2.3.4 scripts/launch_remote.sh sync
#   GPU_HOST=ubuntu@1.2.3.4 scripts/launch_remote.sh run \
#       bash scripts/sweep_submission.sh /scratch/sweep_prod
#   GPU_HOST=ubuntu@1.2.3.4 scripts/launch_remote.sh fetch /scratch/sweep_prod

set -euo pipefail

GPU_HOST="${GPU_HOST:-}"
REMOTE_REPO="${REMOTE_REPO:-~/PYIDR}"
PYTHON="${PYTHON:-python3.11}"
SSH_KEY="${SSH_KEY:-}"
SSH_OPTS="${SSH_OPTS:-}"

if [ -z "$GPU_HOST" ]; then
    echo "error: GPU_HOST is required (e.g. GPU_HOST=user@1.2.3.4)" >&2
    exit 2
fi

SSH=("ssh")
# macOS rsync 2.6 doesn't have --info=progress2; -P (= --partial --progress)
# works on both old and new rsync.
RSYNC=("rsync" "-avzP")
if [ -n "$SSH_KEY" ]; then
    SSH+=("-i" "$SSH_KEY")
    RSYNC+=("-e" "ssh -i $SSH_KEY $SSH_OPTS")
elif [ -n "$SSH_OPTS" ]; then
    RSYNC+=("-e" "ssh $SSH_OPTS")
fi
if [ -n "$SSH_OPTS" ]; then
    # shellcheck disable=SC2206
    SSH+=($SSH_OPTS)
fi

LOCAL_REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# ---------------- subcommands ----------------

cmd_probe() {
    echo ">>> Probing $GPU_HOST"
    "${SSH[@]}" "$GPU_HOST" bash -lc "'
        set -e
        echo
        echo \"=== uname ===\"
        uname -a
        echo
        echo \"=== nvidia-smi ===\"
        if command -v nvidia-smi >/dev/null 2>&1; then
            nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
        else
            echo \"nvidia-smi: not found\"
        fi
        echo
        echo \"=== CUDA toolkit ===\"
        if command -v nvcc >/dev/null 2>&1; then
            nvcc --version | tail -1
        else
            echo \"nvcc: not found (driver-only; jax[cuda12] wheel ships its own runtime, so this may be fine)\"
        fi
        echo
        echo \"=== Python ===\"
        for v in python3.11 python3.12 python3.10 python3.9 python3; do
            if command -v \$v >/dev/null 2>&1; then
                echo \"\$v -> \$(\$v --version 2>&1)\"
            fi
        done
        echo
        echo \"=== Docker ===\"
        if command -v docker >/dev/null 2>&1; then
            docker --version
            docker info --format \"  runtime: {{.DefaultRuntime}}; gpus: {{.Runtimes.nvidia}}\" 2>/dev/null || echo \"  (docker info denied; user not in docker group)\"
        else
            echo \"docker: not found\"
        fi
        echo
        echo \"=== git ===\"
        git --version 2>&1 || echo \"git: not found\"
        echo
        echo \"=== disk space ===\"
        df -h --output=target,size,avail | head -10
        echo
        echo \"=== $REMOTE_REPO ===\"
        if [ -d $REMOTE_REPO ]; then
            (cd $REMOTE_REPO && git rev-parse HEAD 2>/dev/null && du -sh . 2>/dev/null) || true
        else
            echo \"$REMOTE_REPO: not yet present (sync will create it)\"
        fi
    '"
}

cmd_bootstrap() {
    echo ">>> Bootstrapping $GPU_HOST native-venv path"
    "${SSH[@]}" "$GPU_HOST" bash -lc "'
        set -euo pipefail
        mkdir -p $REMOTE_REPO
        cd $REMOTE_REPO
        if [ ! -d .venv ]; then
            $PYTHON -m venv .venv
        fi
        .venv/bin/pip install --upgrade pip wheel
        # The project ships a pyproject.toml with the [dev] extra; install
        # editable so the venv reflects the rsynced source tree.
        if [ -f pyproject.toml ]; then
            .venv/bin/pip install -e \".[dev]\"
        else
            echo \"warning: pyproject.toml not present yet; run scripts/launch_remote.sh sync first\"
        fi
        .venv/bin/pip install --upgrade \"jax[cuda12]\" \\
            -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
        echo
        .venv/bin/python -c \"import jax; print(\\\"backend:\\\", jax.default_backend()); print(\\\"devices:\\\", jax.devices())\"
    '"
}

cmd_sync() {
    echo ">>> Syncing $LOCAL_REPO_ROOT/ -> $GPU_HOST:$REMOTE_REPO/"
    "${RSYNC[@]}" \
        --exclude .venv/ \
        --exclude runs/ \
        --exclude artifacts/ \
        --exclude .git/ \
        --exclude __pycache__/ \
        --exclude .pytest_cache/ \
        --exclude .mypy_cache/ \
        --exclude .ruff_cache/ \
        --exclude '*.egg-info/' \
        --exclude /tmp/ \
        "$LOCAL_REPO_ROOT/" "$GPU_HOST:$REMOTE_REPO/"
}

cmd_run() {
    if [ "$#" -lt 1 ]; then
        echo "usage: $0 run <command> [args...]" >&2
        exit 2
    fi
    local LOGNAME
    LOGNAME="run-$(date +%Y%m%d-%H%M%S).log"
    echo ">>> Launching on $GPU_HOST: $* (log: $REMOTE_REPO/$LOGNAME)"
    # Run in the remote's venv; nohup + background so the SSH session can exit.
    "${SSH[@]}" "$GPU_HOST" bash -lc "'
        cd $REMOTE_REPO
        export JAX_PLATFORMS=cuda
        export XLA_PYTHON_CLIENT_PREALLOCATE=false
        export PATH=\"\$PWD/.venv/bin:\$PATH\"
        nohup $* > $LOGNAME 2>&1 &
        REMOTE_PID=\$!
        echo \"remote PID: \$REMOTE_PID\"
        echo \"log path:   $REMOTE_REPO/$LOGNAME\"
        disown \$REMOTE_PID
    '"
    echo
    echo "Tail the log with:"
    echo "  ${SSH[*]} $GPU_HOST tail -f $REMOTE_REPO/$LOGNAME"
}

cmd_tail() {
    if [ "$#" -lt 1 ]; then
        echo "usage: $0 tail <logfile>" >&2
        exit 2
    fi
    "${SSH[@]}" "$GPU_HOST" tail -f "$REMOTE_REPO/$1"
}

cmd_fetch() {
    if [ "$#" -lt 1 ]; then
        echo "usage: $0 fetch <remote_run_dir>" >&2
        exit 2
    fi
    local REMOTE_DIR="$1"
    local LOCAL_DIR="$LOCAL_REPO_ROOT/artifacts/$(basename "$REMOTE_DIR")"
    mkdir -p "$LOCAL_DIR"
    echo ">>> Pulling $GPU_HOST:$REMOTE_DIR -> $LOCAL_DIR/"
    "${RSYNC[@]}" "$GPU_HOST:$REMOTE_DIR/" "$LOCAL_DIR/"
    echo
    echo "Local copy: $LOCAL_DIR"
}

# ---------------- dispatch ----------------

if [ "$#" -lt 1 ]; then
    cat <<'USAGE'
Usage: scripts/launch_remote.sh <subcommand> [args...]

Subcommands:
  probe              Report remote state (CUDA, Python, Docker, disk).
  bootstrap          Install Python venv + jax[cuda12] on remote. Idempotent.
  sync               rsync local repo to remote (excludes .venv, runs, .git).
  run <cmd> [args]   Launch a command on the remote in the background.
  tail <log>         tail -f a remote log file.
  fetch <run_dir>    rsync a remote run directory back to artifacts/.

Required: GPU_HOST=user@ip.
Optional: REMOTE_REPO (default ~/PYIDR), PYTHON (default python3.11),
          SSH_KEY, SSH_OPTS.
USAGE
    exit 0
fi

SUB="$1"
shift
case "$SUB" in
    probe)     cmd_probe "$@" ;;
    bootstrap) cmd_bootstrap "$@" ;;
    sync)      cmd_sync "$@" ;;
    run)       cmd_run "$@" ;;
    tail)      cmd_tail "$@" ;;
    fetch)     cmd_fetch "$@" ;;
    *)
        echo "unknown subcommand: $SUB" >&2
        exit 2
        ;;
esac
