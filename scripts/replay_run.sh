#!/usr/bin/env bash
# Replay a previously recorded py-idr sim run.
#
# Usage:
#   scripts/replay_run.sh <run_dir> [--out <new_out_dir>]
#
# Reads ``<run_dir>/run_config.json`` (written by ``py-idr sim``) and
# re-executes the same simulation with identical CLI arguments. The new run
# lands under ``<new_out_dir>`` (defaults to ``<run_dir>.replay``) and gets a
# fresh run_id of its own.
#
# Acceptance criterion (plan 13):
#   On CPU, the replay reproduces the original posterior summary to within
#   float tolerance — verifiable by comparing the two ``results.csv`` files
#   row-by-row, or by checking that ``py-idr verdict`` returns the same
#   pass/warn/fail ledger.
#
# GPU replays are NOT bit-identical because nondeterministic CUDA kernels are
# legal under JAX's default settings.

set -euo pipefail

if [ "$#" -lt 1 ]; then
    echo "usage: $0 <run_dir> [--out <new_out_dir>]" >&2
    exit 2
fi

RUN_DIR="$1"
shift

NEW_OUT="${RUN_DIR}.replay"
while [ "$#" -gt 0 ]; do
    case "$1" in
        --out)
            NEW_OUT="$2"
            shift 2
            ;;
        *)
            echo "unknown argument: $1" >&2
            exit 2
            ;;
    esac
done

CONFIG="$RUN_DIR/run_config.json"
if [ ! -f "$CONFIG" ]; then
    echo "error: $CONFIG not found — run was recorded by an older py-idr build?" >&2
    exit 1
fi

PYIDR="${PYIDR:-.venv/bin/py-idr}"

# Decode the run_config.json into CLI flags. Done in Python rather than jq so
# the script has no extra dependencies beyond the py-idr venv. We avoid
# `mapfile` because macOS ships bash 3.2; a portable while/read loop works on
# both bash 3 and bash 4+.
ARGS=()
while IFS= read -r line; do
    ARGS+=("$line")
done < <(
    "${PYIDR%/py-idr}/python" - "$CONFIG" <<'PY'
import json
import sys
from pathlib import Path

cfg = json.loads(Path(sys.argv[1]).read_text())

# Order is informational; py-idr accepts any order.
flags = [
    ("--regime", str(cfg["regime"])),
    ("--K", str(cfg["K"])),
    ("--n", str(cfg["n"])),
    ("--reps", str(cfg["reps"])),
    ("--base-seed", str(cfg["base_seed"])),
    ("--alphas", ",".join(str(a) for a in cfg["alphas"])),
    ("--num-chains", str(cfg["num_chains"])),
    ("--n-warmup", str(cfg["n_warmup"])),
    ("--n-samples", str(cfg["n_samples"])),
    ("--nuts-warmup", str(cfg["nuts_warmup"])),
    ("--M", str(cfg["M"])),
    ("--chain-type", str(cfg["chain_type"])),
    ("--T-max", str(cfg["T_max"])),
    ("--H", str(cfg["H"])),
    ("--py-hyperparam-warmup", str(cfg["py_hyperparam_warmup"])),
    ("--method", str(cfg["method"])),
]
for k, v in flags:
    print(k)
    print(v)
if cfg.get("save_idr"):
    print("--save-idr")
if cfg.get("em_init"):
    print("--em-init")
PY
)

mkdir -p "$NEW_OUT"

echo "Replaying $RUN_DIR -> $NEW_OUT"
echo "Command: $PYIDR sim ${ARGS[*]} --out $NEW_OUT"
"$PYIDR" sim "${ARGS[@]}" --out "$NEW_OUT"

echo "Replay complete."
echo "Compare with: diff <(sort $RUN_DIR/results.csv) <(sort $NEW_OUT/*/results.csv)"
