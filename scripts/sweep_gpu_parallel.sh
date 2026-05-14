#!/usr/bin/env bash
# sweep_gpu_parallel.sh — run sweep_submission.sh across 4 GPUs in parallel.
#
# Splits the 5 simulation regimes (S1..S5) across the 4 A10G GPUs on the
# omids@a10-dev box. Each GPU runs `sweep_submission.sh` with its own
# REGIMES list and writes to its own subdirectory; the scaling sweep
# (F3 / F5) runs on GPU 0 after the regime cells finish. A final step
# concatenates the per-GPU CSVs and runs `py-idr figure` / `table` /
# `verdict` over the merged combined CSV.
#
# Rationale: each py-idr cell at n=2000 reps=100 is ~10h on a single
# GPU because Python orchestration dispatches to the GPU per NUTS
# iteration and the per-call overhead dominates the small-n compute.
# Splitting by regime gives 4x speedup without code changes — each
# regime cell still runs sequentially, just on its own device.
#
# Usage:
#   bash scripts/sweep_gpu_parallel.sh [OUT_ROOT]
#
# Override sweep parameters via env vars (see sweep_submission.sh).
# Default GPU-scale: K=2, n=2000, reps=25, num_chains=4, warmup=100,
# samples=200 — fits in ~2h wall.

set -euo pipefail

cd "$(git rev-parse --show-toplevel 2>/dev/null || echo "$HOME/PYIDR")"

OUT_ROOT="${1:-/spare/sweep_gpu_parallel}"
mkdir -p "$OUT_ROOT"

JAX_PLATFORMS="${JAX_PLATFORMS:-cuda}"
export JAX_PLATFORMS
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"

PYIDR="${PYIDR:-.venv/bin/py-idr}"

# GPU-scale defaults. Tuned so the full pipeline completes in ~2h wall
# on the 4× A10G box.
export REPS="${REPS:-25}"
export N="${N:-2000}"
export K="${K:-2}"
export ALPHAS="${ALPHAS:-0.01,0.05,0.10,0.20}"
export NUM_CHAINS="${NUM_CHAINS:-4}"
export N_WARMUP="${N_WARMUP:-100}"
export N_SAMPLES="${N_SAMPLES:-200}"
export NUTS_WARMUP="${NUTS_WARMUP:-20}"
export SCALING_N_LIST="${SCALING_N_LIST:-500 1000 2000 5000}"
export SCALING_REPS="${SCALING_REPS:-25}"

echo "============================================================"
echo "GPU parallel sweep: K=$K n=$N reps=$REPS chains=$NUM_CHAINS"
echo "OUT_ROOT=$OUT_ROOT"
echo "============================================================"

# Regime → GPU assignment. 5 regimes / 4 GPUs: GPU 0 gets S1, GPU 1
# gets S2, GPU 2 gets S3, GPU 3 gets S4 + S5 (the heavy-tail one runs
# the multi-cluster path which is slower — keep it on the last GPU so
# the others finish first and can pick up the scaling sweep).
declare -a REGIME_BUCKETS=("S1" "S2" "S3" "S4 S5")

PIDS=()
for gpu in 0 1 2 3; do
    bucket="${REGIME_BUCKETS[$gpu]}"
    bucket_label=$(echo "$bucket" | tr ' ' '_')
    BUCKET_OUT="$OUT_ROOT/gpu${gpu}_${bucket_label}"
    echo
    echo ">>> GPU $gpu: regimes [$bucket] -> $BUCKET_OUT"
    (
        export CUDA_VISIBLE_DEVICES="$gpu"
        export REGIMES="$bucket"
        # Skip the scaling sweep in the per-GPU runs; we run it once at
        # the end against GPU 0. The sweep_submission.sh script doesn't
        # have a flag for this so we run a minimal one-element scaling
        # list and ignore the output.
        export SCALING_N_LIST="50"
        export SCALING_REPS="1"
        bash scripts/sweep_submission.sh "$BUCKET_OUT" \
            > "$OUT_ROOT/gpu${gpu}.log" 2>&1
    ) &
    PIDS+=($!)
done

echo
echo "Per-GPU PIDs: ${PIDS[@]}"
echo "Tail any with: tail -f $OUT_ROOT/gpu<N>.log"
echo
echo "Waiting for all 4 GPU buckets to finish..."
FAILED=0
for pid in "${PIDS[@]}"; do
    if ! wait "$pid"; then
        echo "  pid $pid failed; check $OUT_ROOT/gpu*.log" >&2
        FAILED=1
    fi
done
if [ "$FAILED" -ne 0 ]; then
    echo "At least one GPU bucket failed; aborting before merge." >&2
    exit 1
fi

echo
echo "============================================================"
echo "All GPU buckets finished. Merging combined_main.csv files."
echo "============================================================"

MERGED_MAIN="$OUT_ROOT/combined_main.csv"
FIRST=1
for gpu in 0 1 2 3; do
    bucket_label=$(echo "${REGIME_BUCKETS[$gpu]}" | tr ' ' '_')
    csv="$OUT_ROOT/gpu${gpu}_${bucket_label}/combined_main.csv"
    if [ ! -f "$csv" ]; then
        echo "warning: $csv missing — skipping" >&2
        continue
    fi
    if [ "$FIRST" -eq 1 ]; then
        cp "$csv" "$MERGED_MAIN"
        FIRST=0
    else
        tail -n +2 "$csv" >> "$MERGED_MAIN"
    fi
done
echo "Merged main CSV: $(wc -l < "$MERGED_MAIN") lines"

# Use the GPU 3 bucket's S5 NPZ for F4 (--save-idr was on for S5).
S5_NPZ="$(find "$OUT_ROOT/gpu3_S4_S5/main/S5/pyidr" -name idr_traces.npz 2>/dev/null | head -n 1)"

echo
echo "============================================================"
echo "Section B — scaling sweep on S1 (single GPU 0, sequential)"
echo "============================================================"
SCALING_OUT="$OUT_ROOT/scaling"
mkdir -p "$SCALING_OUT"
B_CSVS=()
for SCALING_N in $SCALING_N_LIST; do
    NCELL="$SCALING_OUT/n${SCALING_N}"
    echo "--- S1 / n=$SCALING_N ---"
    CUDA_VISIBLE_DEVICES=0 \
        "$PYIDR" sim \
        --regime S1 --K "$K" --n "$SCALING_N" --reps "$SCALING_REPS" \
        --alphas "$ALPHAS" --method py-idr \
        --num-chains "$NUM_CHAINS" --n-warmup "$N_WARMUP" \
        --n-samples "$N_SAMPLES" --nuts-warmup "$NUTS_WARMUP" \
        --out "$NCELL" --no-save-idr
    LATEST="$(find "$NCELL" -maxdepth 1 -mindepth 1 -type d | sort | tail -n 1)"
    B_CSVS+=("$LATEST/results.csv")
done

SCALING_CSV="$OUT_ROOT/scaling.csv"
head -n 1 "${B_CSVS[0]}" > "$SCALING_CSV"
for f in "${B_CSVS[@]}"; do tail -n +2 "$f" >> "$SCALING_CSV"; done

echo
echo "============================================================"
echo "Generating figures + tables + verdict"
echo "============================================================"
FIGS="$OUT_ROOT/figures"
TABLES="$OUT_ROOT/tables"
mkdir -p "$FIGS" "$TABLES"

"$PYIDR" figure --fig f1 --results "$MERGED_MAIN" --out "$FIGS/fig_f1.pdf"
"$PYIDR" figure --fig f2 --results "$MERGED_MAIN" --out "$FIGS/fig_f2.pdf"
"$PYIDR" figure --fig f3 --results "$SCALING_CSV" --regimes S1 --out "$FIGS/fig_f3.pdf"
if [ -n "${S5_NPZ:-}" ] && [ -f "${S5_NPZ:-}" ]; then
    "$PYIDR" figure --fig f4 --results "$MERGED_MAIN" \
        --idr-npz "$S5_NPZ" --regimes S5 --out "$FIGS/fig_f4.pdf"
else
    echo "warning: no S5 idr_traces.npz; skipping F4" >&2
fi
"$PYIDR" figure --fig f5 --results "$SCALING_CSV" --out "$FIGS/fig_f5.pdf"

"$PYIDR" table --table t4-fdr --results "$MERGED_MAIN" \
    --alpha 0.05 --out "$TABLES/tab_sim_fdr.tex"
"$PYIDR" table --table t4-power --results "$MERGED_MAIN" \
    --alpha 0.05 --out "$TABLES/tab_sim_power.tex"

"$PYIDR" verdict --results "$MERGED_MAIN" --out "$OUT_ROOT/verdicts.json"

echo
echo "============================================================"
echo "Done. Artefacts:"
echo "  $FIGS/fig_f{1..5}.pdf"
echo "  $TABLES/tab_sim_{fdr,power}.tex"
echo "  $OUT_ROOT/verdicts.json"
echo "  $MERGED_MAIN, $SCALING_CSV"
echo "============================================================"
