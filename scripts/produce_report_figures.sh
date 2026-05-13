#!/usr/bin/env bash
# produce_report_figures.sh — regenerate the report's F1..F5 PDFs from real
# (small but legitimate) PY-IDR sweeps. Replaces the illustrative placeholders
# in `docs/report/figures/`.
#
# Usage:
#   scripts/produce_report_figures.sh [OUT_ROOT]
#
# OUT_ROOT defaults to /tmp/pyidr_report_figures. Per-figure parameters can be
# tuned via env vars (default values are smoke-sized; override for the
# production sweep — see plans/09_figures_and_tables.md):
#
#   REPS        : per-cell replicate count          (default 3, report uses 100)
#   N           : per-replicate sample size         (default 80)
#   K           : replicate count                   (default 2)
#   ALPHAS      : comma-separated Sun-Cai levels    (default 0.01,0.05,0.10,0.20)
#   NUM_CHAINS, N_WARMUP, N_SAMPLES, NUTS_WARMUP    : chain budget knobs
#   SCALING_N_LIST : space-separated n values for F3/F5  (default "40 80 160")
#
# Output layout:
#   $OUT_ROOT/main/<regime>/<run_id>/...    # per-regime sweep (Section A)
#   $OUT_ROOT/scaling/n<N>/<run_id>/...     # n-varied sweep   (Section B)
#   $OUT_ROOT/combined_main.csv             # concat of all Section A CSVs
#   $OUT_ROOT/scaling.csv                   # concat of all Section B CSVs
#   $OUT_ROOT/figures/fig_f{1..5}.pdf
#
# F4 needs an `idr_traces.npz` sidecar with the regime it plots; we
# `--save-idr` on the S5 cell (F4's default regime) and point the figure CLI
# at that NPZ.

set -euo pipefail

cd "$(git rev-parse --show-toplevel)"

OUT_ROOT="${1:-/tmp/pyidr_report_figures}"
mkdir -p "$OUT_ROOT/figures"

JAX_PLATFORMS="${JAX_PLATFORMS:-cpu}"
export JAX_PLATFORMS

PYIDR="${PYIDR:-.venv/bin/py-idr}"
if [ ! -x "$PYIDR" ]; then
    echo "error: $PYIDR not found or not executable; set PYIDR=<path>" >&2
    exit 1
fi

# Smoke-size defaults. For the production-size sweep called out in §4.1 of
# the report, set REPS=100 N=10000 K=5 (and budget several hours).
REPS="${REPS:-3}"
N="${N:-80}"
K="${K:-2}"
ALPHAS="${ALPHAS:-0.01,0.05,0.10,0.20}"
NUM_CHAINS="${NUM_CHAINS:-2}"
N_WARMUP="${N_WARMUP:-30}"
N_SAMPLES="${N_SAMPLES:-60}"
NUTS_WARMUP="${NUTS_WARMUP:-15}"
SCALING_N_LIST="${SCALING_N_LIST:-40 80 160}"

# Helper — concatenate one or more results.csv into a single long-format CSV.
# Preserves the header from the first file and skips it on the rest.
concat_csvs() {
    local out="$1"
    shift
    local first="$1"
    head -n 1 "$first" > "$out"
    for f in "$@"; do
        tail -n +2 "$f" >> "$out"
    done
}

echo "============================================================"
echo "Section A — multi-regime sweep (F1 calibration, F2 cluster count)"
echo "============================================================"
A_CSVS=()
S5_NPZ=""
for REGIME in S1 S2 S4 S5; do
    REGIME_OUT="$OUT_ROOT/main/$REGIME"
    # --save-idr only on S5 so F4 has data for its default regime.
    # Use --save-idr / --no-save-idr (typer boolean flag) to keep the
    # expansion non-empty under `set -u`.
    if [ "$REGIME" = "S5" ]; then
        SAVE_IDR_FLAG="--save-idr"
    else
        SAVE_IDR_FLAG="--no-save-idr"
    fi
    echo "--- $REGIME ---"
    "$PYIDR" sim \
        --regime "$REGIME" --K "$K" --n "$N" --reps "$REPS" \
        --alphas "$ALPHAS" --method py-idr \
        --num-chains "$NUM_CHAINS" --n-warmup "$N_WARMUP" \
        --n-samples "$N_SAMPLES" --nuts-warmup "$NUTS_WARMUP" \
        --out "$REGIME_OUT" \
        "$SAVE_IDR_FLAG"
    # Stash the latest run's outputs.
    LATEST_RUN="$(find "$REGIME_OUT" -maxdepth 1 -mindepth 1 -type d | sort | tail -n 1)"
    A_CSVS+=("$LATEST_RUN/results.csv")
    if [ "$REGIME" = "S5" ]; then
        S5_NPZ="$LATEST_RUN/idr_traces.npz"
    fi
done
COMBINED_MAIN="$OUT_ROOT/combined_main.csv"
concat_csvs "$COMBINED_MAIN" "${A_CSVS[@]}"
echo "Combined main CSV: $(wc -l < "$COMBINED_MAIN") lines"

echo
echo "============================================================"
echo "Section B — scaling sweep (F3 runtime, F5 contraction)"
echo "============================================================"
B_CSVS=()
for SCALING_N in $SCALING_N_LIST; do
    SCALING_OUT="$OUT_ROOT/scaling/n${SCALING_N}"
    echo "--- S1, n=$SCALING_N ---"
    "$PYIDR" sim \
        --regime S1 --K "$K" --n "$SCALING_N" --reps "$REPS" \
        --alphas "$ALPHAS" --method py-idr \
        --num-chains "$NUM_CHAINS" --n-warmup "$N_WARMUP" \
        --n-samples "$N_SAMPLES" --nuts-warmup "$NUTS_WARMUP" \
        --out "$SCALING_OUT"
    LATEST_RUN="$(find "$SCALING_OUT" -maxdepth 1 -mindepth 1 -type d | sort | tail -n 1)"
    B_CSVS+=("$LATEST_RUN/results.csv")
done
SCALING_CSV="$OUT_ROOT/scaling.csv"
concat_csvs "$SCALING_CSV" "${B_CSVS[@]}"
echo "Scaling CSV: $(wc -l < "$SCALING_CSV") lines"

echo
echo "============================================================"
echo "Figure generation"
echo "============================================================"
FIGS="$OUT_ROOT/figures"

# F1 — calibration across regimes.
"$PYIDR" figure --fig f1 --results "$COMBINED_MAIN" --out "$FIGS/fig_f1.pdf"
# F2 — cluster-count histogram (auto-narrows to S5 when --regimes default).
"$PYIDR" figure --fig f2 --results "$COMBINED_MAIN" --out "$FIGS/fig_f2.pdf"
# F3 — runtime vs n on S1.
"$PYIDR" figure --fig f3 --results "$SCALING_CSV" --regimes S1 --out "$FIGS/fig_f3.pdf"
# F4 — ROC / PR on S5 (uses the NPZ sidecar from Section A's S5 cell).
if [ -n "$S5_NPZ" ] && [ -f "$S5_NPZ" ]; then
    "$PYIDR" figure --fig f4 --results "$COMBINED_MAIN" \
        --idr-npz "$S5_NPZ" --regimes S5 --out "$FIGS/fig_f4.pdf"
else
    echo "warning: no S5 idr_traces.npz found at $S5_NPZ — skipping F4" >&2
fi
# F5 — Hellinger contraction on S1.
"$PYIDR" figure --fig f5 --results "$SCALING_CSV" --out "$FIGS/fig_f5.pdf"

echo
echo "============================================================"
echo "All figures written to $FIGS"
echo "============================================================"
ls -la "$FIGS"

echo
echo "To overwrite the report's placeholder PDFs, run:"
echo "  cp $FIGS/fig_f1.pdf docs/report/figures/fig_idr_calibration.pdf"
echo "  cp $FIGS/fig_f2.pdf docs/report/figures/fig_py_vs_dp.pdf"
echo "  cp $FIGS/fig_f3.pdf docs/report/figures/fig_runtime.pdf"
echo "  cp $FIGS/fig_f4.pdf docs/report/figures/fig_roc_pr.pdf"
echo "  cp $FIGS/fig_f5.pdf docs/report/figures/fig_contraction.pdf"
