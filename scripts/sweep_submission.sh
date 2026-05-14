#!/usr/bin/env bash
# sweep_submission.sh — run the medium-CPU sweep used to populate the report's
# §4 simulation tables and figures.
#
# Differs from W8.12's produce_report_figures.sh in that we run all three
# implemented methods (PY-IDR, Vanilla IDR, MaxRank) so the report's
# tab:sim-fdr / tab:sim-power tables have real numbers per method.
#
# Usage:
#   scripts/sweep_submission.sh [OUT_ROOT]
#
# Outputs:
#   $OUT_ROOT/main/<regime>/{pyidr,vanilla,maxrank}/<run_id>/...
#   $OUT_ROOT/scaling/n<N>/<run_id>/...
#   $OUT_ROOT/combined_main.csv           # all regimes × all methods
#   $OUT_ROOT/scaling.csv                 # varied n on S1, PY-IDR only
#   $OUT_ROOT/figures/fig_f{1..5}.pdf
#   $OUT_ROOT/tables/tab_sim_{fdr,power}.tex
#
# Default scale: K=2, n=100, reps=10, 5 regimes, 3 methods.
# Wall-clock estimate on a single M-series CPU: ~70 min for the main sweep
# + ~20 min for the scaling sweep + ~1 min for figures/tables. Larger
# (n>=400, reps>=20) cells are intended for the GPU production run.

set -euo pipefail

cd "$(git rev-parse --show-toplevel)"

OUT_ROOT="${1:-/tmp/pyidr_submission}"
mkdir -p "$OUT_ROOT/figures" "$OUT_ROOT/tables"

JAX_PLATFORMS="${JAX_PLATFORMS:-cpu}"
export JAX_PLATFORMS

PYIDR="${PYIDR:-.venv/bin/py-idr}"

# Submission-scale defaults (override via env). Tuned so the full pipeline
# completes in ~90 min on a single M-series CPU; bigger cells go to the
# GPU production run via scripts/launch_remote.sh.
REPS="${REPS:-10}"
N="${N:-100}"
K="${K:-2}"
ALPHAS="${ALPHAS:-0.01,0.05,0.10,0.20}"
NUM_CHAINS="${NUM_CHAINS:-2}"
N_WARMUP="${N_WARMUP:-20}"
N_SAMPLES="${N_SAMPLES:-40}"
NUTS_WARMUP="${NUTS_WARMUP:-10}"
REGIMES="${REGIMES:-S1 S2 S3 S4 S5}"
SCALING_N_LIST="${SCALING_N_LIST:-50 100 200}"
SCALING_REPS="${SCALING_REPS:-5}"
# EM warm start (W8.5) — recommended for short n_warmup; default on.
# Set EM_INIT=0 to disable.
EM_INIT="${EM_INIT:-1}"
if [ "$EM_INIT" = "1" ]; then
    EM_INIT_FLAG="--em-init"
else
    EM_INIT_FLAG="--no-em-init"
fi

concat_csvs() {
    local out="$1"
    shift
    local first="$1"
    head -n 1 "$first" > "$out"
    for f in "$@"; do
        tail -n +2 "$f" >> "$out"
    done
}

A_CSVS=()
S5_NPZ=""

echo "============================================================"
echo "Section A — multi-regime, multi-method sweep"
echo "REPS=$REPS N=$N K=$K REGIMES=\"$REGIMES\""
echo "============================================================"
for REGIME in $REGIMES; do
    echo
    echo ">>> Regime $REGIME <<<"
    # PY-IDR — full Bayesian fit with NPZ sidecar (F4 needs it for S5).
    REGIME_OUT="$OUT_ROOT/main/$REGIME/pyidr"
    if [ "$REGIME" = "S5" ]; then
        SAVE_IDR_FLAG="--save-idr"
    else
        SAVE_IDR_FLAG="--no-save-idr"
    fi
    echo "--- $REGIME / PY-IDR (em_init=${EM_INIT}) ---"
    "$PYIDR" sim \
        --regime "$REGIME" --K "$K" --n "$N" --reps "$REPS" \
        --alphas "$ALPHAS" --method py-idr \
        --num-chains "$NUM_CHAINS" --n-warmup "$N_WARMUP" \
        --n-samples "$N_SAMPLES" --nuts-warmup "$NUTS_WARMUP" \
        --out "$REGIME_OUT" "$SAVE_IDR_FLAG" "$EM_INIT_FLAG"
    LATEST="$(find "$REGIME_OUT" -maxdepth 1 -mindepth 1 -type d | sort | tail -n 1)"
    A_CSVS+=("$LATEST/results.csv")
    if [ "$REGIME" = "S5" ]; then
        S5_NPZ="$LATEST/idr_traces.npz"
    fi

    # Vanilla IDR — pairwise EM. Fast (no NUTS).
    echo "--- $REGIME / Vanilla IDR ---"
    "$PYIDR" sim \
        --regime "$REGIME" --K "$K" --n "$N" --reps "$REPS" \
        --alphas "$ALPHAS" --method vanilla-idr \
        --out "$OUT_ROOT/main/$REGIME/vanilla" --no-save-idr
    LATEST="$(find "$OUT_ROOT/main/$REGIME/vanilla" -maxdepth 1 -mindepth 1 -type d | sort | tail -n 1)"
    A_CSVS+=("$LATEST/results.csv")

    # MaxRank — non-parametric rank baseline. Instant.
    echo "--- $REGIME / MaxRank ---"
    "$PYIDR" sim \
        --regime "$REGIME" --K "$K" --n "$N" --reps "$REPS" \
        --alphas "$ALPHAS" --method maxrank \
        --out "$OUT_ROOT/main/$REGIME/maxrank" --no-save-idr
    LATEST="$(find "$OUT_ROOT/main/$REGIME/maxrank" -maxdepth 1 -mindepth 1 -type d | sort | tail -n 1)"
    A_CSVS+=("$LATEST/results.csv")
done

COMBINED_MAIN="$OUT_ROOT/combined_main.csv"
concat_csvs "$COMBINED_MAIN" "${A_CSVS[@]}"
echo
echo "Combined main CSV: $(wc -l < "$COMBINED_MAIN") lines"

echo
echo "============================================================"
echo "Section B — scaling sweep on S1 (F3 runtime, F5 contraction)"
echo "SCALING_N_LIST=\"$SCALING_N_LIST\" SCALING_REPS=$SCALING_REPS"
echo "============================================================"
B_CSVS=()
for SCALING_N in $SCALING_N_LIST; do
    SCALING_OUT="$OUT_ROOT/scaling/n${SCALING_N}"
    echo "--- S1 / n=$SCALING_N (em_init=${EM_INIT}) ---"
    "$PYIDR" sim \
        --regime S1 --K "$K" --n "$SCALING_N" --reps "$SCALING_REPS" \
        --alphas "$ALPHAS" --method py-idr \
        --num-chains "$NUM_CHAINS" --n-warmup "$N_WARMUP" \
        --n-samples "$N_SAMPLES" --nuts-warmup "$NUTS_WARMUP" \
        --out "$SCALING_OUT" --no-save-idr "$EM_INIT_FLAG"
    LATEST="$(find "$SCALING_OUT" -maxdepth 1 -mindepth 1 -type d | sort | tail -n 1)"
    B_CSVS+=("$LATEST/results.csv")
done

SCALING_CSV="$OUT_ROOT/scaling.csv"
concat_csvs "$SCALING_CSV" "${B_CSVS[@]}"
echo "Scaling CSV: $(wc -l < "$SCALING_CSV") lines"

echo
echo "============================================================"
echo "Figures + tables"
echo "============================================================"
FIGS="$OUT_ROOT/figures"
TABLES="$OUT_ROOT/tables"

# F1 — calibration across S1/S2/S4/S5; default regime list matches the report.
"$PYIDR" figure --fig f1 --results "$COMBINED_MAIN" --out "$FIGS/fig_f1.pdf"
# F2 — cluster count on S5 (T>1 chain path).
"$PYIDR" figure --fig f2 --results "$COMBINED_MAIN" --out "$FIGS/fig_f2.pdf"
# F3 — runtime on S1 (varied n).
"$PYIDR" figure --fig f3 --results "$SCALING_CSV" --regimes S1 --out "$FIGS/fig_f3.pdf"
# F4 — ROC/PR on S5 from the NPZ sidecar.
if [ -n "$S5_NPZ" ] && [ -f "$S5_NPZ" ]; then
    "$PYIDR" figure --fig f4 --results "$COMBINED_MAIN" \
        --idr-npz "$S5_NPZ" --regimes S5 --out "$FIGS/fig_f4.pdf"
else
    echo "warning: no S5 idr_traces.npz at $S5_NPZ — skipping F4" >&2
fi
# F5 — Hellinger contraction on S1.
"$PYIDR" figure --fig f5 --results "$SCALING_CSV" --out "$FIGS/fig_f5.pdf"

# T4 — sim-fdr and sim-power LaTeX fragments.
"$PYIDR" table --table t4-fdr --results "$COMBINED_MAIN" \
    --alpha 0.05 --out "$TABLES/tab_sim_fdr.tex"
"$PYIDR" table --table t4-power --results "$COMBINED_MAIN" \
    --alpha 0.05 --out "$TABLES/tab_sim_power.tex"

echo
echo "============================================================"
echo "Done. Artefacts:"
echo "  $FIGS/fig_f{1..5}.pdf"
echo "  $TABLES/tab_sim_{fdr,power}.tex"
echo "  $COMBINED_MAIN, $SCALING_CSV"
echo
echo "To overwrite the report's placeholder figures:"
echo "  cp $FIGS/fig_f1.pdf docs/report/figures/fig_idr_calibration.pdf"
echo "  cp $FIGS/fig_f2.pdf docs/report/figures/fig_py_vs_dp.pdf"
echo "  cp $FIGS/fig_f3.pdf docs/report/figures/fig_runtime.pdf"
echo "  cp $FIGS/fig_f4.pdf docs/report/figures/fig_roc_pr.pdf"
echo "  cp $FIGS/fig_f5.pdf docs/report/figures/fig_contraction.pdf"
echo "  cp $TABLES/tab_sim_fdr.tex   docs/report/tables/   # then merge with table_sim_results.tex"
echo "  cp $TABLES/tab_sim_power.tex docs/report/tables/"
echo "============================================================"
