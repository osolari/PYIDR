#!/usr/bin/env bash
# Multi-method comparison demo — reproduces a real T4 row with three methods
# instead of placeholder cells.
#
# Walks through:
#   1. Run py-idr sim three times with different --method values, same regime/seeds.
#   2. Concatenate the three results.csv files into one long-format CSV.
#   3. Run py-idr table --table t4-fdr → real LaTeX table with three methods.
#
# This is the end-to-end pipeline a user runs to reproduce a comparator row in
# tab:sim-fdr. It is small / fast on purpose (K=2, n=80, 5 replicates) so the
# script finishes in a minute on CPU; for the report's headline numbers, scale
# up to the §4.1 default (K=5, n=10_000, 100 replicates).

set -euo pipefail

cd "$(git rev-parse --show-toplevel)"

OUT_ROOT=${1:-/tmp/pyidr_multi_method_demo}
mkdir -p "$OUT_ROOT"

JAX_PLATFORMS=${JAX_PLATFORMS:-cpu}
export JAX_PLATFORMS

PYIDR=".venv/bin/py-idr"

REGIME="${REGIME:-S1}"
K="${K:-2}"
N="${N:-80}"
REPS="${REPS:-5}"
ALPHAS="${ALPHAS:-0.01,0.05,0.10}"

echo "=== py-idr sim: PY-IDR (MCMC, T=1) ==="
$PYIDR sim --regime "$REGIME" --K "$K" --n "$N" --reps "$REPS" \
    --alphas "$ALPHAS" --method py-idr \
    --num-chains 2 --n-warmup 30 --n-samples 60 --nuts-warmup 15 \
    --out "$OUT_ROOT/pyidr"

echo "=== py-idr sim: Vanilla IDR (pairwise) ==="
$PYIDR sim --regime "$REGIME" --K "$K" --n "$N" --reps "$REPS" \
    --alphas "$ALPHAS" --method vanilla-idr \
    --out "$OUT_ROOT/vanilla"

echo "=== py-idr sim: MaxRank ==="
$PYIDR sim --regime "$REGIME" --K "$K" --n "$N" --reps "$REPS" \
    --alphas "$ALPHAS" --method maxrank \
    --out "$OUT_ROOT/maxrank"

# Concatenate the three sweep CSVs into one long-format CSV.
COMBINED="$OUT_ROOT/combined.csv"
{
    head -n 1 "$OUT_ROOT/pyidr"/*/results.csv
    tail -q -n +2 "$OUT_ROOT/pyidr"/*/results.csv
    tail -q -n +2 "$OUT_ROOT/vanilla"/*/results.csv
    tail -q -n +2 "$OUT_ROOT/maxrank"/*/results.csv
} > "$COMBINED"

echo
echo "=== Combined CSV row counts ==="
wc -l "$COMBINED"
echo

echo "=== Methods present ==="
awk -F, 'NR>1 { print $5 }' "$COMBINED" | sort -u
echo

echo "=== py-idr table: tab:sim-fdr at alpha=0.05 ==="
$PYIDR table --table t4-fdr --results "$COMBINED" \
    --alpha 0.05 --out "$OUT_ROOT/t4_fdr.tex"

echo
echo "Wrote LaTeX table to $OUT_ROOT/t4_fdr.tex"
echo "Sample rendering:"
echo "---"
cat "$OUT_ROOT/t4_fdr.tex"
echo "---"
