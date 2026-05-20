#!/usr/bin/env bash
# fetch_encode.sh — turn a raw ENCODE file-list dump into a per-experiment
# tree of narrowPeak BEDs suitable for the REAL-01 / REAL-02 loaders.
#
# Workflow:
#   1. ENCODE's portal lets you build a search and click "Download" — that
#      gives you a `files.txt` with (a) a metadata URL on the first line
#      and (b) per-file download URLs on every subsequent line.
#   2. This script reads `files.txt`, fetches the metadata TSV, filters to
#      `.bed.gz` (narrowPeak) entries on released analyses, groups by
#      Experiment Accession, drops experiments with fewer than `--min-reps`
#      replicates, and downloads only the BEDs we need into:
#
#          $OUT/<experiment_accession>/<replicate>_<file_accession>.bed.gz
#
#   3. The resulting tree is what `py_idr.data.loaders.load_encode_ctcf`
#      and `load_encode_atac` consume.
#
# Usage:
#   bash scripts/fetch_encode.sh \
#       --files-txt ~/Downloads/files.txt \
#       --out data/encode_ctcf \
#       --min-reps 2 \
#       [--max-experiments N]
#
# Defaults: --min-reps 2, --max-experiments 0 (no cap)
#
# Required: curl, awk, python3 (for grouping by experiment).
# Optional: --dry-run to just print the download plan.

set -euo pipefail

FILES_TXT=""
OUT=""
MIN_REPS=2
MAX_EXPS=0
DRY_RUN=0

while [ "$#" -gt 0 ]; do
    case "$1" in
        --files-txt) FILES_TXT="$2"; shift 2 ;;
        --out) OUT="$2"; shift 2 ;;
        --min-reps) MIN_REPS="$2"; shift 2 ;;
        --max-experiments) MAX_EXPS="$2"; shift 2 ;;
        --dry-run) DRY_RUN=1; shift ;;
        -h|--help)
            sed -n '/^# /p' "$0" | sed 's/^# //'
            exit 0
            ;;
        *) echo "unknown arg: $1" >&2; exit 2 ;;
    esac
done

if [ -z "$FILES_TXT" ] || [ -z "$OUT" ]; then
    echo "usage: $0 --files-txt <path> --out <dir> [--min-reps N] [--max-experiments N] [--dry-run]" >&2
    exit 2
fi
if [ ! -f "$FILES_TXT" ]; then
    echo "error: $FILES_TXT not found" >&2
    exit 1
fi

mkdir -p "$OUT"

# Line 1 is the metadata URL (wrapped in double-quotes). Strip them and fetch.
METADATA_URL=$(head -1 "$FILES_TXT" | tr -d '"')
echo ">>> Metadata URL: $METADATA_URL"

METADATA_TSV="$OUT/metadata.tsv"
if [ ! -f "$METADATA_TSV" ]; then
    echo ">>> Fetching metadata TSV → $METADATA_TSV"
    curl --silent --show-error --location --output "$METADATA_TSV" "$METADATA_URL"
fi
NROWS=$(wc -l < "$METADATA_TSV")
echo ">>> Metadata: $NROWS rows"

# Use Python to filter + group; awk would work but parsing TSV with quoted
# fields and the variable column layout is fragile.
GROUPED_CSV="$OUT/grouped.csv"
python3 - "$METADATA_TSV" "$GROUPED_CSV" "$MIN_REPS" "$MAX_EXPS" <<'PY'
import sys, csv
from collections import Counter

src, dst, min_reps, max_exps = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])

# ENCODE metadata TSV has a header line with mixed casing; we tolerate either.
with open(src) as f:
    reader = csv.DictReader(f, delimiter="\t")
    rows = list(reader)


def pick(row, *candidates):
    for c in candidates:
        if c in row and row[c]:
            return row[c]
    return ""


# Filter to .bed.gz file URLs.
bed_rows = [r for r in rows if pick(r, "File download URL", "href").endswith(".bed.gz")]

# Sanity-check the Output type distribution. If all rows are
# "IDR thresholded peaks" the user pulled the *consensus* calls, not the
# per-replicate raw calls. PY-IDR needs the latter; print a clear error
# rather than silently dropping everything in the replicate-count filter.
output_types = Counter(pick(r, "Output type", "output_type") for r in bed_rows)
print("output type distribution in .bed.gz subset:")
for ot, n in output_types.most_common():
    print(f"  {n:6d}  {ot!r}")

raw_types = {"peaks", "replicated peaks", "pseudoreplicated peaks", "narrowPeak"}
useful = [r for r in bed_rows if pick(r, "Output type") in raw_types]
if not useful:
    print()
    print("ERROR: no per-replicate raw peak calls in this download.")
    print("Your search URL likely included files.preferred_default=true, which")
    print("returns only the IDR-thresholded consensus call per experiment.")
    print("PY-IDR needs the raw per-replicate calls. Rebuild the file list with:")
    print()
    print("  https://www.encodeproject.org/metadata/"
          "?type=Experiment&assay_title=TF+ChIP-seq&target.label=CTCF"
          "&assembly=GRCh38&status=released"
          "&files.output_type=peaks&files.file_type=bed+narrowPeak")
    print()
    print("Then click Download on the ENCODE search page to get a fresh files.txt.")
    sys.exit(2)

print(f"useful (raw per-replicate) rows: {len(useful)} of {len(bed_rows)} .bed.gz")

# Extract one row per (experiment, biological replicate). Note: ENCODE
# encodes single biological replicates as "1", multi-replicate consensus
# as "1, 2". We keep only single-replicate entries.
def is_single_rep(s):
    return "," not in s and s.strip().isdigit()


keep = []
for r in useful:
    bio_rep = pick(r, "Biological replicate(s)", "biological_replicates")
    if not is_single_rep(bio_rep):
        continue
    keep.append({
        "experiment": pick(r, "Experiment accession"),
        "biorep": bio_rep.strip(),
        "file": pick(r, "File accession"),
        "url": pick(r, "File download URL"),
    })

# Group by experiment, drop experiments with < min_reps biological replicates.
by_exp = {}
for r in keep:
    by_exp.setdefault(r["experiment"], []).append(r)

filtered = {
    exp: files
    for exp, files in by_exp.items()
    if len({f["biorep"] for f in files}) >= min_reps
}
if max_exps > 0:
    filtered = dict(list(filtered.items())[:max_exps])

with open(dst, "w") as out:
    w = csv.writer(out)
    w.writerow(["experiment", "biorep", "file", "url"])
    for exp, files in sorted(filtered.items()):
        for f in files:
            w.writerow([f["experiment"], f["biorep"], f["file"], f["url"]])

n_exp = len(filtered)
n_files = sum(len(v) for v in filtered.values())
print(f"experiments with >= {min_reps} bioreps: {n_exp}; total BEDs to fetch: {n_files}")
PY

NFILES=$(tail -n +2 "$GROUPED_CSV" | wc -l)
echo ">>> Plan: $NFILES BEDs across $(tail -n +2 "$GROUPED_CSV" | cut -d, -f1 | sort -u | wc -l) experiments"

if [ "$DRY_RUN" -eq 1 ]; then
    echo ">>> Dry run; not downloading. See $GROUPED_CSV for the plan."
    exit 0
fi

# Download. xargs -P 8 parallel; -L 1 one URL per invocation.
echo ">>> Downloading (parallel × 8)..."
tail -n +2 "$GROUPED_CSV" | awk -F, '{print $1 "/" $2 "_" $3 ".bed.gz " $4}' \
    | while read -r dest url; do
        mkdir -p "$OUT/$(dirname "$dest")"
        if [ ! -f "$OUT/$dest" ]; then
            echo "$OUT/$dest $url"
        fi
    done | xargs -P 8 -L 1 sh -c 'curl --silent --show-error --location --output "$1" "$2"' _

echo ">>> Done. Tree under $OUT:"
find "$OUT" -name "*.bed.gz" | head -10
echo "..."
echo "Total $(find "$OUT" -name '*.bed.gz' | wc -l) BEDs."
