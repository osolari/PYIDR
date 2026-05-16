#!/usr/bin/env bash
# build_arxiv_tarball.sh — emit a self-contained arXiv-ready tarball.
#
# Bundle the minimal LaTeX project from docs/report/ into a
# .tar.gz that arXiv can compile in its 'pdflatex' workflow. Excludes:
#
#   - assets/ except for what's actually referenced (currently
#     only verdicts_submission.json which is data, not a LaTeX asset)
#   - figures/generate_figures.py (build script, not output)
#   - CHANGELOG.md, CODING_AGENT_HANDOFF.md, README.md (development docs)
#   - latexmkrc (arXiv has its own latexmk policy)
#
# Usage:
#   bash scripts/build_arxiv_tarball.sh [OUT_PATH]
#
# Default OUT_PATH: artifacts/pyidr-arxiv-<gitsha>.tar.gz

set -euo pipefail

cd "$(git rev-parse --show-toplevel)"

GIT_SHA="$(git rev-parse --short=8 HEAD)"
DEFAULT_OUT="artifacts/pyidr-arxiv-${GIT_SHA}.tar.gz"
OUT_PATH="${1:-$DEFAULT_OUT}"
mkdir -p "$(dirname "$OUT_PATH")"

# Build a staging dir so the tarball has a clean top-level "pyidr/" root.
STAGE="$(mktemp -d)"
trap 'rm -rf "$STAGE"' EXIT

ROOT="$STAGE/pyidr"
mkdir -p "$ROOT"

# Required LaTeX sources.
cp docs/report/main.tex "$ROOT/"
cp docs/report/math_commands.tex "$ROOT/"
cp docs/report/saim.cls "$ROOT/"
cp docs/report/refs.bib "$ROOT/"

# Sections + tables (preserve subdir layout used by main.tex's \input).
mkdir -p "$ROOT/sections" "$ROOT/tables"
cp docs/report/sections/*.tex "$ROOT/sections/"
cp docs/report/tables/*.tex "$ROOT/tables/"

# Figures — PDFs only, no generator script.
mkdir -p "$ROOT/figures"
cp docs/report/figures/*.pdf "$ROOT/figures/"

# Assets — copy any non-script assets that the LaTeX actually references.
mkdir -p "$ROOT/assets"
if [ -f docs/report/assets/saim_logo.png ]; then
    cp docs/report/assets/saim_logo.png "$ROOT/assets/"
fi
# verdicts_submission.json is documentation, not a LaTeX source — don't ship.

# Top-level build hint for arXiv (00README.XXX is the canonical name).
cat > "$ROOT/00README.XXX" <<EOF
% arXiv build hints for PY-IDR
%
% Build command: pdflatex main; bibtex main; pdflatex main; pdflatex main
%
% Requires: TeXLive 2025 or newer (for tcolorbox, nicematrix versions used
% by saim.cls). arXiv's default TeXLive (2024+) is sufficient.
%
% Source bundled at git sha: ${GIT_SHA}
EOF

# Make the tarball.
( cd "$STAGE" && tar -czf "$(cd "$OLDPWD" && pwd)/$OUT_PATH" pyidr/ )

echo "Wrote $OUT_PATH"
echo "  size: $(du -h "$OUT_PATH" | cut -f1)"
echo "  contents:"
tar -tzf "$OUT_PATH" | head -25
N=$(tar -tzf "$OUT_PATH" | wc -l)
echo "  total: $N entries"
