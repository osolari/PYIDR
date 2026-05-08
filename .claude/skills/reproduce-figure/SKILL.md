---
name: reproduce-figure
description: Regenerate one of the paper figures F1–F6 from artifacts/*.csv. Use when the user asks to "rebuild the calibration figure", "refresh fig:roc-pr", "redraw figure 3", or "regenerate the runtime plot".
---

# Skill: Reproduce a Figure

Each paper figure has a deterministic CSV-to-PDF pipeline. This skill enforces the
contract: figure scripts consume CSVs only — never raw data.

## When to invoke

Triggered by any of:

- "regenerate fig:<name>"
- "rebuild figure F<n>"
- "refresh the calibration plot"
- "produce the contraction curve PDF"

Do NOT use this skill if the underlying CSV is missing — instead invoke `run-simulation`
or the relevant data-collection skill first.

## Inventory

| ID | LaTeX label | Source CSV | Module |
|----|-------------|------------|--------|
| F1 | fig:calibration | artifacts/sim/calibration_long.csv | stick_idr.figures.f1_idr_calibration |
| F2 | fig:roc-pr | artifacts/sim/roc_pr_S5_K10.csv | stick_idr.figures.f2_roc_pr |
| F3 | fig:contract | artifacts/sim/contraction_S1.csv | stick_idr.figures.f3_contraction |
| F4 | fig:py-vs-dp | artifacts/sim/py_vs_dp_*.csv | stick_idr.figures.f4_py_vs_dp |
| F5 | fig:realdata | artifacts/real/*.csv | stick_idr.figures.f5_realdata |
| F6 | fig:runtime | artifacts/runtime/*.csv | stick_idr.figures.f6_runtime |

## Prerequisites

- `.venv` activated; `stick_idr.doctor()` is green.
- The required CSV exists. If not: tell the user which simulation/data-collection skill
  to run first.

## Procedure

1. **Validate the CSV schema:**
   ```bash
   python -m stick_idr.eval.validate_sim_csv <csv>
   ```
   This is paranoid by design — figure scripts are not robust to schema drift.
2. **Render:**
   ```bash
   python -m stick_idr.figures.<module> \
     --results <csv> \
     --out docs/report/figures/<filename>.pdf \
     --style-file src/stick_idr/figures/_style.py
   ```
3. **Inspect the output.** Open the PDF (or use `pdftoppm <pdf> /tmp/<name>` to render
   to PNG and present it to the user). Confirm legend, palette, and panel labels match
   the placeholder figure shape.
4. **Sign the output.** The script writes a sidecar `manifest.json` next to the PDF
   recording `git_sha`, `csv_path`, `csv_sha256`, `figure_module`, and `timestamp`.
5. **If the figure is part of a paper rebuild:**
   ```bash
   pdflatex docs/report/main.tex
   ```
   from the `docs/report/` directory; confirm no missing-figure errors.

## Expected outputs

- `docs/report/figures/<name>.pdf`
- `docs/report/figures/<name>.manifest.json`

## Failure modes

- **CSV schema mismatch.** The validator reports the offending columns. Either fix the
  data-collection script or update the figure module — never both at once.
- **Matplotlib version drift.** If the regression test for the figure fails with a
  pixel-hash mismatch, check the matplotlib version against `constraints.txt` and
  update the committed reference if intentional.
- **Empty CSV.** Means the upstream sim/data run did not produce rows. Re-invoke the
  upstream skill rather than producing an empty figure.
