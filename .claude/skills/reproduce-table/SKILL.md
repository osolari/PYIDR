---
name: reproduce-table
description: Regenerate one of the paper tables T1–T7 from artifacts/*.csv (or YAML registries). Use when the user asks to "rebuild tab:sim-fdr", "refresh the discovery table", or "regenerate the notation table".
---

# Skill: Reproduce a Table

Tables follow the same CSV-to-LaTeX contract as figures. Some tables (T1, T2, T3, T6)
are sourced from YAML registries inside the package; the rest are from `artifacts/*.csv`.

## When to invoke

Triggered by any of:

- "regenerate tab:<name>"
- "rebuild table T<n>"
- "refresh the FDR table"
- "produce the related-work table"

## Inventory

| ID | LaTeX label | Source | Module |
|----|-------------|--------|--------|
| T1 | tab:notation | src/stick_idr/_notation.yaml | stick_idr.tables.t1_notation |
| T2 | tab:related-work | src/stick_idr/_comparators.yaml | stick_idr.tables.t2_related_work |
| T3 | tab:sim-design | enumerated from simulation/scenarios.py | stick_idr.tables.t3_sim_design |
| T4 | tab:sim-fdr / tab:sim-power | artifacts/sim/calibration_long.csv | stick_idr.tables.t4_sim_fdr_power |
| T5 | tab:sim-K | artifacts/sim/K_effect.csv | stick_idr.tables.t5_sim_K |
| T6 | tab:datasets | src/stick_idr/_datasets.yaml | stick_idr.tables.t6_datasets |
| T7 | tab:real-disc / tab:real-clusters / tab:real-runtime | artifacts/real/*.csv | stick_idr.tables.t7_real_results |

## Prerequisites

- For YAML-sourced tables: the YAML registry must be valid.
- For CSV-sourced tables: the CSV must exist and pass the schema validator.

## Procedure

1. **Validate the source:**
   - YAML: `python -m stick_idr.tables._validate <table-id>`
   - CSV: `python -m stick_idr.eval.validate_sim_csv <csv>`
2. **Render:**
   ```bash
   python -m stick_idr.tables.<module> \
     --in <source> \
     --out docs/report/tables/<filename>.tex
   ```
3. **Audit the output.** Open the `.tex` fragment and confirm column headers, alignments,
   and `\citep{...}` keys are intact.
4. **Sign the output.** Sidecar `manifest.json` next to the `.tex` records `git_sha`,
   `source_path`, `source_sha256`, `table_module`, and `timestamp`.
5. **Rebuild the paper:**
   ```bash
   pdflatex docs/report/main.tex
   ```
   from `docs/report/`; confirm no missing-`\input` errors.

## Expected outputs

- `docs/report/tables/<name>.tex`
- `docs/report/tables/<name>.manifest.json`

## Failure modes

- **Bib-key drift.** If the LaTeX `\citep{...}` key is no longer in `refs.bib`, the
  paper build will fail with `Citation undefined`. The renderer should consult
  `_comparators.yaml`'s `bib_key` field and warn before emitting an unknown key.
- **CSV missing rows.** A partially-completed simulation sweep produces a table with
  blank cells. Re-invoke `run-simulation` for the missing $(K, n)$ combinations.
- **YAML schema drift.** The YAML registries are validated against `_schema.py` (pydantic).
  Schema changes require a coordinated update to both the renderer and the registry.
