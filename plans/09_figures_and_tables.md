# Plan 09 — Figures and Tables Reproducers

**Owner:** W3 (figures F1–F4, tables T3–T6) + W5 (F5, T7) + W4 (F6)
**Depends on:** 07_simulation_studies, 08_real_data_pipelines
**Implements:** Figures fig:calibration, fig:roc-pr, fig:contract, fig:py-vs-dp, fig:realdata, fig:runtime; Tables tab:notation, tab:related-work, tab:sim-design, tab:sim-fdr, tab:sim-power, tab:sim-K, tab:datasets, tab:real-disc, tab:real-clusters, tab:real-runtime

## Goal

Replace the synthetic figures and placeholder tables in `docs/report/figures/` and
`docs/report/tables/` with real outputs computed from package fits.

## Figure inventory

| ID | LaTeX label | Source CSV | Module |
|----|------------|------------|--------|
| F1 | fig:calibration | `artifacts/sim/calibration_long.csv` | `py_idr.figures.f1_idr_calibration` |
| F2 | fig:roc-pr | `artifacts/sim/roc_pr_S5_K10.csv` | `py_idr.figures.f2_roc_pr` |
| F3 | fig:contract | `artifacts/sim/contraction_S1.csv` | `py_idr.figures.f3_contraction` |
| F4 | fig:py-vs-dp | `artifacts/sim/py_vs_dp_clusters.csv` + `cluster_traces.csv` | `py_idr.figures.f4_py_vs_dp` |
| F5 | fig:realdata | `artifacts/real/discovery_counts.csv` + `posterior_band.csv` | `py_idr.figures.f5_realdata` |
| F6 | fig:runtime | `artifacts/runtime/per_iter.csv` + `ess_per_s.csv` | `py_idr.figures.f6_runtime` |

## Table inventory

| ID | LaTeX label | Source | Module |
|----|------------|--------|--------|
| T1 | tab:notation | `src/py_idr/_notation.yaml` (single source of truth) | `py_idr.tables.t1_notation` |
| T2 | tab:related-work | `src/py_idr/_comparators.yaml` | `py_idr.tables.t2_related_work` |
| T3 | tab:sim-design | enumerated from `simulation/scenarios.py` | `py_idr.tables.t3_sim_design` |
| T4 | tab:sim-fdr / tab:sim-power | `artifacts/sim/calibration_long.csv` | `py_idr.tables.t4_sim_fdr_power` |
| T5 | tab:sim-K | `artifacts/sim/K_effect.csv` | `py_idr.tables.t5_sim_K` |
| T6 | tab:datasets | `src/py_idr/_datasets.yaml` | `py_idr.tables.t6_datasets` |
| T7 | tab:real-disc / tab:real-clusters / tab:real-runtime | `artifacts/real/*.csv` | `py_idr.tables.t7_real_results` |

## Deliverables

### `src/py_idr/figures/`

Each module exposes `make(args)` and a Typer subcommand. Style is consolidated in
`_style.py` (matplotlib rcParams, palette matching the synthetic figure script).

The figure scripts MUST NOT do data collection — they consume CSVs only. The CSV is the
contract between the simulation/eval driver and the figure script.

### `src/py_idr/tables/`

Each module renders a single LaTeX table fragment with `booktabs` headers; the resulting
file lands under `docs/report/tables/<name>.tex` exactly where the paper's `\input` directives
expect it. The tables also emit a sidecar `manifest.json` with the source git SHA + dataset
version.

### `scripts/make_all_figures.sh` and `scripts/make_all_tables.sh`

See §9 of `IMPLEMENTATION_PLAN.md`. Each invocation enforces the convention that:
- `--results <csv>` is required.
- `--out <pdf|tex>` is required.
- `--style-file` defaults to `src/py_idr/figures/_style.py`.

## Tests

- `tests/regression/test_figure_f3_contraction_layout.py` — `f3_contraction.make` on a fixture CSV produces a PDF whose pixel hash matches a committed reference (within tolerance for matplotlib version drift).
- `tests/regression/test_table_t4_sim_fdr_layout.py` — produced LaTeX matches the committed fixture byte-for-byte (tabular content) modulo whitespace.
- `tests/property/test_csv_schema.py` — every figure script declares its expected CSV columns; the test loads every committed `artifacts/*.csv.example` fixture and validates it against the declaration.

## Acceptance criteria

- Running `bash scripts/make_all_figures.sh && bash scripts/make_all_tables.sh && pdflatex docs/report/main.tex` rebuilds the paper end-to-end.
- The legend, palette, and panel labels of every figure match the placeholder figures of `docs/report/figures/generate_figures.py` modulo the underlying data.
- Every figure and table writes a sidecar `manifest.json` so a reader can audit what data produced what panel.

## Notes

- The placeholder figures of `docs/report/figures/generate_figures.py` use a fixed palette (`#1b4d3e` for PY-IDR, `#9d4e0d` for Vanilla IDR, etc.). `_style.py` reproduces that palette so swap-in is seamless.
- `t1_notation.py` and `t6_datasets.py` are derived from YAML registries inside `src/py_idr/_notation.yaml` and `src/py_idr/_datasets.yaml` — the YAML is the single source of truth that both the Python code and the LaTeX paper draw from.
