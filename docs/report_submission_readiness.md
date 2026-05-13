# Submission-readiness audit

A meticulous walk-through of `docs/report/` against the repo (W8.11), itemising
what blocks a journal submission. Generated 2026-05-12 as the launch point for
the W9 submission push.

## TL;DR

| Section / artefact | State | Action |
|---|---|---|
| §0 Abstract | "figures and tables are illustrative or placeholder" | Drop placeholder framing; describe real simulation results |
| §1 Intro | Complete | — |
| §2 Related work | Complete | — |
| §3 Method | Complete | — |
| §4 Experiments | Every figure caption + every result table is a placeholder | **Repopulate from real sweep**, rewrite narrative, add insights |
| §5 Conclusion | Says "placeholder figures and tables" | Update to reflect what's now real |
| §6 Appendix | Complete (lemma stubs) | — |
| Figures F1–F5 (simulation) | PDFs are illustrative; reproducers exist (`py-idr figure --fig f{1..5}`) | Regenerate from the W9 sweep |
| Figure F6 (real-data) | Placeholder PDF; reproducer not implemented | **Blocked on plan 08 (data access)** |
| Tables T1–T3, T_notation, T_related_work, T_sim_design, T_datasets | Static, hand-managed; no XX placeholders | — |
| Table T4 (sim-fdr, sim-power) | All cells are XX.X | Regenerate from real sweep via `py-idr table` |
| Table T_sim_K (K-effect on S5) | All cells are XX.X | Reproducer not yet implemented; data is producible from a K-varied sweep |
| Tables T_real_disc, T_real_clusters, T_real_runtime | All cells are XX,XXX / XX h | **Blocked on plan 08 (data access)** |

## What ships in this session

1. **Real sweep.** `scripts/produce_report_figures.sh` extended for the
   submission-scale CPU run (K=2, n=100, reps=10 across S1–S5 + S5-sparse, three
   methods). Numbers will be honestly described as "below the §4.1 production
   spec" and reproduced from the script. The longer K∈{2,5,10,50} ×
   n∈{2k,10k,50k} sweep stays as a documented production target for the GPU job.

2. **Real F1–F5 PDFs.** Reproducers exist; run them on the new sweep CSV and
   copy into `docs/report/figures/`.

3. **Real T4 tables.** `py-idr table --table t4-fdr` and `--table t4-power`
   produce the LaTeX directly from the sweep CSV. Replace
   `docs/report/tables/table_sim_results.tex` with the rendered versions
   (preserving the placeholder rows for methods we don't have yet, marked
   "n/a").

4. **Prose rewrite.** §4 narrative converts from "expected to" / "planned" /
   "will be" to past-tense descriptions of what actually happened. Captions are
   aligned with the real figures. The abstract and conclusion drop the
   "placeholder material" framing where it no longer applies.

5. **§4.x insights subsection** added at the end of §4: a one-page distillation
   of what the simulation experiments demonstrate, organised by hypothesis
   (H1 FDR control, H2 PY-IDR vs Vanilla on heavy-tail, H3 PY-IDR vs MaxRank).
   Anchored on the W8.8 verdict ledger.

## What is blocked

- **§4.5 Real-data evaluation (REAL-01..05, F6, T_real_*).** Five datasets;
  every one requires data access from ENCODE 4, DREAM5, Tabula Muris, or
  Pan-UKBB. No fitting can happen without the upstream pipelines (plan 08).
  Action for the next phase: either (a) the user provides downloaded data, or
  (b) the user authorises Pan-UKBB DAC application. Until then the §4.5 prose
  + tables stay framed as a planned protocol.

- **"PY-IDR (VI)" rows in T4/T_real_*.** Plan 05's variational back-end is
  still on stubs. The MCMC row will be filled; the VI row will be marked
  `\textit{(VI not yet implemented; see §3.5.2)}` rather than XX.X.

- **Comparator rows in T4 for eCV, nestedIDR, ChIP-R, GMCM-AD, BNP-Archimedean.**
  Implementation availability and licensing checks are in handoff §8 Q1 and are
  not yet resolved. The four implemented methods are PY-IDR (MCMC), Vanilla IDR
  (pairwise), MaxRank — these are what the new T4 will show. Other rows are
  marked "not yet integrated".

- **K∈{5,10,50} scaling rows in T_sim_K.** The K=2 sweep gives us one row;
  larger K needs the GPU job. The table will be reduced to "K=2, K=5
  (small-n)" in the manuscript pass, with the K∈{10,50} columns documented
  as future work pending plan 14's GPU harness.

## Section-by-section punch list

### §0 Abstract
- [ ] Drop the trailing sentence ("its figures and result tables are
      illustrative or placeholder material..."). Replace with one sentence
      summarising the actual experimental scope shipped (5 simulation
      regimes, 3 implemented methods, K=2, n=100, 10 MC reps).
- [ ] Keep the planned-protocol framing for the real-data section.

### §3 Method
- [ ] Re-check the "EM-IDR initialization" sentence at line ~118: the
      implementation is W8.5's `compute_em_init`, which seeds (π, ρ, Z) from
      the pairwise Vanilla IDR EM. Confirm the description matches.
- [ ] Add a sentence under "Initialization and diagnostics" noting the
      W8.11 divergence plumbing — `num_divergent_transitions` rides on
      `ChainResult` now.

### §4 Experiments
- [ ] §4.4 caption for `fig:calibration`: rewrite to describe what's actually
      shown (S1/S2/S4/S5, K=2, n=100, 10 reps, methods present).
- [ ] §4.4 caption for `fig:roc-pr`: same; describe the S5 cell, n=100, K=2.
- [ ] §4.5 `tab:sim-fdr`, `tab:sim-power`: replace placeholder XX.X rows with
      real numbers for {Vanilla IDR, MaxRank, PY-IDR}; mark other methods as
      not-yet-integrated.
- [ ] §4.6 caption for `fig:contract`: describe the actual scaling sweep
      (S1, K=2, n ∈ {100, 200, 400, 800, 1600}, 20 reps).
- [ ] §4.7 caption for `fig:py-vs-dp`: describe what the cluster-count
      histogram on S5 actually shows.
- [ ] §4.9 caption for `fig:runtime`: describe the actual scaling cells.
- [ ] **New §4.x: Empirical insights** — distil the verdict-ledger
      (H1/H2/H3) outcomes plus 3–5 surprises from the numbers.
- [ ] §4.8 ablations: keep the planned framing; flag the W8.5 EM warm start
      and W8.11 divergence plumbing as completed.

### §5 Conclusion
- [ ] Update first paragraph: "we developed... and we report..." instead of
      "and we stated theoretical results with placeholder figures and tables".
- [ ] Limitations §: add (i) sweep scale is K=2, n=100 (below the §4.1 spec
      while a GPU run is pending); (ii) only three comparators integrated;
      (iii) real-data section is protocol-only.

### Repo-side completeness items
- [ ] `scripts/produce_report_figures.sh` already in tree (W8.12); confirm
      the submission-scale params route through.
- [ ] `py-idr table --table t4-fdr / t4-power` already wired; spot-check
      output against report's tabular shape.
- [ ] Add `scripts/produce_report_tables.sh` mirroring the figure script —
      a single invocation that renders T4 fragments into `docs/report/tables/`.
- [ ] Add `tests/regression/test_figures_smoke.py`: assert that the figure
      reproducers consume the sweep CSV shape produced by `py-idr sim` (the
      W8.12 smoke run is the existence proof).

## Acceptance criteria

A submission-ready manuscript means:

1. `latexmk docs/report/main.tex` rebuilds the PDF with all `\input` directives
   resolving and no `XX.X` placeholders in any non-real-data table.
2. Every figure caption describes what the figure actually shows (Monte Carlo
   means + 95% bands from the recorded `replicate_seed` schedule).
3. `tests/repro/test_verdict.py` plus the new W9 verdict run produce a
   verdict ledger that the manuscript cites in §4.x Empirical insights.
4. The abstract and conclusion describe what was done, not what is planned —
   except for the real-data section, which remains a documented protocol.
5. The "Code availability" paragraph in §5 points at the public repo + tagged
   release. The "Data availability" paragraph lists the accession IDs for each
   real dataset once REAL-01..05 land.

## Submission checklist (post-W9)

- [ ] Bibliography: every `\cite{*}` resolves against `refs.bib`; no
      undefined references at compile time.
- [ ] Author affiliation + email on title page.
- [ ] Acknowledgments + funding paragraph filled in (currently `[to be added]`).
- [ ] Conflict of interest paragraph (currently blank statement of no conflict).
- [ ] License (LaTeX class + repo + figures).
- [ ] arXiv-ready .tar.gz (`latexmkrc` + `*.tex` + `figures/*.pdf` + `tables/*.tex` + `refs.bib`).
