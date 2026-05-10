# CODING_AGENT_HANDOFF - PY-IDR LaTeX Project

## 1. Project Overview

The paper proposes PY-IDR, a hierarchical Bayesian nonparametric extension of the Irreproducible Discovery Rate framework for multi-replicate reproducibility analysis. The proposed method models the reproducible-component copula as a Pitman--Yor mixture of structured Gaussian copulas with planned tail-dependence atoms, uses replicate-specific Bernstein--Dirichlet marginals, and reports posterior local idr and Bayes-mFDR step-up decisions.

The main theoretical components are finite-sieve identifiability under PLOD and linear-independence/separation assumptions, conditional Hellinger contraction under KL-support and sieve-tail assumptions, threshold-regularity-based Bayes-mFDR control, Harris ergodicity of the finite-sieve sampler, and finite-truncation variational consistency under approximation assumptions. The manuscript now distinguishes completed arguments, proof sketches, explicit assumptions, conjectural/geometric-rate claims, and future proof obligations.

The empirical portion is protocol-first. No experiments have been completed in this manuscript. Figures are synthetic illustrative placeholders, and tables contain placeholder cells that must not be treated as observed results. The next agent should implement the model and run the planned simulations, ablations, real-data preprocessing, diagnostics, and figure/table generation.

## 2. Build Instructions

- Root LaTeX file: `main.tex`
- Document class: `saim.cls`
- Shared macros: `math_commands.tex`
- Bibliography file: `refs.bib`
- Bibliography style: `plainnat`
- Bibliography engine: BibTeX
- Figure directory: `figures/`
- Table directory: `tables/`
- Build command verified in this container:

```bash
pdflatex -interaction=nonstopmode -halt-on-error main.tex
/usr/bin/bibtex.original main
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
```

- Standard Overleaf/local command, if `bibtex` is correctly installed:

```bash
pdflatex main
bibtex main
pdflatex main
pdflatex main
```

- Known build caveat: the container used for this editing pass had a broken system `bibtex` symlink, so `/usr/bin/bibtex.original` was used explicitly. This is an environment issue, not a LaTeX-source issue.
- Revised build status: successful.
- Revised PDF page count: 32.
- Original PDF page count: 31.
- Unresolved references: none detected after final build.
- Unresolved citations: none detected after final build.
- Duplicate labels: none detected.
- Known warnings: no unresolved-reference/citation or overfull-box warnings remained after the final revised build. Routine underfull-box output may occur depending on TeX engine and line-breaking.
- Figure regeneration:

```bash
python figures/generate_figures.py
```

This script writes synthetic placeholder PDFs to the project-local `figures/` directory.

## 3. Implementation Tasks

### IMPL-01: Core data schema

- Description: Define a canonical input schema for raw replicate scores, feature identifiers, replicate metadata, and optional truth labels for simulations.
- Related manuscript section: Sections 3.2 and 4.1.
- Expected input: CSV/Parquet tables with feature ID, replicate ID, score, optional condition/source labels, and optional truth metadata.
- Expected output: Validated in-memory data object with `X[n, K]`, feature metadata, replicate metadata, missingness/tie summary, and frozen feature universe.
- Dependencies: Data manifests from REAL-01 and simulation configs from SIM-01.
- Priority: High.
- Completion criteria: Unit tests cover missing scores, tied ranks, duplicated features, replicate-order changes, and feature-universe filtering.

### IMPL-02: Marginal preprocessing and Bernstein--Dirichlet marginals

- Description: Implement pilot CDFs, empirical scaled ranks for initialization, Bernstein basis matrices, latent basis indicators, and Dirichlet updates.
- Related manuscript section: Section 3.2.3 and Algorithm 1.
- Expected input: `X[n, K]`, degree grid, pilot-CDF options.
- Expected output: `U[n, K]`, basis matrices, sampled/estimated Bernstein weights, marginal diagnostics.
- Dependencies: IMPL-01.
- Priority: High.
- Completion criteria: Probability-integral-transform diagnostics are uniform under simulated known marginals; Dirichlet update matches conjugate counts.

### IMPL-03: Copula atom library

- Description: Implement Gaussian, fixed-df t, Clayton, and Gumbel copula density/evaluation routines, including parameter-domain checks and PLOD checks.
- Related manuscript section: Section 3.2.2.
- Expected input: Uniformized data, atom type, atom parameters.
- Expected output: Stable log copula density values and copula diagnostics.
- Dependencies: Numerical libraries for multivariate normal/t distributions and Archimedean copulas.
- Priority: High.
- Completion criteria: Densities integrate approximately to one in low-dimensional Monte Carlo checks; independence-boundary behavior is tested; invalid parameters are rejected.

### IMPL-04: Valid correlation parameterization

- Description: Implement exchangeable, one-factor, two-factor, and optional LKJ/unstructured correlation parameterizations that preserve unit diagonals and positive definiteness.
- Related manuscript section: Section 3.2.2 and Algorithm 1.
- Expected input: Unconstrained parameter vectors.
- Expected output: Correlation matrix `R`, Jacobian terms if needed, and PLOD/positive-correlation status.
- Dependencies: IMPL-03.
- Priority: High.
- Completion criteria: Unit tests confirm symmetry, unit diagonal, positive definiteness, and positive off-diagonal constraints where PLOD is required.

### IMPL-05: PY Polya-urn auxiliary-atom MCMC

- Description: Implement class assignment, PY occupied/new cluster reassignment, atom updates, atomic-type updates, marginal updates, hyperparameter updates, and mixing-proportion updates.
- Related manuscript section: Algorithm 1 and Eqs. (3.5)--(3.6) in the revised manuscript.
- Expected input: Initialized model state and uniformized data.
- Expected output: Posterior samples for classes, clusters, atom parameters, marginals, hyperparameters, local idr, and Bayes-mFDR decisions.
- Dependencies: IMPL-02 through IMPL-04.
- Priority: High.
- Completion criteria: Recovers known two-component/one-cluster simulations; cluster reassignment probabilities match analytic tests; posterior samples pass invariance and trace diagnostics on small cases.

### IMPL-06: NUTS/Metropolis atom updates

- Description: Implement continuous atom-parameter updates using NUTS where gradients are available and Metropolis alternatives where gradients are not reliable.
- Related manuscript section: Section 3.3.1.
- Expected input: Cluster-specific uniformized data and current atom parameters.
- Expected output: Updated atom parameters and diagnostics including acceptance rate and divergences.
- Dependencies: IMPL-03 and IMPL-04.
- Priority: Medium-high.
- Completion criteria: Acceptance/divergence diagnostics are recorded; gradient checks pass for Gaussian/t atoms; fallback random-walk updates work for Clayton/Gumbel.

### IMPL-07: Local idr and Bayes-mFDR decision module

- Description: Compute posterior local idr, sorted running averages, Bayes-mFDR estimates, rejection sets, and threshold diagnostics from MCMC or VI output.
- Related manuscript section: Section 3.3.3.
- Expected input: Posterior samples or variational posterior probabilities.
- Expected output: `idr_hat`, sorted decision table, selected `k_alpha`, rejection set, and calibration diagnostics.
- Dependencies: IMPL-05 or IMPL-08.
- Priority: High.
- Completion criteria: Handles empty rejection sets; matches analytic toy examples; reports denominator and threshold stability.

### IMPL-08: Structured finite-truncation VI

- Description: Implement a finite-truncation structured mean-field approximation with categorical class/cluster factors, Beta stick factors, atom-parameter factors, and Dirichlet marginal factors.
- Related manuscript section: Section 3.3.2 and Theorem 3.5.
- Expected input: Data object, truncation level, minibatch size, optimizer settings.
- Expected output: ELBO trace, variational local idr, variational cluster summaries, diagnostics.
- Dependencies: IMPL-02 through IMPL-04.
- Priority: Medium.
- Completion criteria: ELBO increases on deterministic small cases; VI approximates MCMC decisions on small simulations; truncation sensitivity is recorded.

### IMPL-09: Diagnostics and logging

- Description: Implement chain-level diagnostics, seed logging, R-hat/ESS, cluster traces, posterior predictive checks, marginal calibration, and failure-case reporting.
- Related manuscript section: Sections 3.3.1, 4.3, and Appendix Remark A.6.
- Expected input: Model output and data manifests.
- Expected output: Diagnostics JSON/CSV, plots, warnings, and pass/fail report.
- Dependencies: IMPL-05, IMPL-07, IMPL-08.
- Priority: High.
- Completion criteria: Every simulation and real-data run produces a machine-readable diagnostic report.

### IMPL-10: Reproducibility infrastructure

- Description: Create a workflow target or containerized pipeline that runs simulations, real-data preprocessing, model fitting, and table/figure generation.
- Related manuscript section: Section 4.9.
- Expected input: Config files, environment specification, data manifests.
- Expected output: Reproducible outputs in a fixed directory schema.
- Dependencies: All implementation tasks.
- Priority: High.
- Completion criteria: A clean checkout can reproduce one smoke-test simulation and regenerate manuscript tables/figures from outputs.

## 4. Experiment Plan

### SIM-01: Factorial simulation suite S1--S5

- Setups: S1 strong Gaussian, S2 weak Gaussian, S3 sparse Gaussian, S4 heterogeneous/skewed marginals, S5 heavy-tail copula mixture, plus S5-sparse for ROC/PR.
- Grid: `K in {2,5,10,50}`, `n in {2000,10000,50000}` for primary regimes; 100 Monte Carlo replications per cell.
- Baselines: vanilla IDR pairwise aggregation, eCV, nestedIDR if available, MaRR, ChIP-R where applicable, GMCM-AD, BNP-Archimedean where feasible, PY-IDR MCMC, PY-IDR VI.
- Metrics: realized FDR, power, AUROC, AUPRC, Hellinger error, local-idr calibration, posterior cluster count, ESS/sec, runtime, R-hat, posterior predictive diagnostics.
- Success criteria: Planned method should be evaluated for calibration, not assumed superior; all claims must be generated from completed runs and uncertainty summaries.
- Reproducibility requirements: seed ledger, config hash, truth-object storage, software versions, hardware metadata.

### SIM-02: Component ablations

- Ablations: PY vs DP; Gaussian-only vs heavy-tail atoms; empirical-rank approximation vs learned marginals; exchangeable vs factor-structured correlations; MCMC vs VI; pairwise aggregation vs joint K-replicate modeling; auxiliary-atom count.
- Expected output: Ablation table and plots with uncertainty bands.
- Success criteria: Identify which model components materially affect calibration, power, runtime, and cluster recovery.

### SIM-03: Robustness/stress tests

- Stressors: weakly dependent irreproducible component, high tie rate, missing replicates, near-singular correlation structures, large K, marginal misspecification, low reproducible mass, heavy boundary tail dependence.
- Expected output: Failure-case report, robustness plots, and guidance for practical use.
- Success criteria: Document when PY-IDR fails or becomes unstable; do not suppress adverse outcomes.

### REAL-01: ENCODE 4 CTCF ChIP-seq

- Inputs: ENCODE accessions, peak calls, replicate metadata, peak scores.
- Outputs: local idr per peak, reproducible peak counts, active-cluster summaries, runtime diagnostics.
- Dependencies: data accession manifest and peak-universe harmonization.

### REAL-02: ENCODE 4 ATAC-seq

- Inputs: ATAC-seq peak/accessibility scores and replicate metadata.
- Outputs: local idr, discovery counts, heavy-tail diagnostics, baseline comparisons.
- Dependencies: consistent broad-peak feature universe and QC policy.

### REAL-03: DREAM5 gene-network challenge

- Inputs: ranked algorithm predictions and gold-standard network labels where available.
- Outputs: reproducible edge sets, local idr, overlap with gold standards, cluster summaries.
- Dependencies: algorithm-rank harmonization and edge-universe definition.

### REAL-04: Tabula Muris pseudobulk

- Inputs: count matrices, tissue labels, technology labels, pseudobulk aggregation config.
- Outputs: reproducible gene signatures and technology/marginal diagnostics.
- Dependencies: cell filtering, tissue-level aggregation, score generation.

### REAL-05: Pan-UKBB GWAS

- Inputs: ancestry-specific GWAS summary statistics, phenotype selection, QC metrics.
- Outputs: reproducible variant sets across ancestry strata, local idr, sensitivity to ancestry missingness/sample imbalance.
- Dependencies: data-use compliance, variant harmonization, allele alignment, phenotype QC.

## 5. Figures and Tables to Generate

- `fig:calibration` | Section 4.4 | Placeholder: yes | Expected content: realized FDR vs nominal IDR by regime and method | Requires SIM-01 outputs.
- `fig:roc-pr` | Section 4.4 | Placeholder: yes | Expected content: ROC and precision-recall curves for S5-sparse | Requires SIM-01/S5-sparse outputs.
- `fig:contract` | Section 4.5 | Placeholder: yes | Expected content: empirical Hellinger error vs sample size and K | Requires true copula density and posterior mean density estimates.
- `fig:py-vs-dp` | Section 4.6 | Placeholder: yes | Expected content: cluster occupancy and cluster-count traces under PY vs DP | Requires SIM-02 ablation outputs.
- `fig:realdata` | Section 4.8 | Placeholder: yes | Expected content: real-data discovery counts and local-idr uncertainty summaries | Requires REAL-01 through REAL-05.
- `fig:runtime` | Section 4.9 | Placeholder: yes | Expected content: measured runtime, ESS/sec, and scaling curves | Requires runtime logs from SIM/REAL tasks.
- `tab:sim-fdr` | Section 4.4 | Placeholder: yes | Expected content: realized FDR mean and Monte Carlo SE by method/regime.
- `tab:sim-power` | Section 4.4 | Placeholder: yes | Expected content: power mean and Monte Carlo SE by method/regime.
- `tab:sim-K` | Section 4.4 | Placeholder: yes | Expected content: power/FDR by replicate count for S5.
- `tab:real-disc` | Section 4.8 | Placeholder: yes | Expected content: reproducible feature counts per real dataset and method.
- `tab:real-clusters` | Section 4.8 | Placeholder: yes | Expected content: posterior median and credible interval for active clusters.
- `tab:real-runtime` | Section 4.8 | Placeholder: yes | Expected content: wall-clock and ESS/sec measurements with hardware metadata.

## 6. Projected or Expected Results

The manuscript contains expected qualitative behavior and synthetic placeholder values. These must not be treated as observed results. Replace expected/projected language only after experiments are completed, outputs are validated, and uncertainty summaries are generated.

Replacement protocol:

1. Run the relevant simulation or real-data workflow from a frozen config.
2. Verify diagnostics and convergence/failure reports.
3. Generate table-ready CSV files with means, Monte Carlo SEs, intervals, and sample sizes.
4. Regenerate figures from result files, not from hard-coded synthetic values.
5. Replace placeholder table cells and captions only where actual outputs exist.
6. Change ``expected''/``planned'' wording to observed-result wording only for completed analyses.
7. Record every replacement in a new changelog.

Items currently expected/projected include calibration improvements, ROC/PR ordering, empirical contraction behavior, PY-vs-DP cluster behavior, real-data discovery counts, active-cluster summaries, runtime scaling, and practical recommendations about heavy-tail atoms and MCMC/VI tradeoffs.

## 7. Theory-to-Code Connections

- PLOD and separation assumptions: implement atom-domain checks, positive-correlation checks, and independence-boundary rejection. Stress-test with weak dependence and boundary-adjacent atoms.
- Marginal/coupla factorization: implement rank initialization plus learned Bernstein--Dirichlet marginals; diagnose PIT/rank uniformity and marginal calibration.
- Finite-sieve identifiability: in simulations, store true clusters and atom parameters; evaluate cluster recovery using label-invariant summaries.
- Conditional contraction: compute empirical Hellinger or integrated squared error where the true copula density is known; report dimension and sample-size dependence.
- PY tail/proof obligation: compare PY and DP cluster-size distributions; record active-cluster counts and residual/truncation diagnostics.
- Bayes-mFDR theorem: compute realized FDR under known truth and compare to posterior Bayes-mFDR threshold decisions.
- Harris/geometric ergodicity: report trace plots, ESS, R-hat, autocorrelation, cluster-switching diagnostics, and failures.
- VI theorem: compare VI to MCMC on small simulations; run truncation sensitivity and ELBO diagnostics.

## 8. Open Technical Questions

- What exact repository/package implementations are available for eCV, nestedIDR, MaRR, ChIP-R, GMCM-AD, and BNP-Archimedean, and what are their default settings?
- Should the irreproducible component remain exactly independent in all experiments, or should weak-dependence sensitivity be part of the main table?
- Which correlation parameterization gives the best balance of valid PLOD support, numerical stability, and interpretability?
- How should ties and missing replicate scores be handled in each real dataset?
- What subset of Pan-UKBB phenotypes and variants is computationally feasible for the first real-data run?
- How should the companion paper distinguish methodology-validation simulations from publication-ready benchmark results?
- Can the positive-discount PY sieve-tail condition be proved for the selected computational truncation, or should the final paper state it as an assumption?
- What threshold should define acceptable convergence for cluster-specific parameters that are label-switching invariant only after postprocessing?
- Should heavy-tail atoms remain in the main model if boundary-singular theory is not completed, or should they be framed exclusively as a planned robustness extension?

## 9. Files Changed or Added

Changed source files:

- `main.tex`
- `math_commands.tex`
- `README.md`
- `refs.bib`
- `sections/0-abstract.tex`
- `sections/1.intro.tex`
- `sections/2.related_work.tex`
- `sections/3.method.tex`
- `sections/4.experiments.tex`
- `sections/5.conclusion.tex`
- `sections/6.appendix.tex`
- `tables/table_notation.tex`
- `tables/table_related_work.tex`
- `tables/table_sim_design.tex`
- `tables/table_sim_results.tex`
- `tables/table_datasets.tex`
- `tables/table_real_results.tex`
- `figures/generate_figures.py`
- `figures/fig_idr_calibration.pdf`
- `figures/fig_roc_pr.pdf`
- `figures/fig_contraction.pdf`
- `figures/fig_py_vs_dp.pdf`
- `figures/fig_realdata.pdf`
- `figures/fig_runtime.pdf`

Added/replaced deliverable files:

- `CHANGELOG.md`
- `CODING_AGENT_HANDOFF.md`

Bibliography entries updated:

- `Iakovidis2024eCV`
- `PanNietoBarajasCraiu2024`
- `Karczewski2024PanUKBB`

No new bibliography keys were added.

## 10. Do-Not-Change Constraints

Do not remove or silently replace the following without explicit author approval:

- Planned S1--S5 simulations and S5-sparse stress test.
- Planned real-data analyses for ENCODE CTCF, ENCODE ATAC-seq, DREAM5, Tabula Muris, and Pan-UKBB.
- Placeholder figures and placeholder result tables before actual outputs exist.
- Projected/expected-result language before experiments are run.
- Theoretical direction: PY copula mixture, PLOD/separation, Bernstein--Dirichlet marginals, Bayes-mFDR decision rule.
- Methodology direction: PY auxiliary-atom MCMC, finite-truncation VI, structured Gaussian copula atoms, planned heavy-tail atoms.
- Implementation plans and reproducibility infrastructure requirements.
- Existing project layout, document class `saim.cls`, bibliography style `plainnat`, and root file `main.tex` unless the author approves structural changes.
- Citation metadata unless verified from primary or authoritative sources.
- Any wording that would convert planned experiments, synthetic figures, placeholder tables, expected behavior, or projected results into completed empirical findings before the experiments are actually run.
