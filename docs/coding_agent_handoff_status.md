# Coding-Agent Handoff Status

**Source of truth:** [`docs/report/CODING_AGENT_HANDOFF.md`](report/CODING_AGENT_HANDOFF.md)
(handed off by the manuscript author).

This page mirrors the IMPL / SIM / REAL task list from the handoff and tracks status
in this repo. Update on every PR that closes a task. The `Plan` column points at the
sub-plan that owns the task; the `Status` column uses `TODO / PARTIAL / DONE`.

> Last refreshed: W7.4 (F5 contraction figure). The `git log` is authoritative for
> the latest state — this page is a courtesy index.

## Implementation tasks

| ID | Task | Plan | Status | Notes |
|----|------|------|--------|-------|
| IMPL-01 | Core data schema | [13](../plans/13_reproducibility.md), [08](../plans/08_real_data_pipelines.md) | DONE | `py_idr.data.schema` provides `DatasetManifest`, `RunManifest`, `SeedLedger` (pydantic v2 with `extra='forbid'`); `py-idr sim` writes `run_manifest.json` per sweep with `git_sha` + SHA-256 config hash. |
| IMPL-02 | Marginal preprocessing + Bernstein–Dirichlet | [03](../plans/03_marginals.md) | DONE | `py_idr.marginals.{bernstein, dirichlet_weights, pilot, degree_prior, transform}` all implemented; the conjugate update is step 5 of Algorithm 1 and lives in the multi-cluster sweep. |
| IMPL-03 | Copula atom library | [02](../plans/02_algebra_and_copulas.md) | DONE | Independence, Gaussian, $t_5$, Clayton, Gumbel implemented and PLOD-tested at K=2,3,5; multivariate Gumbel ($K \ge 3$) is the one open follow-on (excluded from the type-update proposal pool today). |
| IMPL-04 | Valid correlation parameterization | [02](../plans/02_algebra_and_copulas.md) | DONE | Cholesky/LKJ + Exchangeable parameterizations correct; PLOD positive-pairwise predicates in `py_idr.algebra.plod`. The factor-structured variants are deferred since the multi-cluster path covers the report's experiments via LKJ on each cluster. |
| IMPL-05 | PY Pólya-urn auxiliary-atom MCMC | [04](../plans/04_inference_mcmc.md) | DONE | `py_idr.pym.polya_urn` predictive + `inference.mcmc.sweep_multi._polya_urn_reassign_all` implements step 2 of Algorithm 1 with $H$ auxiliary atoms and the $n_{m,-i}$ / $T_{-i}$ predictives. `py_idr.pym.eppf.log_eppf` + `py_hyperparam_update` close step 7. |
| IMPL-06 | NUTS / Metropolis atom updates | [04](../plans/04_inference_mcmc.md) | DONE | `py_idr.inference.mcmc.nuts_atoms` uses NumPyro NUTS on the Cholesky parameterisation for Gaussian / $t_5$ / Clayton / Gumbel-K2 atoms; `type_update.py` is the atomic-type IM step. `run_chain_simple` + `run_chain_multi` (plus their `run_multi_chain_*` siblings) are the top-level drivers. |
| IMPL-07 | Local idr + Bayes-mFDR + Sun–Cai | [06](../plans/06_decision_theory.md) | DONE | `decision.local_idr.from_mcmc`, `decision.sun_cai.step_up_threshold` (with `max(1, E[R])` denominator), and `decision.bayes_mfdr.bayes_mfdr_curve` are implemented and tested. ROC/PR helpers added in W7.2 (`decision.roc_pr`). |
| IMPL-08 | Structured finite-truncation VI | [05](../plans/05_inference_vi.md) | TODO | `py_idr.inference.vi` is the only inference back-end still on stubs. Plan 09 covers the design (mean-field normalising flow over the partition + atoms). |
| IMPL-09 | Diagnostics + logging | [12](../plans/12_diagnostics.md) | DONE | `py_idr.eval.diagnostics_report` packages arviz split-$\hat R$, ESS, divergence-fraction into a pass/warn/fail verdict with documented thresholds. |
| IMPL-10 | Reproducibility infrastructure | [13](../plans/13_reproducibility.md), [14](../plans/14_docker_a10dev.md) | PARTIAL | Pydantic manifests + per-run `run_manifest.json` from `py-idr sim` are landed. NPZ idr sidecar (`idr_traces.npz`) added in W7.3. Dockerfile + Snakemake + verdict-ledger are still planned (plan 14). |

## Simulation tasks

| ID | Task | Plan | Status | Notes |
|----|------|------|--------|-------|
| SIM-01 | Factorial S1–S5 + S5-sparse | [07](../plans/07_simulation_studies.md) | DONE | `py_idr.simulation.scenarios.simulate_S{1..5}` + `simulate_S5_sparse` all implemented. `run_replicate` dispatcher + `run_sweep` / `run_sweep_with_traces` + the `py-idr sim` CLI complete the loop. S5 routes through the multi-cluster chain by default; T=1 ablation available via `--chain-type T=1`. |
| SIM-02 | Component ablations | [07](../plans/07_simulation_studies.md) | PARTIAL | The `--chain-type` flag enables T=1 vs T>1 ablation on S5; per-component (PY vs DP, Gaussian-only vs heavy-tail, etc.) ablation sweeps are still on paper. |
| SIM-03 | Robustness / stress tests | [07](../plans/07_simulation_studies.md), [add-stress-test](../.claude/skills/add-stress-test/SKILL.md) | TODO | Catalogue per §4.6; one config per stressor planned. |

## Real-data tasks

| ID | Task | Plan | Status | Notes |
|----|------|------|--------|-------|
| REAL-01 | ENCODE 4 CTCF ChIP-seq | [08](../plans/08_real_data_pipelines.md) | TODO | Loader stub only; pre-fit `DatasetManifest` not yet implemented. |
| REAL-02 | ENCODE 4 ATAC-seq | [08](../plans/08_real_data_pipelines.md) | TODO | Same as REAL-01. |
| REAL-03 | DREAM5 gene-network challenge | [08](../plans/08_real_data_pipelines.md) | TODO | $K = 35$ — high-K stress concomitant. |
| REAL-04 | Tabula Muris pseudobulk | [08](../plans/08_real_data_pipelines.md) | TODO | Pseudobulk aggregation pipeline + technology-stratified marginals. |
| REAL-05 | Pan-UKBB GWAS | [08](../plans/08_real_data_pipelines.md) | TODO | DAC compliance check needed before download; subset of variants TBD per handoff §8. |

## Report figures & tables

| Artifact | Reference | CLI | Status | Notes |
|----------|-----------|-----|--------|-------|
| F1 | `fig:calibration` | `py-idr figure --fig f1` | DONE | Realised FDR vs nominal $\alpha$ per regime; reads sweep CSV. |
| F2 | `fig:py-vs-dp` | `py-idr figure --fig f2` | DONE | Posterior cluster-count histogram; reads `T_posterior_mean` from CSV. |
| F3 | `fig:runtime` | `py-idr figure --fig f3` | DONE | Walltime vs $n$ on log-log with fitted slope; reads `walltime_s` from CSV. |
| F4 | `fig:roc-pr` | `py-idr figure --fig f4` | DONE | ROC + PR curves with bootstrap CI; reads per-replicate `(idr, Z)` from the NPZ sidecar produced by `py-idr sim --save-idr`. |
| F5 | `fig:contract` | `py-idr figure --fig f5` | DONE | Hellinger contraction vs $n$ (closed-form for exchangeable Gaussian copulas via `py_idr.eval.hellinger`); reads `rho_true` / `rho_posterior_mean` from CSV. |
| F6 | `fig:realdata` | — | TODO | Blocked on plan 08 (REAL-01..05). |
| T4-fdr | `tab:sim-fdr` | `py-idr table --table t4-fdr` | DONE | LaTeX `tabular` with per-method × per-regime mean (SE) at $\alpha=0.05$. Placeholder rows for absent competitor methods. |
| T4-power | `tab:sim-power` | `py-idr table --table t4-power` | DONE | Same shape with `metric=power`. |
| T1..T3, T5..T7 | | — | TODO | T1/T2/T3 are static reference tables; T5/T6/T7 need plan 08 data. |

## Theory items

The revised report distinguishes completed arguments, conditional statements, and proof
obligations. The implementation tracks both the testable claims and the documented
gaps:

| Theorem | Status | Test (when implementable) | Notes |
|---|---|---|---|
| Thm 3.1 finite-sieve identifiability | provable on the finite sieve | `tests/identifiability/test_finite_sieve_id.py` | Requires PLOD + linear-independence pre-checks. |
| Thm 3.2 conditional Hellinger contraction | conditional on Assumptions 3.1–3.4 | `tests/numerical/test_contraction_simulation.py` | Boundary-singular tail-dependent atoms are out of scope per Remark 3.6. |
| Thm 3.3 Bayes-mFDR ($\alpha + O(\delta_n)$) | conditional on Assumption 3.5 | `tests/decision/test_sun_cai_calibration.py`, `test_mfdr_rate.py` | $\delta_n$ is exposed via `decision.local_idr.local_idr_error_scale`. |
| Thm 3.4 Harris (+ conditional geometric) | Harris provable; geometric is a separate drift+small-set obligation | `tests/inference/test_chain_irreducibility.py`, `test_ess_growth.py`; `test_geometric_drift_skip.py` | Geometric rate not asserted. |
| Thm 3.5 finite-truncation VI consistency | provable at fixed $T$ | `tests/integration/test_vi_matches_mcmc_means.py` | Matching-rate as $T \to \infty$ is future work. |
| Lemma A.3 PY truncation residual | polynomial decay only for $\sigma > 0$ | `tests/numerical/test_py_truncation.py` | Exponential sieve-complement bound is **Assumption 3.4**, not derived. |

## Open questions (from handoff §8)

1. Which exact comparator implementations are available for eCV, nestedIDR, MaRR,
   ChIP-R, GMCM-AD, BNP-Archimedean, and what are their default settings? — *open*
2. Should the irreproducible component remain exactly independent in all experiments,
   or should weak-dependence sensitivity be part of the main table? — *defaulting to
   "yes, independent in main; weak-dependence in stress"*
3. Which correlation parameterization gives the best balance of valid PLOD support,
   numerical stability, and interpretability? — *open; PSD-then-normalize is the
   default, with PLOD rejection on the loadings*
4. How should ties and missing replicate scores be handled in each real dataset?
   — *per-dataset; the `DatasetManifest` records the chosen policy*
5. Which subset of Pan-UKBB phenotypes / variants is feasible for the first run?
   — *open; recommended a small EUR-vs-CSA pair before full $K = 6$*
6. How should the companion paper distinguish methodology-validation simulations from
   publication-ready benchmark results? — *open; verdict ledger separates them*
7. Can the positive-discount PY sieve-tail condition be proved for the selected
   computational truncation, or should the final paper state it as an assumption?
   — *current default: state as Assumption 3.4*
8. What threshold defines acceptable convergence for cluster-specific parameters that
   are label-switching invariant only after postprocessing? — *use the deterministic
   Stephens (2000) relabeller; verdict ledger records label-switching events*
9. Should heavy-tail atoms remain in the main model if boundary-singular theory is not
   completed? — *yes, per the explicit decision in §3.2.2; framed as methodology with
   contraction theorem out-of-scope per Remark 3.6*

## Maintenance

When a task moves to `DONE`:

1. Update the row above (status + a one-line note linking to the closing PR).
2. If the task was a STUB or PARTIAL, also update the relevant sub-plan's
   "Acceptance criteria" checklist.
3. If the change closes one of the §8 open questions, update the answer here.
