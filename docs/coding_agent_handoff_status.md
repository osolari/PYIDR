# Coding-Agent Handoff Status

**Source of truth:** [`docs/report/CODING_AGENT_HANDOFF.md`](report/CODING_AGENT_HANDOFF.md)
(handed off by the manuscript author).

This page mirrors the IMPL / SIM / REAL task list from the handoff and tracks status
in this repo. Update on every PR that closes a task. The `Plan` column points at the
sub-plan that owns the task; the `Status` column uses `TODO / PARTIAL / DONE`.

> Last refreshed by the alignment commit. The `git log` is authoritative for the
> latest state — this page is a courtesy index.

## Implementation tasks

| ID | Task | Plan | Status | Notes |
|----|------|------|--------|-------|
| IMPL-01 | Core data schema | [13](../plans/13_reproducibility.md), [08](../plans/08_real_data_pipelines.md) | TODO | `src/py_idr/data/schema.py` not yet implemented; pydantic `DatasetManifest` + `RunManifest` + `SeedLedger` planned. |
| IMPL-02 | Marginal preprocessing + Bernstein–Dirichlet | [03](../plans/03_marginals.md) | STUB | `src/py_idr/marginals/bernstein.py` exists as `NotImplementedError`. |
| IMPL-03 | Copula atom library | [02](../plans/02_algebra_and_copulas.md) | DONE | Independence, Gaussian, $t_5$, Clayton, Gumbel implemented and PLOD-tested at K=2,3,5; Gumbel multivariate density (K≥3) is the one open follow-on. |
| IMPL-04 | Valid correlation parameterization | [02](../plans/02_algebra_and_copulas.md) | PARTIAL | Cholesky/LKJ + Exchangeable correct; OneFactor/TwoFactor still on the **old parameterization** and need replacement with the PSD-then-normalize form (revised §3.2.2). PLOD positive-pairwise rejection helper not yet implemented. |
| IMPL-05 | PY Pólya-urn auxiliary-atom MCMC | [04](../plans/04_inference_mcmc.md) | STUB | `src/py_idr/pym/algo8.py` is a stub and is scheduled for rename to `polya_urn.py`. New predictives use $n_{m,-i}$ and $T_{-i}$. |
| IMPL-06 | NUTS / Metropolis atom updates | [04](../plans/04_inference_mcmc.md) | TODO | Will integrate via NumPyro NUTS on the unconstrained Cholesky parameterization with PLOD rejection at the proposal level. |
| IMPL-07 | Local idr + Bayes-mFDR + Sun–Cai | [06](../plans/06_decision_theory.md) | STUB | `decision/sun_cai.py` stub exists; `bayes_mfdr.py` and `local_idr.py` not yet present. Revised denominator is `max(1, E[R])`. |
| IMPL-08 | Structured finite-truncation VI | [05](../plans/05_inference_vi.md) | TODO | `src/py_idr/inference/vi/` only has `__init__.py`. Family per revised §3.3.2 uses per-cluster $q(V_m)$ Beta. |
| IMPL-09 | Diagnostics + logging | [12](../plans/12_diagnostics.md) | TODO | Plan 12 is new; `src/py_idr/eval/diagnostics_report.py` not yet implemented. |
| IMPL-10 | Reproducibility infrastructure | [13](../plans/13_reproducibility.md), [14](../plans/14_docker_a10dev.md) | TODO | `Dockerfile`, Snakemake, seed ledger, replay script all planned. |

## Simulation tasks

| ID | Task | Plan | Status | Notes |
|----|------|------|--------|-------|
| SIM-01 | Factorial S1–S5 + S5-sparse | [07](../plans/07_simulation_studies.md) | TODO | `simulation/scenarios.py` empty. S5-sparse added per revised Table 3. |
| SIM-02 | Component ablations | [07](../plans/07_simulation_studies.md) | TODO | Hydra config skeleton planned under `experiments/sim_<base>/ablations/`. |
| SIM-03 | Robustness / stress tests | [07](../plans/07_simulation_studies.md), [add-stress-test](../.claude/skills/add-stress-test/SKILL.md) | TODO | Catalogue per §4.6; one config per stressor in `experiments/sim_<base>/stress/`. |

## Real-data tasks

| ID | Task | Plan | Status | Notes |
|----|------|------|--------|-------|
| REAL-01 | ENCODE 4 CTCF ChIP-seq | [08](../plans/08_real_data_pipelines.md) | TODO | Loader stub only; pre-fit `DatasetManifest` not yet implemented. |
| REAL-02 | ENCODE 4 ATAC-seq | [08](../plans/08_real_data_pipelines.md) | TODO | Same as REAL-01. |
| REAL-03 | DREAM5 gene-network challenge | [08](../plans/08_real_data_pipelines.md) | TODO | $K = 35$ — high-K stress concomitant. |
| REAL-04 | Tabula Muris pseudobulk | [08](../plans/08_real_data_pipelines.md) | TODO | Pseudobulk aggregation pipeline + technology-stratified marginals. |
| REAL-05 | Pan-UKBB GWAS | [08](../plans/08_real_data_pipelines.md) | TODO | DAC compliance check needed before download; subset of variants TBD per handoff §8. |

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
