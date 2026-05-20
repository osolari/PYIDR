# Coding-Agent Handoff Status

**Source of truth:** [`docs/report/CODING_AGENT_HANDOFF.md`](report/CODING_AGENT_HANDOFF.md)
(handed off by the manuscript author).

This page mirrors the IMPL / SIM / REAL task list from the handoff and tracks status
in this repo. Update on every PR that closes a task. The `Plan` column points at the
sub-plan that owns the task; the `Status` column uses `TODO / PARTIAL / DONE`.

> Last refreshed: W10.10 (W9 submission push + W10 polish/scaffolding shipped;
> plans/04b and plans/05b laid out for W11/W12). The `git log` is authoritative
> for the latest state — this page is a courtesy index.

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
| IMPL-08 | Structured finite-truncation VI | [05](../plans/05_inference_vi.md), [05b](../plans/05b_inference_vi_design.md) | TODO | `py_idr.inference.vi` is still a stub package. Concrete three-commit build plan (W12.1/2/3) landed in W10.8 — `plans/05b_inference_vi_design.md` pins the variational family, ELBO estimator, module layout, and acceptance criteria. Implementation deferred to W12, post the MCMC `lax.scan` refactor (W11) so both share the same `optax + lax.scan + state pytree` infrastructure. |
| IMPL-09 | Diagnostics + logging | [12](../plans/12_diagnostics.md) | DONE | `py_idr.eval.diagnostics_report` packages arviz split-$\hat R$, ESS, divergence-fraction into a pass/warn/fail verdict with documented thresholds. Divergent-transition plumbing landed in W8.11: `sweep_simple` / `multi_cluster_sweep` now return a per-sweep `num_divergences` count, `run_chain_*` aggregate across post-warmup sweeps, and the count rides on `ChainResult*.num_divergent_transitions`. |
| IMPL-10 | Reproducibility infrastructure | [13](../plans/13_reproducibility.md), [14](../plans/14_docker_a10dev.md) | PARTIAL | Pydantic manifests + per-run `run_manifest.json`, NPZ idr sidecar, verdict ledger (`py-idr verdict`, W8.8), `run_config.json` + `scripts/replay_run.sh` (W8.9), arviz cross-version compat helper (W9.2, `0958375`), arXiv-tarball builder `scripts/build_arxiv_tarball.sh` (W10.4), and the **W10.9 `.gitignore` fix** that finally tracked `src/py_idr/data/{__init__,schema}.py` in git all landed. Dockerfile + `.dockerignore` already checked in for plan 14; Snakemake workflow target + GHCR-image GitHub Actions are still TODO. |

## Simulation tasks

| ID | Task | Plan | Status | Notes |
|----|------|------|--------|-------|
| SIM-01 | Factorial S1–S5 + S5-sparse | [07](../plans/07_simulation_studies.md) | DONE | `py_idr.simulation.scenarios.simulate_S{1..5}` + `simulate_S5_sparse` all implemented. `run_replicate` dispatcher + `run_sweep` / `run_sweep_with_traces` + the `py-idr sim` CLI complete the loop. S5 routes through the multi-cluster chain by default; T=1 ablation available via `--chain-type T=1`. |
| SIM-02 | Component ablations | [07](../plans/07_simulation_studies.md) | PARTIAL | `--chain-type T=1 / T>1` ablation lives; comparator ablations land via `--method {py-idr, vanilla-idr, maxrank}` (W8.4 / W8.6) and the multi-method demo in `scripts/multi_method_demo.sh` (W8.7). EM warm-start (W8.5) is opt-in via `em_init=True` in the chain drivers. Per-component (PY vs DP, Gaussian-only vs heavy-tail) sweeps are still on paper. |
| SIM-03 | Robustness / stress tests | [07](../plans/07_simulation_studies.md), [add-stress-test](../.claude/skills/add-stress-test/SKILL.md) | TODO | Catalogue per §4.6; one config per stressor planned. |

## Real-data tasks

| ID | Task | Plan | Status | Notes |
|----|------|------|--------|-------|
| REAL-01 | ENCODE 4 CTCF ChIP-seq | [08](../plans/08_real_data_pipelines.md) | TODO | Loader stub only; pre-fit `DatasetManifest` not yet implemented. |
| REAL-02 | ENCODE 4 ATAC-seq | [08](../plans/08_real_data_pipelines.md) | TODO | Same as REAL-01. |
| REAL-03 | DREAM5 gene-network challenge | [08](../plans/08_real_data_pipelines.md) | TODO | $K = 35$ — high-K stress concomitant. |
| REAL-04 | Tabula Muris pseudobulk | [08](../plans/08_real_data_pipelines.md) | TODO | Pseudobulk aggregation pipeline + technology-stratified marginals. |
| REAL-05 | Pan-UKBB GWAS | [08](../plans/08_real_data_pipelines.md) | TODO | DAC compliance check needed before download; subset of variants TBD per handoff §8. |

## Method-correctness audit (W8.1)

A focused review of Algorithm 1 against the report turned up two
PRNG-key reuse bugs and several cross-checks. Findings:

**Fixed bugs (commit `4e98a63`).**

- *Step 8 π update.* `multi_cluster_sweep` was splitting the per-sweep
  key into 6 subkeys and re-using `k_class` (Step 1) as the input to
  `update_pi` at Step 8. Independent draws share a subkey → silently
  correlated samples. Now splits into 7.
- *Pólya-urn auxiliary loop.* `_polya_urn_reassign_all` split the
  auxiliary key into `H + 1` subkeys with the indexing scheme
  `type_pick=keys[h]`, `params=keys[h+1]` — so the params key for atom
  `h` equalled the type-pick key for atom `h+1`. Now splits into
  `2H` independent subkeys.
- Regression test added: `tests/inference/test_run_chain_multi.py::test_multi_cluster_sweep_uses_distinct_keys_per_step`
  inspects the source for both `split(key, 7)` and `split(aux_key, 2 * H)`.

**Cross-checks (all pass).**

- Gaussian copula log-density vs `scipy.stats.multivariate_normal`:
  max abs diff $\sim 10^{-6}$ on four test points.
- Clayton density vs closed form at $K = 2, 3$: max abs diff
  $\sim 10^{-7}$.
- Log-EPPF vs direct rising-factorial sum at three partitions: max abs
  diff $\sim 10^{-5}$.
- Sun-Cai step-up: hand-computed 8-point example reproduces
  $\hat k_\alpha = 4$, $\hat\gamma_\alpha = 0.10$. Permutation-invariance
  verified. Empty-rejection corner case correctly returns
  $\hat k_\alpha = 0$.
- Hellinger between two exchangeable Gaussian copulas: closed form
  vs $10^5$-sample MC agrees to within MC noise.

**Known limitations (not bugs).**

- The current default chain lengths in integration tests are short
  (50–150 warmup, 100–300 samples) and the random initialisation
  `(pi_init=0.5, rho_init=0.3)` is far from the S1 truth
  `(0.30, 0.85)`. Recovery on real-scale problems needs longer warmup
  (the report's §3.3.1 recommends $\ge 500$ warmup + $\ge 500$ samples
  on 4 over-dispersed chains). The EM-initialised start documented in
  plan 04 is still TODO.
- The multi-cluster chain on S5 can drift to a pathological mode
  (very small posterior $\pi$, $T$ saturating $T_\max$) when started
  from the Gaussian-only initial atom and run with tiny warmup.
  The fix is the same: longer warmup, or wire up the EM init.

## Follow-on workstreams (W8.2 – W10.10)

| Tag | Workstream | Closes | Commit / file |
|-----|------------|--------|---------------|
| W8.2 | jaxtyping shape strings cleared for comparator modules | ruff F722 noise blocking comparator merges | `pyproject.toml` per-file ignore |
| W8.3 | numpy 2.0 `trapezoid` shim | ROC/PR AUC computation on the post-2.0 numpy stack | `decision/roc_pr.py` |
| W8.4 | Vanilla IDR (LBHB 2011) comparator | benchmarks against the foundational IDR baseline | `comparators/vanilla_idr.py`, `run_replicate_vanilla_idr` |
| W8.5 | EM warm start | chain-mixing limitation noted in the audit; pairwise Vanilla IDR seeds $(\pi, \rho, Z)$ | `inference/mcmc/em_init.py`, `em_init=True` knob on all chain drivers |
| W8.6 | MaxRank (Philtron–Lyu–Li–Ghosh 2018) comparator | non-parametric rank baseline | `comparators/maxrank.py`, `run_replicate_maxrank` |
| W8.7 | Multi-method demo | end-to-end "real T4 row with three methods" pipeline | `scripts/multi_method_demo.sh` |
| W8.8 | Verdict ledger | plan 13 acceptance-criteria machinery (H1 FDR control, H2/H3 power dominance) | `repro/verdict.py`, `py-idr verdict`, `tests/repro/test_verdict.py` |
| W8.9 | Replay script + `run_config.json` | plan 13 replayability — `<run_dir>` is self-contained for replay on CPU | `scripts/replay_run.sh`, `run_config.json`, `tests/repro/test_replay.py` |
| W8.11 | NUTS divergence plumbing | plan 12's divergence-fraction status moves from PARTIAL → DONE; the diagnostics report now sees a real signal instead of `0` | `sweep_simple`, `multi_cluster_sweep`, `run_chain.py`, `tests/inference/test_run_chain_multi.py::test_run_chain_multi_aggregates_divergences_from_sweep_diag` |
| W9.0 | `em_init` plumbed through sim CLI | sweep CLI now honours `--em-init` and records it in `run_config.json` | `dc7a11f`: `cli.py`, `simulation/sweep.py`, `simulation/replicates.py`, `scripts/replay_run.sh` |
| W9.1 | Submission-readiness audit + prose rewrite | drops "illustrative synthetic placeholder" framing across §4 captions, adds §4.7 "Empirical insights", rewrites abstract / §5 to reflect submission-scale slice | `2051959`: `docs/report_submission_readiness.md`, `scripts/sweep_submission.sh`, `docs/report/sections/{0-abstract,3.method,4.experiments,5.conclusion}.tex` |
| W9.2 | arviz cross-version compat | `KeyError 'pi'` on a10-dev with arviz 0.23 vs working on local arviz 1.1 | `0958375`: `utils/arviz_compat.idata_from_posterior`, four call sites in `run_chain.py`, three in `tests/diagnostics/test_report.py` |
| W9.3 | Real F1-F5 PDFs + T4 LaTeX | placeholder figures + tables replaced with real reproducer output from the CPU sweep; verdict ledger baked into `docs/report/assets/` | `b320f12`: `docs/report/figures/*.pdf`, `docs/report/tables/table_sim_results.tex`, `docs/report/assets/verdicts_submission.json` |
| W9.4 | Final submission checklist | bibliography (64 cites all resolve), cross-refs (every `\ref` has a `\label`), placeholder inventory all intentional | `0c97fd9`: `docs/report_submission_readiness.md` rewritten end-to-end |
| W10.1 | Bibliography polish | audit pass over `refs.bib` (no fix needed; 72 entries clean) | _(no commit; in audit doc)_ |
| W10.2 | Acknowledgments / funding stubs | sentence-form stubs in §5 with `TODO(submission)` markers | `0d7dd9f`: `docs/report/sections/5.conclusion.tex` |
| W10.3 | LaTeX compile on remote | a10-dev has TeXLive 2020; `saim.cls` needs TeXLive 2025 (no sudo, no CTAN reach). Deferred to submission machine. | _(no commit; documented in audit doc)_ |
| W10.4 | arXiv `.tar.gz` bundle | one-shot script to package the self-contained LaTeX project | `0d7dd9f`: `scripts/build_arxiv_tarball.sh` |
| W10.5 | eCV comparator integrated | fourth method in the CSV; `--method ecv`; 6 unit tests | `e9e50ad`: `comparators/ecv.py`, `run_replicate_ecv`, `run_sweep_ecv`, `tests/unit/test_comparator_ecv.py` |
| W10.6 | MCMC chain-loop scan refactor design | three-commit `lax.scan` plan; unblocks the production-scale GPU sweep | `5025cff`: `plans/04b_inference_mcmc_scan_refactor.md` _(design only; implementation is W11)_ |
| W10.8 | VI back-end design | three-commit build plan with variational family, ELBO, acceptance criteria | `c4a6e5a`: `plans/05b_inference_vi_design.md` _(design only; implementation is W12)_ |
| W10.9 | Real-data loader stubs + critical `.gitignore` fix | five `NotImplementedError` loader stubs + per-dataset acquisition checklist; the gitignore bug had silently excluded `src/py_idr/data/{__init__,schema}.py` from git for the whole project history | `0ee450a`: `src/py_idr/data/loaders/__init__.py`, `docs/data_acquisition_checklist.md`, `.gitignore`, **first commit of `src/py_idr/data/__init__.py` and `src/py_idr/data/schema.py`** |
| W10.10 | Plans + handoff doc refresh | the present commit | _(this commit)_ |

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
