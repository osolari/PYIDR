# Plan 11 — Align Repo with the Revised Report

**Owner:** all workstreams (cross-cutting)
**Triggered by:** the Phase Three revision pass to `docs/report/` and `docs/report/CODING_AGENT_HANDOFF.md`.
**Goal:** bring `IMPLEMENTATION_PLAN.md`, the per-workstream sub-plans, the `.claude/skills/`,
and the in-progress code under `src/py_idr/` in line with the new theorem statements,
the renamed sampler, the corrected atom parameterizations, the new finite-sieve framing,
the `S5-sparse` regime, the new diagnostics list, and the reproducibility manifests.

This plan is **delta-only**: it lists the concrete edits required and the dependency
order. Items already correct under the prior pass are not re-listed.

---

## 0. What changed in the report

The `CODING_AGENT_HANDOFF.md` and `CHANGELOG.md` document 80+ tagged edits. The
ones with code/plan implications are grouped below.

### 0.1 Sampler renamed
- Old: "slice-sampled Pólya-urn MCMC" (Algorithm 8 with H aux atoms).
- New: "**Pólya-urn auxiliary-atom MCMC**". Same Neal-style auxiliary atoms, but the
  manuscript no longer describes a stick-breaking slice sampler — there is no infinite
  stick to maintain during cluster reassignment.
- Cluster-reassignment predictives now use `n_{m,-i}` and `T_{-i}` (after temporarily
  removing observation $i$) rather than `n_m`, `T`:
  $$\Pr(z_i = m \mid \cdot) \propto (n_{m,-i} - \sigma)\, c_{\theta_m, t_m}(\bU_i)$$
  $$\Pr(z_i = \widetilde m_h \mid \cdot) \propto \frac{\alpha + \sigma\, T_{-i}}{H}\,
     c_{\widetilde\theta_h, \widetilde t_h}(\bU_i)$$

### 0.2 Atom parameterizations corrected
- **Exchangeable**: $0 < \rho < 1$ (strict; was looser).
- **One-factor / two-factor**: replaced by the **normalize-from-PSD** parameterization:
  start from $\Sigma = D + \Lambda \Lambda^\top$ (PSD by construction), then form
  $R = \mathrm{diag}(\Sigma)^{-1/2}\, \Sigma\, \mathrm{diag}(\Sigma)^{-1/2}$.
  *Our current `OneFactor` / `TwoFactor` build $D + \Lambda\Lambda^\top$ with $D$ chosen
  to give unit diagonal — the old parameterization. Needs replacement.*
- **PLOD on factor models**: factor loadings constrained to a common sign OR rejection
  sampling until all pairwise correlations are strictly positive. New runtime hook.
- **Gumbel**: $\theta > 1$ for strict separation from independence (already what we
  enforce, but worth reaffirming in the docstring).
- **PY stick-breaking explicit**: $V_m \sim \Beta(1-\sigma, \alpha + m\sigma)$,
  $w_m = V_m \prod_{\ell < m}(1 - V_\ell)$. Notation now exposes $V_m$.

### 0.3 Decision rule corrected
- **Bayes-mFDR**: denominator is now $\max\{1, \E[R(\gamma) \mid \mathrm{data}]\}$ to
  handle empty rejection sets. Operationally equivalent to the original definition only
  when $R$ is large; the revised formula avoids 0/0.
- **Sun–Cai step-up**: explicit operational form
  $$\widehat k_\alpha = \max\left\{k \geq 1 : \frac{1}{k}\sum_{j=1}^k \widehat\idr_{(j)}
       \le \alpha\right\}$$
  with **no rejections if the set is empty** (we record this as a return-value contract).

### 0.4 Theory framing — finite-sieve and conditional
- **Theorem 3.1** is now "**Finite-sieve identifiability**" — it covers finite mixtures,
  not the unrestricted infinite PY mixture. The hypotheses are PLOD + no atom on the
  independence boundary + linear independence of $\{c_0, c_{\theta_1, t_1}, \ldots,
  c_{\theta_T, t_T}\}$.
- **Assumption 3.4 (sieve-and-tail)**: the exponential sieve-complement bound
  $\Pi(\mathcal F_n^c) \lesssim \exp\{-(C+4)\,n\epsilon_n^2\}$ is now an **assumption / proof
  obligation** for $\sigma > 0$, not a consequence of Lemma A.3's polynomial residual
  expectation. This corrects a real flaw in the prior pass.
- **Theorem 3.2** is "**Conditional Hellinger contraction**" — same rate, but conditional
  on Assumptions 3.1–3.4. Remark 3.6 (bounded scope) says the theorem covers bounded
  smooth copula truths; Clayton/Gumbel/$t_5$ are methodology but boundary-singular
  theory is future work.
- **Theorem 3.3 (mFDR)** is now $\alpha + O(\delta_n)$ where $\delta_n$ is a *local-idr*
  error scale, not the global Hellinger rate $\epsilon_n$. The threshold-regularity
  Assumption 3.5 is required.
- **Theorem 3.4** combines Harris (unconditional on the finite sieve) with a *conditional*
  geometric-rate strengthening — the latter only with an explicit drift / small-set
  hypothesis on the chosen finite parameterization.
- **Theorem 3.5** is "**Finite-truncation** variational consistency". Growing-$T$ case
  needs additional truncation-error and variational-approximation assumptions.

### 0.5 Experiments
- **S5-sparse** is a new planned regime: same heavy-tail mixture as S5 but $\pi^* = 0.10$,
  used for ROC/PR.
- Placeholder simulation tables now say $K = 5$, not $K = 4$.
- Per-replication artifact storage list is now explicit: true class labels, true cluster
  assignments, true copula family/parameters, marginal transformations, rank-tie info,
  posterior decision summaries.
- §4.6 ablations and stress tests are itemized in the report; we should match them in
  the simulation harness.
- §4.7 real-data plumbing now requires per-dataset: data manifest, feature-universe,
  replicate-matching file, score-column dictionary, tie/missingness policy,
  rank-transformation output, dataset-specific QC summary — *before* fitting.

### 0.6 Diagnostics
- §3.3.1 lists explicit required diagnostics: deterministic seed ledgers,
  rank-tie summaries, overdispersed chain starts, split-$\widehat R \le 1.01$,
  ESS $\ge 400$ for reported scalars, cluster-count traces, label-invariant summaries,
  NUTS divergent-transition checks, sensitivity to $H$.
- Remark A.6 (theory-to-code) maps diagnostics to assumptions: PLOD/positive-correlation
  checks, rank/marginal calibration for the Sklar step, posterior cluster-count traces
  for the PY prior, empirical Hellinger error for contraction, realized-FDR curves for
  the threshold-transfer.

### 0.7 Reproducibility infrastructure
- Reproducibility package list (§4.9): seed ledger, environment file, frozen data
  manifests, simulation configs, model-output schemas, figure/table generation scripts,
  container or workflow target.

---

## 1. Audit of repo against the IMPL-01..10 / SIM / REAL tasks

| Task | Status | Notes |
|------|--------|-------|
| IMPL-01 Core data schema | **TODO** | No `data/schema.py` yet; no `DataManifest` dataclass. |
| IMPL-02 Marginals | **STUB** | `marginals/bernstein.py` is `NotImplementedError`. |
| IMPL-03 Copula atom library | **DONE (W1)** | All four atomic types + independence implemented and tested. |
| IMPL-04 Valid correlation | **PARTIAL** | Cholesky/LKJ + Exchangeable correct; OneFactor/TwoFactor use the **OLD parameterization** (need rewrite per §0.2). PLOD rejection logic absent. |
| IMPL-05 PY Pólya-urn MCMC | **STUB** | All `pym/*.py` and `inference/mcmc/*.py` are stubs. Predictives need new $n_{m,-i}$, $T_{-i}$ formulas. |
| IMPL-06 NUTS/Metropolis atom updates | **TODO** | |
| IMPL-07 Local idr + Bayes-mFDR | **STUB** | `decision/sun_cai.py` stub. Needs `local_idr.py`, `bayes_mfdr.py` with `max(1, E[R])`. |
| IMPL-08 Finite-truncation VI | **TODO** | |
| IMPL-09 Diagnostics + logging | **TODO** | `inference/mcmc/diagnostics.py` not yet implemented. |
| IMPL-10 Reproducibility infrastructure | **TODO** | No Docker, no manifest schema. |
| SIM-01 Factorial S1–S5 + S5-sparse | **TODO** | `simulation/scenarios.py` empty. |
| SIM-02 Component ablations | **TODO** | |
| SIM-03 Robustness/stress tests | **TODO** | |
| REAL-01..05 | **TODO** | |

### Items the new report **invalidates** in our existing artifacts

1. **`.claude/skills/algorithm8-sanity/SKILL.md`** uses the predictive
   `(α + Tσ)/H` for auxiliary atoms — wrong; must be `(α + σ T_{-i})/H`. Same skill
   uses `(n_m - σ)` for occupied — must be `(n_{m,-i} - σ)`. The skill name is also
   misleading; rename to `polya-urn-sanity` to match the report's renamed sampler.
2. **`plans/04_inference_mcmc.md`** says "slice-sampled Pólya-urn MCMC" throughout.
   Rename to "Pólya-urn auxiliary-atom MCMC". Update predictive formulas to use
   `n_{m,-i}` / `T_{-i}`.
3. **`IMPLEMENTATION_PLAN.md`** preamble describes the sampler as "slice-sampled
   Pólya-urn (Neal Algorithm 8)". Update.
4. **`src/py_idr/algebra/correlation.py::OneFactor`** / `TwoFactor` use the old
   "set-D-to-make-unit-diagonal" parameterization. Replace with the
   "PSD-then-normalize" parameterization.
5. **`src/py_idr/algebra/plod.py`** docstring claims PLOD breaks the label symmetry
   between the reproducible and irreproducible components and "the residual mass
   recovers $\pi$" — that derivation is precisely what the new Theorem 3.1 *removes*.
   Revise the docstring to say PLOD distinguishes admissible reproducible atoms from
   the independence boundary; identifiability of $\pi$ comes from the finite-mixture
   linear-independence argument.
6. **`plans/10_theory_tests.md`** lists tests for old Theorem 3.1 wording and old
   Lemma A.3 exponential bound. Update to:
   - Theorem 3.1 → finite-sieve test on a small finite mixture.
   - Lemma A.3 → polynomial decay for $\sigma > 0$, exponential for $\sigma = 0$;
     gap to exponential sieve-complement explicitly noted.
   - Theorem 3.3 → calibration test reports the local-idr error scale $\delta_n$ and
     the threshold-stability diagnostic.
   - Theorem 3.4 → Harris-only on the finite sieve; geometric-rate test gated on a
     drift+small-set check we can choose to skip.
7. **`plans/02_algebra_and_copulas.md`**: docstring/formula for `OneFactor`/`TwoFactor`
   in the plan must be updated to the PSD-normalize form.
8. **`plans/06_decision_theory.md`**: the formula for Bayes-mFDR must include the
   `max(1, …)` denominator; the Sun–Cai contract must explicitly handle the
   empty-rejection case.
9. **`plans/07_simulation_studies.md`**: add S5-sparse; switch placeholder $K = 4 \to 5$;
   itemize the per-replication storage list (true labels, cluster assignments, copula
   family/parameters, marginal transformations, rank-tie info, decision summaries).
10. **`plans/08_real_data_pipelines.md`**: add the per-dataset *manifest +
    feature-universe + replicate-matching + score-dictionary + QC* checklist.

### Items the new report **adds** that we did not plan for

- **`S5-sparse`** simulation cell.
- **`plans/12_diagnostics.md`** (new): diagnostics module map per Remark A.6.
- **`plans/13_reproducibility.md`** (new): seed ledger, env file, dataset manifests,
  output schema, container.
- **Docker** deliverable (already discussed in the chat — the user has 4× A10s on `a10-dev`).
- **`δ_n` local-idr error scale** as a first-class quantity in `decision/`.

---

## 2. Concrete edits — ordered by dependency

The order below is so you can pick up at the top of the next branch and work
sequentially without circular blockers.

### Phase A — Plans / docs alignment (no code changes)

A1. **`IMPLEMENTATION_PLAN.md`** — replace "slice-sampled Pólya-urn (Neal Algorithm 8)"
    with "Pólya-urn auxiliary-atom MCMC (Neal-style auxiliary atoms; no explicit infinite
    stick during cluster reassignment)". Update the §3 algorithm map table, the
    cross-reference index, and the verification-of-done checklist. Mention §3.4's
    finite-sieve scoping for theorems.

A2. **`plans/02_algebra_and_copulas.md`** — replace the OneFactor/TwoFactor formula with
    the PSD-then-normalize spec. Add a note about the PLOD common-sign rejection logic.

A3. **`plans/04_inference_mcmc.md`** — rename the sampler. Update predictive equations.
    Update the diagnostics list to match §3.3.1's explicit required set
    (split-$\widehat R$, ESS, divergent transitions, NUTS, label-invariant summaries,
    sensitivity to $H$).

A4. **`plans/05_inference_vi.md`** — switch the variational family from `q(w)` to per-cluster
    `q(V_m)` Beta. Note that consistency is finite-truncation-only; growing-$T$ needs
    additional assumptions.

A5. **`plans/06_decision_theory.md`** — add the `max(1, E[R])` denominator. Add the
    empty-set return contract for Sun–Cai. Add the local-idr error scale $\delta_n$ as
    a first-class output (used in mFDR-rate diagnostics).

A6. **`plans/07_simulation_studies.md`** — add S5-sparse; bump $K = 4 \to 5$; itemize the
    per-replication storage list; add the §4.6 ablations + stress-test checklist.

A7. **`plans/08_real_data_pipelines.md`** — add the per-dataset manifest checklist
    (manifest + feature universe + replicate-matching + score dictionary + tie/missingness
    + QC).

A8. **`plans/10_theory_tests.md`** — restate every theorem test against the revised
    statements (finite-sieve identifiability, conditional contraction, $\delta_n$-rate
    mFDR, Harris-only on the finite sieve, finite-truncation VI).

A9. **`plans/12_diagnostics.md`** (new file) — the diagnostics module map per Remark A.6:
    PLOD/positive-correlation checks, rank/marginal calibration plots, posterior
    cluster-count traces, empirical Hellinger curves, realized-FDR vs nominal curves.

A10. **`plans/13_reproducibility.md`** (new file) — seed ledger schema, env file,
     dataset-manifest pydantic schema, output-directory schema, Docker target. Cross-references
     to IMPL-10 in the handoff.

A11. **`plans/14_docker_a10dev.md`** (new file) — Dockerfile spec, GHCR push, run-on-a10dev
     script, sharding strategy across the 4 GPUs (round-robin via `CUDA_VISIBLE_DEVICES`).
     This is the previously-discussed Docker plan, now formally a sub-plan.

A12. **`.claude/skills/algorithm8-sanity/`** — rename to `polya-urn-sanity/`, fix the
     predictive formulas to use $n_{m,-i}$ and $T_{-i}$, update the trigger keywords
     (drop "Algorithm 8" specifically, add "Pólya-urn auxiliary-atom").

A13. **`.claude/skills/run-simulation/SKILL.md`** — add S5-sparse to the inventory.

A14. **`.claude/skills/`** — new skill `add-stress-test` — for adding the §4.6 stress
     tests to the simulation harness without re-running the basic simulation flow.

A15. **`.claude/skills/`** — new skill `mcmc-diagnostics-review` is already there; just
     ensure its checklist matches §3.3.1's explicit list.

A16. **`docs/experiments/ledger.md`** schema — add columns for $\delta_n$, true-cluster
     recovery summary, and the per-replication artifacts so the verdict ledger can
     consume them.

### Phase B — Code corrections (small, well-scoped)

B1. **`src/py_idr/algebra/correlation.py`** — replace `OneFactor.matrix()` and
    `TwoFactor.matrix()` with the PSD-then-normalize form. Add `cov_to_corr(Σ)` helper.
    Update tests to verify unit diagonal and PSD on the normalized output. Drop the
    `jnp.maximum(..., 1e-8)` defensive clamp once the new parameterization is in place
    (it's no longer needed because $\Sigma$ is PSD by construction).

B2. **`src/py_idr/algebra/plod.py`** — rewrite the module docstring to match Theorem 3.1's
    finite-sieve framing. PLOD distinguishes reproducible atoms from $c_0$, but does not
    by itself identify $\pi$. Add a `plod_with_positive_pairwise(R)` helper for the factor
    rejection logic.

B3. **`src/py_idr/copulas/gaussian.py`** / `student_t.py` — add a runtime check that all
    pairwise correlations are strictly positive *if the atom is going to be registered
    under the PLOD constraint*. Currently the PLOD probe only checks the diagonal section;
    add the stronger pre-check used by §3.2.2 for factor-model atoms.

B4. **`src/py_idr/copulas/registry.py`** — update the docstring; the `gate_with_plod()`
    function is fine as-is (it already does a runtime PLOD check on representative
    atoms).

B5. **`src/py_idr/decision/sun_cai.py`** — implement the explicit step-up rule with the
    "no rejections if the set is empty" contract. Add a regression test that the empty
    case returns an empty rejection set, not a NaN.

B6. **`src/py_idr/decision/bayes_mfdr.py`** (new) — implement the new denominator
    `max(1, E[R(γ) | data])`. Add a `local_idr_error_scale(samples) -> δ_n` helper
    (Monte Carlo estimate of the largest deviation between feature-wise posterior idr
    estimates across chains/folds; used by Theorem 3.3 diagnostics).

B7. **`src/py_idr/pym/algo8.py`** — rename to `polya_urn.py` (the "Algo 8" label is
    legacy; `polya_urn_step` matches the report's section title). Update the predictive
    formulas to use `n_{m,-i}` and `T_{-i}`. Keep the `step()` entrypoint name so the
    skills/tests don't churn.

B8. **`src/py_idr/doctor.py`** — add a `polya-urn-step` probe that exercises one cluster
    reassignment on a fixture state and verifies the predictive formula at a single
    feature against an analytic reference. Flips from SKIP to PASS once B7 lands.

### Phase C — New scaffolding (preparing for W2/W3)

C1. **`src/py_idr/data/schema.py`** (new) — pydantic models for `DataManifest`,
    `ReplicateMeta`, `ScoreColumn`, `TruthMeta`. Implements IMPL-01.

C2. **`src/py_idr/marginals/bernstein.py`** — port the Bernstein cumulative basis from
    the W1 plan; now a real implementation per IMPL-02 / plans/03.

C3. **`src/py_idr/decision/local_idr.py`** — Monte Carlo posterior averaging of
    $\Pr(Z_i = 0 \mid \text{data})$, used by IMPL-07.

C4. **`src/py_idr/inference/mcmc/sweep.py`** — initial real implementation: class
    assignment + Pólya-urn cluster reassignment + atom-parameter NUTS + atomic-type IM
    + Bernstein conjugate + degree update + PY hyperparameters + π update. Routes calls
    to all the corrected predictives.

C5. **`src/py_idr/inference/mcmc/diagnostics.py`** — `arviz`-based wrapper that emits the
    explicit §3.3.1 diagnostics (split-$\widehat R$, ESS, divergent count, label-invariant
    summaries).

C6. **`Dockerfile`** + `scripts/run_on_a10dev.sh` — per `plans/14_docker_a10dev.md`.

### Phase D — Tests + verification

D1. **`tests/property/test_plod_invariance.py`** — extend the property tests to cover the
    new factor parameterization (PSD-then-normalize) and the common-sign loadings
    rejection logic.

D2. **`tests/identifiability/test_finite_sieve_id.py`** (new) — sample two finite
    sieves with the same marginals and the same `(π, c_1)` mixture density up to label
    permutation; assert posterior recovers them up to permutation.

D3. **`tests/decision/test_sun_cai_empty_rejection.py`** (new) — the all-large-idr case
    returns an empty set, not a NaN.

D4. **`tests/decision/test_bayes_mfdr_max_one.py`** (new) — the `max(1, E[R])` denominator
    keeps mFDR bounded as $\gamma \to 0$.

D5. **`tests/inference/test_polya_urn_predictive.py`** (new, replaces the planned Algo-8
    test) — Monte Carlo check of Eq. (3.5) and (3.6) on a fixed state matches the
    analytic predictive to within 3 SE.

D6. **Update `tests/numerical/test_py_truncation.py`** plan — split into the
    $\sigma = 0$ exponential bound (provable) and the $\sigma > 0$ polynomial bound
    (the only thing we can verify; the exponential sieve-complement gap from
    Assumption 3.4 is documented but not tested as a closed result).

### Phase E — Cleanup and consistency

E1. Walk every plan and skill once more after Phases A–D and confirm no remaining
    "slice-sampled" references.

E2. Update `IMPLEMENTATION_PLAN.md` §17 (cross-reference index) so each handoff task
    `IMPL-XX` and each `SIM` / `REAL` task in the handoff has a back-pointer to the
    sub-plan that owns it.

E3. Add a top-level `docs/coding_agent_handoff_status.md` with a live status table —
    `IMPL-01..10`, `SIM-01..03`, `REAL-01..05` each with `TODO/PARTIAL/DONE` and a link
    to the implementing PR.

---

## 3. Suggested commit order

1. **Commit 1 — "Align plans + skills with revised report"** (Phase A): docs only.
   Easy to review, no risk to running tests.
2. **Commit 2 — "Fix factor-model parameterization + PLOD positive-pairwise check"**
   (B1–B4 + D1): touches `algebra/correlation.py`, `algebra/plod.py`,
   `copulas/gaussian.py`, `copulas/student_t.py`, property tests.
3. **Commit 3 — "Decision module: Sun–Cai + Bayes-mFDR"** (B5, B6 + D3, D4):
   `decision/sun_cai.py`, `decision/bayes_mfdr.py`, `decision/local_idr.py` (stub OK).
4. **Commit 4 — "Pólya-urn auxiliary-atom rename + step formulas"**
   (B7, B8, A12 + D5): rename `pym/algo8.py` → `pym/polya_urn.py`, update doctor probe,
   update the renamed skill.
5. **Commit 5 — "Repro infrastructure: data schema + Docker on a10-dev"**
   (C1, C6 + plans 13/14): `src/py_idr/data/schema.py`, `Dockerfile`, run script.
6. **Commit 6 — "Marginals (Bernstein) + first real sampler sweep"**
   (C2, C4, C5 + integration smoke test): the first end-to-end sweep on a tiny S1 cell.

After commit 4 we are theoretically aligned with the new report; after commit 6 we
have the first running PY-IDR fit.

---

## 4. Open questions to confirm before / during execution

These mirror handoff §8 and surface choices we will need from the user:

- **OneFactor PLOD rejection vs. constrained loadings.** The report mentions both; the
  sampler can either reject proposals or use a sign-constrained parameterization.
  Plan B5 records the choice; the ablation table in §4.6 will measure whichever is
  picked.
- **Should the irreproducible component remain exactly independent in all experiments?**
  Defaulting to yes, with weak-dependence sensitivity recorded as part of stress tests
  per handoff §10.
- **PY positive-discount $\sigma > 0$ default.** The plan defaults to $\sigma$ free
  with a flat prior on $[0, 1)$; we record that the exponential sieve-complement is an
  assumption.
- **Heavy-tail atoms in the main model vs. robustness extension.** Defaulting to "in
  the main model" (the methodology section is unambiguous about this), but framing the
  contraction theorem statement as bounded-only per Remark 3.6.

---

## 5. Definition of done for this plan

- [ ] All `slice-sampled` references in the repo replaced with the new sampler name.
- [ ] OneFactor / TwoFactor parameterization matches §3.2.2 verbatim; tests cover unit
      diagonal + PSD + positive-pairwise (when PLOD-required).
- [ ] `decision/bayes_mfdr.py` uses `max(1, E[R])`; `decision/sun_cai.py` returns an
      empty rejection set on the empty-rule case.
- [ ] `pym/polya_urn.py` (renamed from `algo8.py`) implements Eq. (3.5) and (3.6) with
      `n_{m,-i}` and `T_{-i}`; doctor probe `polya-urn-step` flips from SKIP to PASS.
- [ ] `plans/12_diagnostics.md`, `plans/13_reproducibility.md`, `plans/14_docker_a10dev.md`
      land.
- [ ] `tests/identifiability/test_finite_sieve_id.py` is green.
- [ ] `docs/coding_agent_handoff_status.md` exists and tracks the IMPL/SIM/REAL tasks.
- [ ] One new commit per phase; each commit is independently reviewable.
