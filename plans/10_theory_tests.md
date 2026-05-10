# Plan 10 — Theory Tests (Revised)

**Owner:** W3 + W4
**Depends on:** 02_algebra_and_copulas, 03_marginals, 04_inference_mcmc, 05_inference_vi, 06_decision_theory
**Implements:** Theorems 3.1–3.5 and Lemmas A.1–A.4 of the **revised** report.

> The revised report restates several theorems as **conditional** (on explicit assumptions)
> or as **finite-sieve only** (Theorem 3.1) statements. Some claims that were tested as
> closed results in the prior pass are now proof obligations or future work
> (Lemma A.3 exponential sieve-complement for $\sigma > 0$; geometric ergodicity
> beyond the finite parameterisation; matching-rate VI for $T \to \infty$). The tests
> below assert what is provable on the finite computational sieve and document the
> remaining proof obligations rather than asserting them.

## Coverage

### Theorem 3.1 — Finite-sieve identifiability

Replaces the prior "Identifiability up to label permutation". Assumes:
1. PLOD holds for every reproducible atom in the finite sieve.
2. No reproducible atom lies on the independence boundary (i.e., positive PLOD slack).
3. The collection $\{c_0, c_{\theta_1, t_1}, \ldots, c_{\theta_T, t_T}\}$ is linearly independent.

`tests/identifiability/`

- `test_finite_sieve_id.py` — instantiate two finite sieves with the same marginals and
  the same mixing density up to label permutation; assert the posterior recovers them
  up to permutation, by computing a label-invariant cluster summary (Wasserstein on the
  cluster-correlation distribution).
- `test_yakowitz_spragins.py` — for $T \le 8$ and Gaussian-copula atoms with distinct
  correlation matrices, the matrix of log-densities on a $u$-grid has full row rank.
  This is the linear-independence pre-condition of Theorem 3.1.
- `test_plod_failure_breaks_id.py` — turn off the PLOD constraint, instantiate two
  mixtures that agree marginally but differ in $\pi$ with at least one boundary atom,
  show the EM/MCMC fits become ambiguous; turn PLOD back on and identification is
  restored.

### Theorem 3.2 — Conditional Hellinger posterior contraction

Same rate $\epsilon_n = n^{-\beta/(2\beta+K)}(\log n)^t$ but conditional on Assumptions
3.1–3.4. Remark 3.6 limits scope to bounded smooth copula truths.

`tests/numerical/`

- `test_contraction_simulation.py` — sample-size sweep on regime S1, $K = 4$, with
  $\beta = 1.5$ (SciPy-smoothed Gaussian copula); empirical Hellinger error decays
  matching $\epsilon_n$ within a 2× constant factor across $n \in \{500, 1000, 2000, 5000\}$.
- `test_curse_of_dimensionality.py` — at fixed $n = 10^4$, Hellinger error grows as $K$
  increases per the $K$ in the exponent.
- `test_py_truncation.py` — Lemma A.3: $\E[r_T] \leq \exp\{-T \log(1 + 1/\alpha)\}$ for
  $\sigma = 0$ exactly, and $\E[r_T] = O(T^{-(1-\sigma)/\sigma})$ for $\sigma \in (0, 1)$.
  We test only what is established. **The exponential sieve-complement bound is
  Assumption 3.4 in the revised report — a proof obligation, not a consequence of the
  polynomial residual expectation.** The test docstring documents this gap explicitly
  and skips it with a `pytest.skip` referencing Assumption 3.4.
- `test_sieve_covering.py` — Lemma A.4 covering counts on a small explicit sieve agree
  with the analytic bound.

### Theorem 3.3 — Asymptotic Bayes-mFDR control

Now $\alpha + O(\delta_n)$ where $\delta_n$ is a *local-idr* error scale, not the global
Hellinger rate. Requires Assumption 3.5 (decision-threshold regularity).

`tests/decision/`

- `test_sun_cai_calibration.py` — at nominal $\alpha = 0.05$, realised FDR is within 1 SE
  of nominal on $\ge 80$ of 100 reps under regime S1 ($K = 5$, $n = 10^4$).
- `test_mfdr_rate.py` — across $n \in \{10^3, 10^4, 10^5\}$, the deviation between
  realised and nominal FDR shrinks at a rate consistent with the empirical $\delta_n$
  (computed via `decision.local_idr.local_idr_error_scale`). This is operational
  rate-checking, not an asymptotic proof.
- `test_threshold_regularity.py` — empirical density of $\widehat\idr_i$ is bounded
  away from zero in a neighbourhood of $\gamma_\alpha^*$ on the simulation cells used
  in the rate test (Assumption 3.5 sanity check).

### Theorem 3.4 — Harris (+ conditional geometric)

Harris ergodicity is unconditional on the finite sieve under Assumption 3.6. Geometric
ergodicity is conditional on a separate drift+small-set hypothesis on the finite
parameterisation.

`tests/inference/`

- `test_chain_irreducibility.py` — starting two chains from disjoint cluster
  configurations produces overlapping posterior summaries after burn-in.
- `test_ess_growth.py` — ESS scales linearly with chain length on a fixed problem
  (necessary for non-degenerate mixing; not equivalent to geometric rate).
- `test_geometric_drift_skip.py` — explicitly skipped, with a docstring noting that
  geometric-rate certification requires a Foster–Lyapunov drift analysis on the
  chosen finite parameterisation; we do not attempt to verify it computationally.

### Theorem 3.5 — Finite-truncation variational consistency

Fixed-$T$ statement. Growing-$T$ matching-rate is future work.

`tests/integration/`

- `test_vi_matches_mcmc_means.py` — at fixed $T$, VI and MCMC posterior means agree
  within a tolerance proportional to $1/\sqrt{n}$.
- `test_vi_recovers_pi.py` — on a synthetic S1 cell at $T = 10$, VI recovers $\pi^*$
  within 0.02.
- `test_vi_truncation_sensitivity.py` — at $T \in \{5, 10, 20, 30\}$, the predictive
  Hellinger error decreases monotonically; the matching-rate as $T \to \infty$ is **not
  asserted** (it requires the additional Alquier–Ridgway / Yang–Pati–Bhattacharya
  conditions cited in the report's Theorem 3.5 statement).

### Lemma A.1 — Sklar regularity

`tests/property/test_sklar_regularity.py`

Empirically verify $h^2(c, c^*) \le C \KL(f, f^*) + C \sum_j \KL(f_j, f_j^*)$ on
Gaussian and skew-normal pairs across a parameter sweep, recovering an explicit
constant $C$ that matches the analytic argument **when the bounded-density and regular-
transform conditions hold**. The revised lemma adds these as explicit conditions; the
test asserts the inequality only on the parameter regime where they hold.

### Lemma A.2 — KL decomposition

`tests/property/test_mixent_decomposition.py`

For matched marginal transforms (true marginals known), assert
$\KL(f^*, f) = \KL(c^*, c) + \sum_j \KL(f_j^*, f_j)$. For mismatched transforms, assert
the bounded-distortion replacement under the regularity conditions of Lemma A.1.

### Lemma A.3 — PY truncation residual

Tested under Theorem 3.2 (`test_py_truncation.py`).

### Lemma A.4 — Sieve covering

Tested under Theorem 3.2 (`test_sieve_covering.py`).

## Acceptance criteria

- Every theorem in §3.4 of the revised report has at least one test in this plan.
- Every lemma in §A of the appendix has at least one test.
- Tests that target proof obligations (Lemma A.3 exponential sieve-complement;
  geometric ergodicity beyond finite parameterisation; matching-rate VI as $T \to \infty$)
  are explicitly skipped with `pytest.skip` and reference the assumption number; they
  are not red.
- Tests are tagged `slow` when their runtime exceeds 30 s; the fast subset runs in CI
  on every PR.

## Notes

- Operational sanity ≠ mathematical proof. These tests catch regressions in *code*,
  not in *math*. Math review lives in the LaTeX appendix and the companion paper.
- Where the theorem is asymptotic (3.2, 3.3), we test the *direction* (e.g., FDR
  deviation shrinks with $n$) rather than the constant or the sharp rate.
- Theorem 3.4's geometric-rate strengthening requires a drift+small-set analysis on a
  specific finite parameterisation; we do not assert it. Harris ergodicity on the
  finite sieve is the testable claim.
