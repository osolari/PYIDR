# Plan 10 — Theory Tests

**Owner:** W3 + W4
**Depends on:** 02_algebra_and_copulas, 03_marginals, 04_inference_mcmc, 05_inference_vi, 06_decision_theory
**Implements:** Theorems 3.1–3.5; Lemmas A.1–A.4

## Goal

Every theorem and supporting lemma in the report has a corresponding test or empirical
property check. These are not full mathematical proofs; they are *operational sanity checks*
that catch regressions in code that would silently violate the theoretical guarantees.

## Coverage

### Theorem 3.1 — Identifiability

`tests/identifiability/`

- `test_yakowitz_spragins.py` — generate $M$ random Gaussian-copula correlation matrices; build the matrix of log-densities on a $u$-grid; assert the row rank is $M$ (linear independence).
- `test_plod_failure_breaks_id.py` — synthesize two copula mixtures that agree marginally but differ in $\pi$, both PLOD-violating; both produce ambiguous EM/MCMC fits. Switch on the PLOD-constrained prior; show identification is restored.
- `test_marginal_identifiability_via_sklar.py` — given the joint distribution, marginals are recovered to within MC error after a finite-sample sweep. (Sklar Step 1 of the proof.)

### Theorem 3.2 — Posterior contraction

`tests/numerical/`

- `test_contraction_simulation.py` — sample size sweep on regime S1, $K = 4$; empirical Hellinger error matches $\epsilon_n = n^{-\beta/(2\beta+K)} (\log n)^t$ within a 2× constant factor.
- `test_curse_of_dimensionality.py` — at fixed $n = 10^4$, the Hellinger error grows as $K$ increases per the $K$ in the exponent.
- `test_py_truncation.py` — Lemma A.3 bound on $\E[r_T]$ is satisfied empirically.
- `test_sieve_covering.py` — Lemma A.4 covering counts on a small explicit sieve agree with the bound.

### Theorem 3.3 — Asymptotic Bayes-mFDR control

`tests/decision/`

- `test_sun_cai_calibration.py` — at nominal $\alpha = 0.05$, realized FDR is within 1 SE of nominal on $\ge 80$ of 100 reps under regime S1.
- `test_mfdr_rate.py` — across $n \in \{10^3, 10^4, 10^5\}$, the deviation between realized and nominal FDR shrinks at the rate $\epsilon_n$ predicted by Theorem 3.3.

### Theorem 3.4 — Harris ergodicity (geometric ergodicity is a conjecture)

`tests/inference/`

- `test_chain_irreducibility.py` — starting two chains from disjoint cluster configurations produces overlapping posterior summaries after burn-in (Monte Carlo check, not analytic).
- `test_ess_growth.py` — ESS scales linearly with chain length on a fixed problem (a necessary condition for non-degenerate mixing).

### Theorem 3.5 — Variational consistency

`tests/integration/`

- `test_vi_matches_mcmc_means.py` — on a tiny problem, VI and MCMC posterior means agree within tolerance.
- `test_vi_recovers_pi.py` — on a synthetic S1 cell, VI recovers $\pi^*$ within 0.02.

### Lemma A.1 — Sklar regularity

`tests/property/test_sklar_regularity.py`

Empirically verify $h^2(c, c^*) \le C \KL(f, f^*) + C \sum_j \KL(f_j, f_j^*)$ on Gaussian and skew-normal pairs across a parameter sweep, recovering an explicit constant $C$ that matches the analytic argument.

### Lemma A.2 — Mixture-entropy decomposition

`tests/property/test_mixent_decomposition.py`

Assert $\KL(f, f^*) = \KL(c, c^*) + \sum_j \KL(f_j, f_j^*)$ on synthetic densities to numerical precision.

### Lemma A.3 — PY truncation tail bound

`tests/numerical/test_py_truncation.py`

For $\sigma \in \{0, 0.2, 0.4, 0.6, 0.8\}$, $\alpha \in \{0.5, 1, 2\}$: estimate $\E[r_T]$ by Monte Carlo and compare against the bound.

### Lemma A.4 — Sieve covering numbers

`tests/numerical/test_sieve_covering.py`

For $T \in \{5, 10, 20\}$, $K \in \{2, 5\}$, $M \in \{1, 2\}$: enumerate covering counts on a small explicit sieve and compare against the analytic bound.

## Acceptance criteria

- Every theorem in §3.9 of the report has at least one test in this plan.
- Every lemma in §A of the appendix has at least one test.
- Tests are tagged `slow` when their runtime exceeds 30 s; the fast subset runs in CI on every PR.

## Notes

- Operational sanity ≠ mathematical proof. These tests catch regressions in *code*, not in *math*. The math review lives in the LaTeX appendix and the companion paper.
- Where the theorem is asymptotic (Thm 3.2, 3.3), we test the *direction* (e.g., FDR deviation shrinks with $n$) rather than the constant.
- Theorem 3.4's geometric-rate strengthening is stated as a conjecture in the report; we do not test it. Harris ergodicity is the testable claim.
