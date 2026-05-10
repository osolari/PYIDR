---
name: algorithm8-sanity
description: Verify Algorithm 8 cluster reassignment correctness on a saved sampler state. Use when the user asks to check the Pólya-urn step, debug cluster reassignment, validate Neal Algorithm 8 correctness, or investigate suspicious cluster traces.
---

# Skill: Algorithm 8 Sanity Check

A correctness check for the Algorithm 8 (Neal 2000) cluster reassignment step used by
PY-IDR's MCMC. Compares the JAX implementation's reassignment probabilities against a
brute-force enumeration on a small held-out state.

## When to invoke

Triggered by any of:

- "verify Algorithm 8" / "sanity check Algo 8"
- "debug the Pólya-urn step"
- "the cluster trace looks wrong, check reassignment"
- "is the auxiliary-atom step correct?"

Do NOT use this skill for general MCMC diagnostics — use `mcmc-diagnostics-review` instead.

## Prerequisites

- A saved `SamplerState` PyTree under `runs/<run_id>/state.pkl` OR ability to construct
  one from a fixture.
- `H` (auxiliary-atom count) known from the run config (default 5).

## Procedure

1. **Load state.** Read `runs/<run_id>/state.pkl` (or build a tiny fixture: $K=2$, $n=20$, 3 active clusters).
2. **Pick a feature.** Choose a single feature index $i$ with $Z_i = 1$.
3. **Compute reassignment probs analytically.** For each of the $T + H$ candidate clusters
   (the existing $T$ + $H$ auxiliary atoms drawn from $P_0$), compute:
   $\Pr(z_i = m) \propto (n_m - \sigma) \cdot c_{\theta_m}(\bU_i)$ for occupied clusters
   $\Pr(z_i = T+h) \propto (\alpha + T\sigma) / H \cdot c_{\theta_{T+h}}(\bU_i)$ for aux atoms.
4. **Compare to JAX implementation.** Call `py_idr.pym.algo8.reassignment_log_probs(state, i, key)` and
   diff against the analytical computation. Tolerance: `1e-6` in float64.
5. **Monte-Carlo check.** Repeat the JAX `algo8.step(state, i, key)` $10^5$ times with
   different keys; histogram the new $z_i$ values; assert each bucket frequency matches
   the analytical probability within 3 SE.
6. **Edge cases:**
   - Singleton cluster (deletion test): $n_m = 1$, $\sigma > 0$ — verify the predictive weight is $1 - \sigma$.
   - Empty active-cluster pool: only auxiliary atoms — verify the categorical normalises properly.
7. **Report.** Print a short summary: max abs diff, KS statistic, pass/fail.

## Expected outputs

- Console: pass/fail summary.
- `artifacts/algo8-sanity/<run-id>/report.json` with: max abs diff per category, KS p-value, sampling time.

## Failure modes

- **Mismatched predictive weights.** Suggests the PY-vs-DP branch in `pym/algo8.py` is wrong (forgot the $-\sigma$ on occupied clusters or the $+T\sigma$ on the auxiliary mass).
- **Auxiliary-atom mass off by $H$.** Suggests the user implemented the DP-style $\alpha/H$ rather than the PY-style $(\alpha + T\sigma)/H$.
- **Singleton deletion behaves badly.** Verify `algo8.step` correctly drops empty clusters and reassigns indices.
