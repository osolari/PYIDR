---
name: polya-urn-sanity
description: Verify the Pólya-urn auxiliary-atom cluster reassignment (Eqs. (3.5)–(3.6) of the report) on a saved sampler state. Use when the user asks to check the Pólya-urn step, debug cluster reassignment, validate the auxiliary-atom predictive, or investigate suspicious cluster traces.
---

# Skill: Pólya-Urn Auxiliary-Atom Sanity Check

A correctness check for the Pitman–Yor Pólya-urn auxiliary-atom cluster reassignment
(§3.3.1 of the revised report; Eqs. (3.5)–(3.6)). Compares the JAX implementation's
reassignment probabilities against an analytic enumeration on a small held-out state.

> **Naming note.** The revised report names the sampler "Pólya-urn auxiliary-atom MCMC".
> Earlier drafts called it "slice-sampled Pólya-urn (Algorithm 8)"; the current sampler
> still uses Neal-style auxiliary atoms, but it does **not** maintain an explicit infinite
> stick during reassignment. The predictive formulas use the post-removal counts
> $n_{m,-i}$ and $T_{-i}$ — see the procedure below.

## When to invoke

Triggered by any of:

- "verify the Pólya-urn step" / "sanity check the auxiliary-atom predictive"
- "debug cluster reassignment"
- "the cluster trace looks wrong, check the predictive"
- "is Eq. (3.5) / (3.6) correct?"
- "verify Algorithm 8" (legacy keyword; same skill)

Do NOT use this skill for general MCMC diagnostics — use `mcmc-diagnostics-review` instead.

## Prerequisites

- A saved `SamplerState` PyTree under `runs/<run_id>/state.pkl` OR a tiny fixture
  ($K = 2$, $n = 20$, 3 active clusters).
- $H$ (auxiliary-atom count) known from the run config (default 5).

## Procedure

1. **Load state.** Read `runs/<run_id>/state.pkl` (or build the fixture).
2. **Pick a feature.** Choose a single feature index $i$ with $Z_i = 1$.
3. **Compute the post-removal counts.** Temporarily remove $i$ from its current cluster
   to obtain $n_{m,-i}$ and $T_{-i}$. **This step is critical** — the predictive uses the
   post-removal counts, not the pre-removal counts. A common bug is to use $T$ and $n_m$
   directly, which yields an off-by-one bias on singleton-deletion paths.
4. **Compute reassignment probabilities analytically.** For each of the $T_{-i} + H$
   candidate clusters (the existing $T_{-i}$ + $H$ auxiliary atoms drawn from $P_0$),
   compute:
   - Occupied (Eq. 3.5):
     $\Pr(z_i = m \mid \cdot) \propto (n_{m,-i} - \sigma)\, c_{\theta_m, t_m}(\bU_i)$
     for $m = 1, \ldots, T_{-i}$.
   - Auxiliary (Eq. 3.6):
     $\Pr(z_i = \widetilde m_h \mid \cdot) \propto \frac{\alpha + \sigma\, T_{-i}}{H}\,
       c_{\widetilde\theta_h, \widetilde t_h}(\bU_i)$ for $h = 1, \ldots, H$.
   Then normalise jointly across the $T_{-i} + H$ candidates.
5. **Compare to the JAX implementation.** Call
   `py_idr.pym.polya_urn.reassignment_log_probs(state, i, key, H)` and diff against
   the analytical computation. Tolerance: `1e-6` in float64.
6. **Monte-Carlo check.** Repeat `polya_urn.step(state, i, key)` $10^5$ times with
   different keys; histogram the new $z_i$ values; assert each bucket frequency matches
   the analytical probability within 3 SE.
7. **Edge cases:**
   - Singleton cluster (deletion test): when $i$ is the only member of its current
     cluster, $n_{m,-i} = 0$ for that cluster and $T_{-i} = T - 1$. The cluster is
     dropped from the occupied list; the auxiliary mass uses $T_{-i}$.
   - Empty active-cluster pool: only auxiliary atoms — verify the categorical normalises
     properly.
   - $\sigma = 0$ (DP special case): the predictive should reduce to Neal Algorithm 8
     for the DP.
8. **Report.** Print a short summary: max abs diff, KS statistic per bucket, pass/fail.

## Expected outputs

- Console: pass/fail summary.
- `artifacts/polya-urn-sanity/<run-id>/report.json` with:
  - `max_abs_diff_per_category` (occupied vs auxiliary)
  - `ks_p_value_per_bucket`
  - `singleton_path_passed: bool`
  - `dp_special_case_passed: bool`
  - `sampling_time_ms`

## Failure modes

- **Wrong PY-vs-DP branch.** Suggests `pym/polya_urn.py` is using `(n_m)` instead of
  `(n_m - σ)` for occupied clusters or `(α/H)` instead of `((α + σ T_{-i})/H)` for
  auxiliary atoms. The DP special case ($\sigma = 0$) should still reduce correctly.
- **`T_{-i}` vs `T` confusion.** Suggests the implementation uses the pre-removal $T$ in
  the auxiliary mass. This is the most common bug; isolate by running the singleton-
  deletion test alone.
- **Singleton deletion behaves badly.** Verify `polya_urn.step` correctly drops empty
  clusters and reassigns indices.
- **Auxiliary-atom mass off by $H$.** Suggests the user implemented the predictive
  without dividing by $H$, leading to inflated probability for new clusters.
