# Plan 04 — Inference: Pólya-Urn Auxiliary-Atom MCMC

**Owner:** W2
**Depends on:** 02_algebra_and_copulas, 03_marginals
**Implements:** Algorithm 1 of the revised report; Eqs. (3.5)–(3.6); §3.3.1 ("Pólya-urn auxiliary-atom MCMC")

## Status snapshot (W8.1)

| Step | Module | Status |
|------|--------|--------|
| 1. Class assignment | `inference/mcmc/class_assign.py` | DONE |
| 2. Pólya-urn cluster reassignment | `inference/mcmc/sweep_multi._polya_urn_reassign_all` + `pym/polya_urn.py` | DONE |
| 3. Atom-parameter NUTS | `inference/mcmc/nuts_atoms.py` + `atom_dispatch.py` | DONE (Gaussian / Clayton / Gumbel-K2 / $t_5$) |
| 4. Atomic-type IM | `inference/mcmc/type_update.py` | DONE (Gumbel excluded from proposal pool at $K \ge 3$) |
| 5. Bernstein-weight conjugate | `marginals/dirichlet_weights.py` | DONE |
| 6. Marginal-degree update | `marginals/degree_prior.py` | PARTIAL (helper present, not currently wired into sweep) |
| 7. PY $(\alpha, \sigma)$ NUTS | `inference/mcmc/py_hyperparam_update.py` + `pym/eppf.py` | DONE |
| 8. $\pi$ Beta update | `inference/mcmc/pi_update.py` | DONE |

**Sweep entry points:** `inference/mcmc/sweep_simple.py` ($T = 1$ fast path) and `inference/mcmc/sweep_multi.py` (full Algorithm 1). Top-level drivers `run_chain_simple`, `run_multi_chain_simple`, `run_chain_multi`, `run_multi_chain_multi` in `inference/mcmc/run_chain.py`. Output is an `arviz.InferenceData` plus a Z-trace (and a cluster-label trace for the multi-cluster path).

**Correctness audit findings (W8.1):** two PRNG-key reuse bugs fixed in commit `4e98a63`. (1) Step-8 $\pi$ update was reusing the step-1 class-assignment key. (2) Polya-urn aux loop reused subkeys between adjacent iterations. Regression test in `tests/inference/test_run_chain_multi.py` guards both.

> **Naming.** The revised manuscript names this scheme **Pólya-urn auxiliary-atom MCMC**.
> Earlier drafts called it "slice-sampled Pólya-urn (Algorithm 8)"; that label has been
> retired because the sampler does not maintain an explicit infinite stick during
> cluster reassignment. The Neal-style auxiliary-atom mechanism is retained.

## Goal

A correct, JAX-friendly implementation of one full Algorithm 1 sweep:

```
Step 1.  Class assignment       Z_i ~ Bernoulli(π̂_i)        Eq. (3.4)
Step 2.  Cluster reassignment   PY Pólya-urn with H aux atoms drawn from P_0
                                using Eqs. (3.5)–(3.6) over (n_{m,-i}, T_{-i})
Step 3.  Atom-parameter update  NUTS or Metropolis on a valid unconstrained
                                parameterization; covariance factors normalised
                                to correlation matrices via cov_to_corr; reject
                                proposals outside the PLOD support
Step 4.  Atomic-type update     IM proposal over {Gauss, t5, Clayton, Gumbel}
Step 5.  Bernstein-weight update  conjugate p_j | latent S
Step 6.  Marginal-degree update  MH-within-Gibbs on (M_j, α_F)
Step 7.  PY hyperparameters      auxiliary-variable / slice updates on (α, σ)
Step 8.  Mixing proportion       π ~ Beta(1 + n_1, 1 + n_0)
```

## Predictive formulas (revised)

For feature $i$ with $Z_i = 1$, let $T_{-i}$ be the number of occupied clusters and
$n_{m,-i}$ the count in occupied cluster $m$, **after temporarily removing observation
$i$** from its current cluster. Draw $H$ auxiliary atoms $\widetilde\theta_h \sim P_0$.
Then

$$\Pr(z_i = m \mid \cdot) \propto (n_{m,-i} - \sigma)\, c_{\theta_m, t_m}(\bU_i),
   \qquad m = 1, \ldots, T_{-i} \quad\text{(Eq. 3.5)}$$

$$\Pr(z_i = \widetilde m_h \mid \cdot) \propto \frac{\alpha + \sigma\, T_{-i}}{H}\,
   c_{\widetilde\theta_h, \widetilde t_h}(\bU_i),
   \qquad h = 1, \ldots, H \quad\text{(Eq. 3.6)}.$$

The occupied and auxiliary weights are normalised jointly; choosing an auxiliary atom
instantiates a new occupied cluster.

> **Common error to avoid.** Earlier internal drafts used `(α + Tσ)/H` (no `-i` subscript)
> for the auxiliary mass — that yields an off-by-one bias in singleton-deletion paths.
> Always remove the observation first, then form `T_{-i}`.

## Deliverables

### `src/py_idr/pym/`

- `stickbreak.py` — `gem_sample(key, alpha, sigma, T)` returning truncated stick-breaking weights $w_{1:T}$, with the per-cluster ratio $V_m \sim \Beta(1-\sigma, \alpha + m\sigma)$ exposed (because the variational family uses per-$V_m$ factors, not joint $w$).
- `predictive.py` — PY Blackwell–MacQueen predictive (Pitman 1996) — analytical density used by the unit tests.
- `polya_urn.py` — `step(state, key, H)` implementing one Pólya-urn auxiliary-atom reassignment per Eqs. (3.5)–(3.6). The function takes the **post-removal** counts and returns the new cluster index plus the new state. (Module is named `polya_urn.py`, not `algo8.py`, to match the report.)
- `alpha_update.py` — Escobar–West auxiliary-variable update for $\alpha$.
- `sigma_update.py` — slice-sampled update for $\sigma \in [0, 1)$.

### `src/py_idr/inference/mcmc/`

- `class_assign.py` — vectorized $Z_i \sim \mathrm{Bernoulli}(\widehat\pi_i)$.
- `reassign.py` — `vmap` over features calling `pym.polya_urn.step`.
- `nuts_atoms.py` — uses NumPyro's NUTS to update $\theta_m$ on the unconstrained Cholesky log-diagonal + free off-diagonals. Inside the NUTS step, after every leapfrog the unconstrained $\Sigma$ is normalised via `algebra.correlation.cov_to_corr`; proposals whose normalised $R$ fails `algebra.plod.plod_with_positive_pairwise` are rejected. One short chain per active cluster, configured for ~10 leapfrog steps with persistent step-size adaptation across sweeps.
- `type_update.py` — independence Metropolis over the four atomic-type families (Algorithm 1 step 4).
- `marginal_update.py` — Bernstein latent $S$ + Dirichlet conjugate draw; $\log\alpha_F$ random walk; discrete proposal on $M_j$.
- `pi_update.py` — Beta posterior draw.
- `sweep.py` — the orchestrator: takes a `SamplerState` PyTree and returns a new one.
- `diagnostics.py` — wraps `arviz` for the explicit §3.3.1 list:
  - deterministic seed ledger
  - rank-tie summaries
  - overdispersed chain starts (4 chains is the default)
  - split-$\widehat R \le 1.01$
  - ESS $\ge 400$ for reported scalar summaries
  - cluster-count traces
  - posterior label-invariant cluster summaries (after a deterministic relabeling postprocessor)
  - NUTS divergent-transition counts (warn if $> 0.5\%$)
  - sensitivity to the auxiliary-atom count $H \in \{1, 3, 5, 10\}$

### `src/py_idr/model/`

- `state.py` — `SamplerState` PyTree (frozen dataclass) carrying $\pi$, $\alpha$, $\sigma$, active clusters, atomic types, Bernstein weights, $M_j$, $\alpha_F$, latent $Z_i$, $z_i$.
- `joint.py` — `joint_log_prob(state, U)` for diagnostics.
- `em_init.py` — bivariate IDR EM on each $\binom{K}{2}$ pair, then $k$-means or hierarchical clustering on the rank-transformed data to seed cluster assignments. (Per §3.3.1 of the revised report.)

## Tests

- `tests/inference/test_polya_urn_predictive.py` — single Pólya-urn reassignment on a fixed state matches Eqs. (3.5)–(3.6) by Monte Carlo over $10^5$ trials; bucket frequencies for occupied clusters and auxiliary-atom buckets each match the analytic predictive within 3 SE. Includes the singleton-deletion edge case (when $i$ is the only member of its current cluster, $n_{m,-i}=0$ for that $m$ and $T_{-i} = T - 1$).
- `tests/inference/test_nuts_atoms.py` — NUTS on a synthetic K=4 Gaussian copula recovers the true correlation matrix; 0 divergences; split-$\widehat R \le 1.01$; ESS $\ge 400$ on 1k post-warmup samples; PLOD-rejected proposals are accounted for in the diagnostic acceptance log.
- `tests/inference/test_full_sweep_smoke.py` — one full Algorithm 1 sweep on a tiny S1 cell (K=2, n=200) completes in < 5 s and yields a finite log-likelihood.
- `tests/inference/test_alpha_sigma_updates.py` — KS test of Escobar–West $\alpha$ and σ-slice draws against analytic conditionals on a frozen state, p > 0.01 on 95% of reps.
- `tests/inference/test_pi_update.py` — Beta posterior draw matches `Beta(1 + n_1, 1 + n_0)` on a fixed state.
- `tests/inference/test_factor_proposal_rejection.py` — Gaussian-copula NUTS step on a one-factor cluster: when a leapfrog produces a covariance $\Sigma$ whose normalised $R$ has any negative pairwise correlation, the proposal is rejected and the chain stays put. Tested at the algorithm step level, not full-sweep level.

## Acceptance criteria

- Algorithm 1 sweep is functionally pure: `sweep(state, key) -> (new_state, key)`.
- The full sweep runs at $\ge 5$ sweeps/s on a tiny S1 cell on CPU and $\ge 25$ sweeps/s on a single A100.
- Diagnostics module emits an `arviz.InferenceData` from a chain of states (so we get traceplots and split-$\widehat R$ "for free").
- Doctor probe `polya-urn-step` flips from SKIP to PASS once `pym/polya_urn.py` lands.

## Notes

- The PY Pólya-urn step is a *Gibbs* step (not Metropolis); the auxiliary atoms are drawn from $P_0$, evaluated against the per-feature likelihood, then the new cluster index is sampled categorically. JAX-vectorize over features within each iteration; the auxiliary atoms are drawn fresh each iteration.
- The NUTS step is short (~50 leapfrog steps total per cluster) and cheap; do not run a full warmup each sweep — keep the step-size adaptation persistent across sweeps.
- The Cholesky parameterisation is $L_{j,j} = \exp(\eta_{j,j})$, $L_{j,k}$ free for $j > k$; $R = LL^\top$. The LKJ($\eta=2$) prior is on $R$, but the sampler operates on $\eta$. Use `numpyro.distributions.transforms.CholeskyTransform` plus the LKJ log-density.
- For factor-structured atoms, the parameter operating space is the unconstrained $(D, \Lambda)$, with $\Sigma = D + \Lambda\Lambda^\top$ then normalised via `cov_to_corr` before density evaluation. Proposals failing `plod_with_positive_pairwise` are rejected at the step level. (See plans/02_algebra_and_copulas.md §3.2.2 update.)
