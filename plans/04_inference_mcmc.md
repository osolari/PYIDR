# Plan 04 — Inference: Slice-Sampled Pólya-Urn MCMC

**Owner:** W2
**Depends on:** 02_algebra_and_copulas, 03_marginals
**Implements:** Algorithm 1 of the report; Eq. (3.8)–(3.9); §3.6 ("Slice-sampled Pólya-urn MCMC")

## Goal

A correct, JAX-friendly implementation of the full Algorithm 1 sweep:

```
Step 1.  Class assignment       Z_i ~ Bernoulli(π̂_i)        Eq. (3.8)
Step 2.  Cluster reassignment   Algorithm 8 with H aux atoms (Neal 2000)
Step 3.  Atom-parameter update  NUTS on Cholesky parameterization
Step 4.  Atomic-type update     IM proposal over {Gauss, t5, Clayton, Gumbel}
Step 5.  Bernstein-weight update  conjugate p_j | latent S
Step 6.  Marginal-degree update  MH-within-Gibbs on (M_j, α_F)
Step 7.  PY hyperparameters      Escobar–West for α; slice for σ ∈ [0,1)
Step 8.  Mixing proportion       π ~ Beta(1+n_1, 1+n_0)
```

## Deliverables

### `src/stick_idr/pym/`

- `stickbreak.py` — `gem_sample(key, alpha, sigma, T)` returning truncated stick-breaking weights $w_{1:T}$.
- `predictive.py` — PY Blackwell–MacQueen predictive (Pitman 1996).
- `algo8.py` — `step(state, key, H)` performing one cluster reassignment for one feature with $H$ auxiliary atoms drawn from $P_0$.
- `slice.py` — slice variable for stick-breaking; threshold sampler.
- `alpha_update.py` — Escobar–West auxiliary-variable update for $\alpha$.
- `sigma_update.py` — slice-sampled update for $\sigma \in [0, 1)$.

### `src/stick_idr/inference/mcmc/`

- `class_assign.py` — vectorized $Z_i \sim \mathrm{Bernoulli}(\widehat\pi_i)$.
- `reassign.py` — `vmap` over features calling `pym.algo8.step`.
- `nuts_atoms.py` — uses NumPyro's NUTS to update $\theta_m$ on the unconstrained Cholesky log-diagonal + free off-diagonals; one short chain per active cluster, configured for ~10 leapfrog steps.
- `type_update.py` — independence Metropolis over the four atomic-type families (Algorithm 1 step 4).
- `marginal_update.py` — Bernstein latent $S$ + Dirichlet conjugate draw; $\log\alpha_F$ random walk; discrete proposal on $M_j$.
- `pi_update.py` — Beta posterior draw.
- `sweep.py` — the orchestrator: takes a `SamplerState` PyTree and returns a new one.
- `diagnostics.py` — wraps `arviz` for split-$\widehat R$, ESS, divergences, energy, and cluster-occupancy traces.

### `src/stick_idr/model/`

- `state.py` — `SamplerState` PyTree (frozen dataclass) carrying $\pi$, $\alpha$, $\sigma$, active clusters, atomic types, Bernstein weights, $M_j$, $\alpha_F$, latent $Z_i$, $z_i$.
- `joint.py` — `joint_log_prob(state, U)` for diagnostics.
- `em_init.py` — bivariate IDR EM on each $\binom{K}{2}$ pair, then `kmeans` on the rank-transformed data to seed cluster assignments.

## Tests

- `tests/inference/test_algo8_step.py` — single Algorithm-8 reassignment moves features between clusters with the correct conditional probabilities, by Monte Carlo over 1e5 trials matching a brute-force enumeration to within 3 SE.
- `tests/inference/test_nuts_atoms.py` — NUTS on a synthetic K=4 Gaussian copula recovers the true correlation matrix; 0 divergences; split-$\widehat R \le 1.01$; ESS ≥ 400 on 1k post-warmup samples.
- `tests/inference/test_full_sweep_smoke.py` — one full Algorithm 1 sweep on a tiny S1 cell (K=2, n=200) completes in < 5 s and yields a finite log-likelihood.
- `tests/inference/test_alpha_sigma_updates.py` — KS test of Escobar–West $\alpha$ and σ-slice draws against analytic conditionals on a frozen state, p > 0.01 on 95% of reps.
- `tests/inference/test_pi_update.py` — Beta posterior draw matches `Beta(1+n_1, 1+n_0)` on a fixed state.

## Acceptance criteria

- Algorithm 1 sweep is functionally pure: `sweep(state, key) -> (new_state, key)`.
- The full sweep runs at ≥ 5 sweeps/s on a tiny S1 cell on CPU and ≥ 25 sweeps/s on a single A100.
- Diagnostics module emits an `arviz.InferenceData` from a chain of states (so we get traceplots and split-$\widehat R$ "for free").

## Notes

- Algorithm 8 with $H$ auxiliary atoms is a *Gibbs step*, not a Metropolis step; the auxiliary atoms are drawn from $P_0$, evaluated against the per-feature likelihood, then the new cluster index is sampled categorically. JAX-vectorize over features within each iteration; the auxiliary atoms are drawn fresh each iteration.
- The NUTS step is short (~50 leapfrog steps total per cluster) and cheap; do not run a full warmup each sweep — keep the step size adaptation persistent across sweeps.
- The Cholesky parameterization is $L_{j,j} = \exp(\eta_{j,j})$, $L_{j,k}$ free for $j > k$; $R = LL^\top$. The LKJ($\eta=2$) prior is on $R$, but the sampler operates on $\eta$. Use `numpyro.distributions.transforms.CholeskyTransform` plus the LKJ log-density.
