# Plan 03 — Bernstein–Dirichlet Marginals

**Owner:** W1
**Depends on:** 01_repo_skeleton
**Implements:** §3.3.3, Eq. (3.7), §3.3.5 hyperprior on $M_j$

## Goal

A per-replicate marginal model
$F_j(x) = \sum_{r=1}^{M_j} p_{j,r}\, B_r(F_j^{(0)}(x); M_j)$
with conjugate Dirichlet weight updates and a discrete prior on $M_j$.

## Deliverables

### `src/stick_idr/marginals/`

- `pilot.py` — `kde_pilot_cdf(x, bandwidth)` returns a coarse pilot $F_j^{(0)}$ used as the index in the Bernstein basis.
- `bernstein.py`
  - `cumulative_basis(t: jnp.ndarray, M: int) -> jnp.ndarray` returning $B_r(t; M) = I_t(r, M-r+1)$ via `jax.scipy.special.betainc`.
  - `density_basis(t, M)` returning $b_r(t; M) = M\binom{M-1}{r-1} t^{r-1}(1-t)^{M-r}$ in log-space.
  - `cdf(x, weights, M, F0)` and `pdf(x, weights, M, F0)` evaluators.
- `dirichlet_weights.py`
  - `sample_latent_S(rng, X, weights, M, F0)` for the Bernstein-membership latents (used in the Gibbs update).
  - `update_weights(rng, S, alpha_F)` returning the conjugate Dirichlet posterior draw.
- `degree_prior.py` — discrete prior on $M_j \in \{5, 10, 20, 40, 80\}$ with weights $\propto 1/M$, normalized; predictive sampler and log-pmf evaluator.
- `transform.py` — `fit_marginal(X, weights, M, F0)` and `transform_X_to_U(X, marginal)` and the inverse, used by the simulation harness and the data loaders.

## Tests

- `tests/unit/test_bernstein_basis.py`
  - `cumulative_basis(t, M)` matches `betainc(r, M-r+1, t)` exactly.
  - $\sum_r B_r(t; M) - B_{r-1}(t; M) = 1$ is *not* trivially true for the cumulative basis; instead test that $\sum_r p_{j,r} (B_r(t;M) - B_{r-1}(t;M)) = F_j(t) \in [0,1]$ on a grid of weights.
  - density basis integrates to 1 on $[0, 1]$.
- `tests/unit/test_dirichlet_conjugate.py` — synthetic latent $S_{i,j}$ and a Dirichlet $\alpha_F$ → posterior matches `Dir(α_F + n_{j,1}, ..., α_F + n_{j,M_j})`.
- `tests/property/test_marginal_inverse.py` — `transform_X_to_U(X)` then `transform_U_to_X(U)` round-trips to within 1e-6 on a grid.

## Acceptance criteria

- All Bernstein evaluators are JAX-jit-friendly with `vmap` over $r$ and over the data axis.
- $M = 80$ is numerically stable in float32 (use the `betainc` path for the cumulative; log-domain accumulation for the density).
- Pilot CDF is shape-stable across all replicates and never returns 0 or 1 exactly (clamped to `[1e-6, 1-1e-6]`).

## Notes

- The pilot $F_j^{(0)}$ is by design a *bad* CDF; the Bernstein mixture is what corrects it. Don't over-engineer the pilot — a Gaussian-KDE smoothed empirical CDF with a wide bandwidth is the report's recommendation.
- $\alpha_F$ has a `Gamma(1,1)` hyperprior; its update is via Metropolis-within-Gibbs on `log α_F`.
