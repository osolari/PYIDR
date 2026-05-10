# Plan 02 — Algebra & Copulas

**Owner:** W1
**Depends on:** 01_repo_skeleton
**Implements:** §3.3.2 (reproducible-component copula), §3.3.4 (PLOD), Eq. (3.4)

## Goal

A correct, JAX-jitted reference implementation of the copula log-densities used by every
atom in the PY mixture, plus the PLOD probe that gates the Yakowitz–Spragins identifiability
argument.

## Deliverables

### `src/py_idr/algebra/`

- `correlation.py`
  - `cholesky_to_corr(L) -> R` and `corr_to_cholesky(R) -> L` round-trip.
  - `cov_to_corr(Σ) -> R` — diagonal-normalisation helper: $R = \mathrm{diag}(\Sigma)^{-1/2}\,\Sigma\,\mathrm{diag}(\Sigma)^{-1/2}$. Used by Algorithm 1 step 3 to normalise covariance proposals to correlation matrices before the PLOD check.
  - `lkj_log_density(L, eta=2)` (in unconstrained Cholesky parameterization).
  - Structured-correlation classes per §3.2.2 of the report:
    - `Exchangeable(K, rho)` with $0 < \rho < 1$ — $R = (1-\rho)I + \rho\,\mathbf{1}\mathbf{1}^\top$.
    - `OneFactor(K, loadings)` and `TwoFactor(K, loadings)` use the **PSD-then-normalise** form: build $\Sigma = D + \Lambda\Lambda^\top$ with $D$ a free positive-diagonal scale and $\Lambda$ the factor loadings, then return `cov_to_corr(Σ)`. This replaces the older "set-D-to-make-unit-diagonal" parameterisation, which is silently invalid when the loadings are large.
- `plod.py`
  - `holds(C: Copula, u_grid: jnp.ndarray) -> jnp.ndarray` returning a boolean grid for $C(u,...,u) > u^K$ at each $u$.
  - `assert_plod(C)` — convenience used by tests and by `copulas/registry.py` at registration time.
  - `plod_with_positive_pairwise(R) -> bool` — the §3.2.2 runtime hook for factor models: returns `True` only if every pairwise off-diagonal entry of $R$ is strictly positive. Used by the sampler to either constrain factor loadings to a common sign or reject and re-propose.

PLOD framing per the revised Theorem 3.1: the PLOD condition distinguishes admissible
reproducible atoms from the independence boundary ($C_0(u,\ldots,u)=u^K$ exactly).
It does **not** by itself identify the mixing weight $\pi$ — identifiability of $\pi$
on the finite sieve comes from the linear-independence argument of Yakowitz–Spragins
applied to $\{c_0, c_{\theta_1, t_1}, \ldots, c_{\theta_T, t_T}\}$. Update the PLOD
docstring accordingly; do not claim diagonal-section recovery of $\pi$.
- `ranks.py` — empirical-rank to pseudo-uniform $U$, with three tie-breaking strategies (`average`, `random`, `truncate`).
- `numerics.py` — `safe_log`, `logsumexp_axis`, jitter helpers for Cholesky.

### `src/py_idr/copulas/`

- `base.py` — abstract `Copula` with `log_density(u: jnp.ndarray) -> jnp.ndarray` and `cdf_diagonal(u: jnp.ndarray) -> jnp.ndarray` (the diagonal section, used by `plod.holds`).
- `independence.py` — $c_0(u) = 1$.
- `gaussian.py` — Cholesky-parameterized; reduces to $\mathcal{N}_K(0, R)$ density evaluated at $\Phi^{-1}(u)$ minus the marginal Gaussian densities.
- `student_t.py` — $t_5$ copula with correlation $R$ ($\nu = 5$).
- `clayton.py` — Archimedean Clayton; $\theta > 0$.
- `gumbel.py` — Archimedean Gumbel; $\theta > 1$ (strict separation from independence per §3.2.2).
- `registry.py` — `ATOMIC_TYPES = ("gauss","t5","clayton","gumbel")`; on import calls `algebra.plod.assert_plod` against representative parameter draws.

## Tests

- `tests/unit/test_correlation.py` — Cholesky round-trip, structured-class shapes, LKJ sampling positive-definite.
- `tests/unit/test_copula_logpdfs.py` — Gaussian density agrees with `scipy.stats.multivariate_normal`'s log-pdf at $\Phi^{-1}(u)$ minus marginal corrections; Clayton / Gumbel match closed forms.
- `tests/property/test_plod_invariance.py` — sampled atoms from the prior support pass `plod.holds` on a $u$-grid.
- `tests/property/test_grade_preserving.py` (Sklar regularity) — empirical Hellinger / KL relationships on synthetic data.

## Acceptance criteria

- All copula `log_density` functions are JAX-jit-compatible and `vmap`-friendly along the feature axis.
- `algebra.plod.holds` returns `True` for every prior-supported atom in 1000 random draws across 4 atomic types and $K \in \{2, 5, 10\}$.
- Numerical agreement with SciPy reference: `max_abs_diff < 1e-9` in float64.

## Notes

- $t_5$ density uses the Cholesky-parameterized multivariate-$t$ with $\nu = 5$ fixed (the report does not vary $\nu$).
- Archimedean copulas store $\theta$ in log-space for numerical stability.
- The Gaussian copula log-density is the canonical pitfall: be careful to subtract the marginal Gaussian densities at the inverse-CDF arguments, not the multivariate density itself.
