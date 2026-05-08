# Plan 02 — Algebra & Copulas

**Owner:** W1
**Depends on:** 01_repo_skeleton
**Implements:** §3.3.2 (reproducible-component copula), §3.3.4 (PLOD), Eq. (3.4)

## Goal

A correct, JAX-jitted reference implementation of the copula log-densities used by every
atom in the PY mixture, plus the PLOD probe that gates the Yakowitz–Spragins identifiability
argument.

## Deliverables

### `src/stick_idr/algebra/`

- `correlation.py`
  - `cholesky_to_corr(L) -> R` and `corr_to_cholesky(R) -> L` round-trip.
  - `lkj_log_density(L, eta=2)` (in unconstrained Cholesky parameterization).
  - `Structured(class_: Literal["exchangeable","one_factor","two_factor"], ...)` constructors that materialize $R$ with unit diagonal.
- `plod.py`
  - `holds(C: Copula, u_grid: jnp.ndarray) -> jnp.ndarray` returning a boolean grid for $C(u,...,u) > u^K$ at each $u$.
  - `assert_plod(C)` — convenience used by tests and by `copulas/registry.py` at registration time.
- `ranks.py` — empirical-rank to pseudo-uniform $U$, with three tie-breaking strategies (`average`, `random`, `truncate`).
- `numerics.py` — `safe_log`, `logsumexp_axis`, jitter helpers for Cholesky.

### `src/stick_idr/copulas/`

- `base.py` — abstract `Copula` with `log_density(u: jnp.ndarray) -> jnp.ndarray` and `cdf_diagonal(u: jnp.ndarray) -> jnp.ndarray` (the diagonal section, used by `plod.holds`).
- `independence.py` — $c_0(u) = 1$.
- `gaussian.py` — Cholesky-parameterized; reduces to $\mathcal{N}_K(0, R)$ density evaluated at $\Phi^{-1}(u)$ minus the marginal Gaussian densities.
- `student_t.py` — $t_5$ copula with correlation $R$ ($\nu = 5$).
- `clayton.py` — Archimedean Clayton; $\theta > 0$.
- `gumbel.py` — Archimedean Gumbel; $\theta \ge 1$.
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
