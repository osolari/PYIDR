# Simulation regimes (S1-S5)

The PY-IDR simulation grid mirrors §4.1 of the report. There are five
regimes, four of which are implemented today; the table below is the
canonical reference.

| Regime | Generator                          | $\pi^*$ | $\rho$  | Marginals                       | Latent copula      | What it stresses                                  |
|--------|------------------------------------|---------|---------|---------------------------------|--------------------|---------------------------------------------------|
| S1     | `simulate_S1`                      | 0.30    | 0.85    | Standard Gaussian (shared)      | Gaussian, exchangeable | Baseline IDR; should be a slam-dunk for both T=1 and T>1. |
| S2     | `simulate_S2`                      | 0.30    | 0.40    | Standard Gaussian (shared)      | Gaussian, exchangeable | Weak signal; tests Sun-Cai under low separability. |
| S3     | `simulate_S3`                      | 0.10    | 0.85    | Gaussian, per-replicate $\sigma_j$ | Gaussian, exchangeable | Sparse signal + marginal heterogeneity; stresses the Bernstein-Dirichlet update. |
| S4     | `simulate_S4`                      | 0.30    | 0.85    | Skew-normal, per-replicate $\xi_j$ | Gaussian, exchangeable | Marginal mismatch; rank pilot strips shape; chain must learn it back. |
| S5     | `simulate_S5` (not yet)            | 0.30    | —       | $t_5$ marginals                 | Clayton + Gumbel + $t_5$ mixture | Heavy-tail dependence and multiple copula clusters. |

All four implemented generators live in [`py_idr.simulation.scenarios`][]
and share the helper `_gaussian_copula_mixture_X`. The driver
`run_replicate_<Sx>` wraps each one into an end-to-end pipeline that
returns a `ReplicateRow`.

## S1 — baseline

```python
sim = simulate_S1(K=3, n=300, seed=0)
```

Standard textbook IDR. Reproducible features have an exchangeable
$\rho = 0.85$ Gaussian copula; irreproducible features are independent;
both share standard-Gaussian marginals. The chain should recover
$(\pi^*, \rho)$ near truth and the realised FDR at $\alpha = 0.05$ should
sit close to 0.05.

## S2 — weak signal

```python
sim = simulate_S2(K=3, n=300, seed=0)  # rho = 0.40 default
```

Same shape as S1, smaller $\rho$. The reproducible vs. irreproducible
local idr distributions overlap much more, so power drops sharply but
FDR control should still hold.

Typical posterior means at $n = 300$: $\hat\pi \approx 0.27$,
$\hat\rho \approx 0.36$. Realised FDR at $\alpha = 0.05$:
$\le 0.05$ on average. Power: 0.30-0.50 depending on the seed.

## S3 — sparse + heterogeneous Gaussian

```python
sim = simulate_S3(K=3, n=500, seed=0, sigma_grid=(1.0, 1.5, 2.0))
```

$\pi^* = 0.10$ — only 10% of features are reproducible. Each replicate
has a different marginal *scale*: $\sigma_1 = 1.0, \sigma_2 = 1.5, \sigma_3 = 2.0$.
The pilot rank transform strips the scale, so the chain sees the same
copula as S1 — but the Bernstein-Dirichlet correction has to *fit* a
non-uniform marginal to recover the right scaling.

The interesting test here is that the rank pilot is approximate, not
exact, on a finite sample. The Bernstein-Dirichlet update closes the
gap.

## S4 — marginal mismatch

```python
sim = simulate_S4(K=3, n=300, seed=0, skewness_grid=(-1.0, 0.0, 1.0))
```

Each replicate's marginal is a skew-normal with a different shape
parameter $\xi_j$: negative for $j = 1$, symmetric for $j = 2$, positive
for $j = 3$. The latent copula is the S1 Gaussian copula. The rank
pilot maps every replicate to the uniform, so the chain still sees the
right copula — but the Bernstein-Dirichlet has to learn three very
different marginal shapes, one per replicate.

S4 is the toughest test of the marginal model. The chain should still
recover $(\pi^*, \rho)$ but the per-replicate Bernstein weights will be
substantially non-uniform.

## S5 — heavy-tail mixture (next commit)

Not yet implemented. The plan:

- Reproducible component is a mixture of a Clayton copula (lower-tail
  dependence) and a Gumbel copula (upper-tail dependence), with the
  PY-mixture deciding the weights.
- Marginals are $t_5$, the same across all replicates (a "messy but
  shared" marginal model).
- $\pi^* = 0.30$.

S5 is the *only* regime that genuinely exercises the multi-cluster
($T > 1$) path. The other four should resolve to $T = 1$ posterior
mass even when the multi-cluster sampler is in use.

S5-sparse is the same as S5 with $\pi^* = 0.05$ and one of the two
copula clusters carrying only a tiny fraction of the reproducible mass.
It is a stress test of the new-cluster predictive in the Pólya-urn step.

## Running a sweep

For a fixed regime, run multiple seeds:

```python
from py_idr.simulation.replicates import run_replicate_S3

rows = [
    run_replicate_S3(K=3, n=500, seed=s, num_chains=2, n_samples=200)
    for s in range(20)
]

import pandas as pd
df = pd.DataFrame([r.to_dict() for r in rows])
print(df[["replicate_seed", "realized_fdr", "power"]].describe())
```

The scripts under `scripts/` automate the full grid. The sweep CSV
writer (plan 13, in progress) will produce the long-format file that
Table 4 / Figure 1 consume.

## Adding a new regime

To add an S6:

1. Add `simulate_S6` to `py_idr/simulation/scenarios.py`. Follow the
   pattern: validate args via `_validate_basic_args`, define a `marginal`
   closure, call `_gaussian_copula_mixture_X`, finalise via `_finalise`.
2. Register it in `_SIMULATORS` in `py_idr/simulation/replicates.py`.
3. Optionally add `run_replicate_S6` as a thin ergonomic wrapper.
4. Add tests in `tests/simulation/test_scenarios.py` covering: shapes,
   determinism, regime-specific structural property, validator errors.

The cross-regime validator parametrised test catches missing
`_validate_basic_args` calls.
