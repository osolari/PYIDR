# Quickstart

This page walks through one end-to-end synthetic fit so you can see the
moving parts before reading the math primers. It assumes you have
installed PY-IDR with the `dev` extra and have a working Python session.

## 1. Generate a synthetic dataset

The simulation regimes from §4.1 of the report (S1–S4 implemented, S5 in
progress) all live in [`py_idr.simulation.scenarios`][]. The simplest is
S1: a $K$-replicate Gaussian-copula mixture with shared standard-Gaussian
marginals.

```python
from py_idr.simulation.scenarios import simulate_S1

sim = simulate_S1(K=3, n=300, pi_star=0.30, rho=0.85, seed=0)
print(sim.X.shape)        # (300, 3) — raw scores
print(sim.true_Z.shape)   # (300,)   — 0/1 ground truth
print(sim.true_pi)        # 0.30     — reproducible fraction
print(sim.true_rho)       # 0.85     — Gaussian-copula correlation
```

`sim.X` is a `(n, K)` matrix of *raw* replicate scores on the
standard-Gaussian scale. `sim.true_Z[i] = 1` means feature $i$ was drawn
from the reproducible component; `0` means irreproducible. The two blocks
are shuffled so the chain does not see the block structure.

## 2. Build the pilot CDF input

The chain consumes a "pilot CDF" — every column of $F_0$ should lie in
$[0, 1]$ approximately uniform under the irreproducible component. The
default is the empirical rank transform:

```python
import jax.numpy as jnp
from py_idr.marginals.transform import empirical_rank_transform

F0 = jnp.stack(
    [empirical_rank_transform(sim.X[:, j]) for j in range(sim.K)],
    axis=1,
)
print(F0.shape)  # (300, 3) — same shape, values in [0, 1]
```

The pilot is strictly *initial* — the chain's Bernstein-Dirichlet update
nudges the marginal away from the empirical rank toward whatever shape is
consistent with the latent copula.

## 3. Run a multi-chain fit (T = 1 path)

For S1–S4 the **T = 1 fast path** is the right model: a single Gaussian
copula atom plus the independence-copula irreproducible component. Use
`run_multi_chain_simple` from [`py_idr.inference.mcmc.run_chain`][]:

```python
import jax
from py_idr.inference.mcmc.run_chain import run_multi_chain_simple

result = run_multi_chain_simple(
    F0=F0,
    M=5,                  # Bernstein polynomial degree
    key=jax.random.PRNGKey(0),
    num_chains=2,
    n_warmup=50,
    n_samples=200,
    nuts_warmup=20,
)

print(result.idata)        # arviz InferenceData
print(result.z_samples.shape)  # (num_chains, n_samples, n)
```

The `result.idata.posterior` group has scalar traces for `pi`,
`rho`, `alpha`, `sigma`, the Bernstein weights, and the bookkeeping
variable `n_reproducible`. The `z_samples` field is the **Z-trace** — one
binary indicator per feature per draw — which the proper Monte Carlo
local-idr estimator consumes.

## 4. Compute posterior local idr & decisions

The Sun-Cai step-up rule operates on the per-feature local idr, which is
the posterior probability that feature $i$ is *irreproducible* given the
chain. The cleanest estimate is the Monte Carlo average $1 - \bar z_i$:

```python
from py_idr.decision.local_idr import from_mcmc
from py_idr.decision.sun_cai import sun_cai_step_up

z_flat = result.z_samples.reshape(-1, result.z_samples.shape[-1])
idr = from_mcmc(jnp.asarray(z_flat))   # shape (n,), in [0, 1]

decision = sun_cai_step_up(idr, alpha=0.05)
print(decision.k_alpha, decision.threshold)
print(int(decision.is_called.sum()), "features called at alpha=0.05")
```

`decision.is_called` is a boolean vector — the reproducible features the
rule selects at the requested $\alpha$.

## 5. Evaluate against ground truth

Because this is synthetic data, the **realised** FDR and power are
checkable:

```python
from py_idr.simulation.evaluation import evaluate_recovery

recovery = evaluate_recovery(idr, sim.true_Z, alpha=0.05)
print(recovery.realized_fdr, recovery.power, recovery.k_alpha)
```

## End-to-end one-liner

Steps 1–5 are bundled in `run_replicate_S1`:

```python
from py_idr.simulation.replicates import run_replicate_S1

row = run_replicate_S1(
    K=3, n=300,
    pi_star=0.30, rho=0.85, seed=0,
    alpha=0.05,
    num_chains=2, n_samples=200,
)
print(row.to_dict())
```

The `ReplicateRow` is the canonical sweep-row layout; it is what the
Table 4 / Figure 1 reproducers consume.

## Where to go next

- The same pattern works for S2 (`run_replicate_S2`, weak signal), S3
  (`run_replicate_S3`, sparse + heterogeneous Gaussian scale), and S4
  (`run_replicate_S4`, skewed marginals).
- For a real-data fit (no synthetic ground truth), build `F0` from your
  own score matrix via `empirical_rank_transform` and skip the
  `evaluate_recovery` step. The chain code is identical.
- If you suspect the reproducible component is *not* a single Gaussian
  copula — e.g. heavy upper-tail dependence — use the multi-cluster path
  `run_chain_multi` instead of `run_chain_simple`. See
  [Concepts](concepts.md) for the trade-off.
