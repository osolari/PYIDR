# Architecture

This page is a code-map for new contributors. It assumes you have read
[Concepts](../getting_started/concepts.md) and the math primers.

## Layered responsibilities

```
┌───────────────────────────────────────────────────────────────┐
│  scripts/  (sweep CLI, figure reproducers)                    │
├───────────────────────────────────────────────────────────────┤
│  py_idr.cli  (typer entrypoint, py_idr.doctor self-check)     │
├───────────────────────────────────────────────────────────────┤
│  py_idr.simulation                                            │
│    scenarios.py (S1-S4 generators)                            │
│    replicates.py (run_replicate dispatcher + per-regime)      │
│    evaluation.py (Sun-Cai realised-FDR helper)                │
├───────────────────────────────────────────────────────────────┤
│  py_idr.inference.mcmc                                        │
│    run_chain.py (top-level driver + arviz packaging)          │
│    sweep_simple.py (T=1 fast path)                            │
│    sweep_multi.py (full Algorithm 1)                          │
│    class_assign.py / pi_update.py (steps 1+8)                 │
│    nuts_atoms.py / atom_dispatch.py (atom NUTS)               │
│    type_update.py (atomic-type IM, step 4)                    │
│    py_hyperparam_update.py (PY (alpha, sigma) NUTS, step 7)   │
├───────────────────────────────────────────────────────────────┤
│  py_idr.decision                                              │
│    local_idr.py (from_mcmc → 1 - mean(Z))                     │
│    sun_cai.py (step-up rule)                                  │
│    bayes_mfdr.py (expected discoveries / posterior mFDR)      │
├───────────────────────────────────────────────────────────────┤
│  py_idr.marginals                                             │
│    transform.py (empirical_rank_transform, cdf_pilot)         │
│    bernstein.py (basis evaluation)                            │
│    dirichlet_weights.py (conjugate update)                    │
│    pilot.py / degree_prior.py                                 │
├───────────────────────────────────────────────────────────────┤
│  py_idr.copulas                                               │
│    gaussian.py / clayton.py / gumbel.py / student_t.py        │
│    independence.py / registry.py / base.py                    │
├───────────────────────────────────────────────────────────────┤
│  py_idr.pym                                                   │
│    stickbreak.py / polya_urn.py / eppf.py                     │
├───────────────────────────────────────────────────────────────┤
│  py_idr.algebra (plod.py, correlation.py)                     │
│  py_idr.model (state.py, multi_state.py)                      │
│  py_idr.eval (diagnostics_report.py)                          │
│  py_idr.data (schema.py — Pydantic manifests)                 │
│  py_idr.utils (run.py — IDs, timestamps)                      │
└───────────────────────────────────────────────────────────────┘
```

## State containers

PY-IDR uses two state types because the multi-cluster sampler has
*variable-length* per-cluster parameters that do not fit cleanly into a
fixed-shape JAX array.

### `SimpleSweepState` (T = 1 path)

A flat NamedTuple in `model.state` holding:

- `rho` — scalar Gaussian-copula correlation.
- `pi` — scalar reproducible mass.
- `M`, `weights` — Bernstein degree and weights $(K, M+1)$.
- `z` — current $Z$ vector, $(n,)$ int.
- `alpha`, `sigma` — PY hyperparameters (carried but unused at T=1).

Pure JAX — works inside `jax.lax.scan`, `jit`, `vmap`. The fast path
is **fully traced** so the chain runs as one XLA computation per
chain.

### `MultiClusterState` (T > 1 path)

A hybrid container in `model.multi_state` mixing:

- **JAX state**: scalars and fixed-shape arrays (`pi`, `M`,
  `weights`, `z`, `alpha`, `sigma`).
- **Python tuple `atoms`**: variable-length tuple of `Atom` dataclasses
  (`GaussianAtom`, `ClaytonAtom`, etc.), each carrying its own copula
  parameters.

The atoms tuple cannot be JAX-traced because its length varies as
clusters are born and die. The chain works around this by running the
*outer* loop in Python and JIT-ing the per-sweep step bodies inside.
The atoms tuple is reconstructed every sweep.

This trade-off is conscious: it costs a small constant factor of
Python overhead per sweep but makes the multi-cluster machinery
correct and easy to extend.

## The 8-step Gibbs sweep (Algorithm 1)

Implemented in [`py_idr.inference.mcmc.sweep_multi`][]:

1. **Class assignment** (`class_assign.py`) — resample $Z_i$ from its
   full conditional given the current cluster structure.
2. **Pólya-urn cluster reassignment** — for each feature with $Z_i = 1$,
   resample $L_i$ from the PY predictive (see
   [Pólya urn primer](../math/polya_urn.md)).
3. **Atom NUTS** (`atom_dispatch.py` → `nuts_atoms.py`) — for each
   cluster, run NumPyro NUTS to update its copula parameters.
4. **Atomic-type IM** (`type_update.py`) — propose a copula-family
   switch for each cluster via independence-Metropolis.
5. **Bernstein-Dirichlet conjugate update** (`dirichlet_weights.py`).
6. **Bernstein-degree update** (`degree_prior.py`).
7. **PY hyperparameter NUTS** (`py_hyperparam_update.py`).
8. **Mass update** (`pi_update.py`) — conjugate Beta-Bernoulli.

The T = 1 fast path (`sweep_simple.py`) implements steps 1, 3 (single
atom), 5, 6, 8 — the rest are no-ops because there is one cluster
that is by construction Gaussian.

## How `run_chain.py` orchestrates

`run_chain_simple(F0, M, key, n_warmup, n_samples, nuts_warmup)`:

1. Build `SimpleSweepInputs` from `F0, M, key`.
2. Build `initial_simple_state(F0, M, key)`.
3. Run `n_warmup` warmup sweeps (discarded).
4. Run `n_samples` retained sweeps; on each, append the state
   to a list of "samples."
5. Pack the samples into an `arviz.InferenceData` via
   `arviz.from_dict({"posterior": {...}})`.
6. Stack the per-sweep $Z$ vectors into `z_samples` of shape
   `(n_samples, n)` and attach to the result.

`run_multi_chain_simple` parallelises across `num_chains` and stacks
the `z_samples` to shape `(num_chains, n_samples, n)`. The multi-chain
arviz `InferenceData` then has the standard `(chain, draw, dim)` axis
order.

## Where each test marker lives

| Marker        | Convention                                                              |
|---------------|-------------------------------------------------------------------------|
| `unit`        | Pure-function test, no chain. Fast.                                     |
| `property`    | Hypothesis-driven property test.                                        |
| `decision`    | Sun-Cai / local-idr / Bayes mFDR tests.                                 |
| `integration` | End-to-end with a tiny chain (few seconds).                             |
| `slow`        | Multi-second chain. Skip during normal dev cycle.                       |
| `smoke`       | Top-level "everything imports and basic constructors run" tests.        |
| `benchmark`   | `pytest-benchmark` micro-benchmarks; skipped unless explicitly invoked. |

The fast development cycle is:

```bash
JAX_PLATFORMS=cpu pytest -m "smoke or unit or property or decision" -q
```

This runs ~200 tests in 30-40 seconds on a Mac M4.

## Adding a new copula family

The path of least resistance:

1. Add `py_idr/copulas/<myname>.py` with the same surface as the
   existing families: `<name>_copula_log_pdf(U, theta)`, a
   `draw(rng, n, theta)`, and a small dataclass holding the parameters.
2. Register it in `py_idr/copulas/registry.py`.
3. Add NUTS support in `py_idr/inference/mcmc/nuts_atoms.py` — wire up
   a NumPyro model that exposes the prior and likelihood.
4. Update `py_idr/inference/mcmc/type_update.py`'s
   `supported_types_for_K` and `prior_propose_atom`.
5. Add a test under `tests/copulas/` exercising density, CDF, and a
   small NUTS-only sanity check.

The independence-Metropolis design means the new family can be tested
in isolation; the type-update step does not need any tuning to
incorporate it.

## Adding a new simulation regime

See the [Simulation regimes guide](../guides/simulation_regimes.md).
The pattern is small enough to fit on one screen: simulator function +
registry entry + thin wrapper + tests.
