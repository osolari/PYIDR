# Plan 04b — Chain inner-loop `jax.lax.scan` refactor

**Owner:** W11
**Status:** scaffolding only (W10.6 lays out the design; the refactor itself is W11+)
**Motivation:** the GPU production sweep on `omids@a10-dev` (4× A10G) was 16×
slower per-iter than local CPU, and 99% of the wall time was idle
`futex` waits between CUDA kernel launches. The bottleneck is **not** GPU
compute — at $n=2000$ each NUTS forward pass is microseconds — but the
Python-then-NumPyro orchestration overhead between sweeps:

1. `run_chain_simple` / `run_chain_multi` run the outer warmup+samples
   loop in plain Python.
2. Each iteration calls `sweep_simple` / `multi_cluster_sweep`, which
   call `nuts_atom_update`.
3. `nuts_atom_update` builds a fresh `numpyro.infer.MCMC(NUTS(...))`
   object **per sweep** and calls `.run(...)`. That bring-up cost is
   amortised internally over the per-sweep `num_warmup + num_samples`
   but **not** across sweeps. With our `nuts_warmup=20, num_samples=1`,
   the bring-up cost dominates.
4. Each iteration also reads JAX device tensors back to host via
   `int(jnp.sum(...))` and similar — every read triggers a synchronous
   device→host transfer.

The empirical signature: GPU utilisation ~0–2%, Python at 99% CPU, no
single op slow but the loop slow overall. Indistinguishable from a
multi-process deadlock except on `strace` (the parallel sweep hit a
real deadlock; the sequential sweep is just slow).

## Design goal

Make the **post-warmup** portion of one cell run in a single `jax.jit`
call so the per-iter Python overhead becomes amortised. Target: ≥
10× speedup on GPU at $n=2000$, $\text{warmup}+\text{samples}=300$,
$\text{chains}=4$.

## What needs to change

The current Gibbs sweep (Algorithm 1) is **almost** JAX-traceable but
has three hot spots:

### Hot spot 1: data-dependent control flow in `sweep_simple`

```python
n_reproducible = int(jnp.sum(Z_new))     # device→host sync
if n_reproducible > 0:
    rho_new, _ = update_exchangeable_gaussian(...)
else:
    rho_new = state.rho                  # no update
```

The `int()` call forces a sync. The Python `if` breaks tracing.

**Fix:** replace with `jax.lax.cond` and a vectorised "compact reproducible features" step that always evaluates the NUTS update on a fixed-size buffer.

### Hot spot 2: NumPyro `MCMC.run()` per sweep

`update_exchangeable_gaussian` constructs `MCMC(NUTS(model), n_warmup=W, n_samples=1).run(...)` every sweep. NumPyro builds a fresh state, traces the model, JIT-compiles, then runs $W+1$ leapfrog steps. The per-sweep bring-up is a substantial fraction of the per-sweep wall time.

**Fix options:** (a) call `numpyro.infer.util.initialize_model` once outside the scan, pass the JIT-compiled `kernel_fn` into the scan body, and step it manually inside the scan; (b) drop NumPyro for the atom update and hand-roll a JAX-native NUTS or HMC; (c) use BlackJAX (which exposes scan-friendly kernel building blocks).

Recommendation: (a) — keep NumPyro for correctness (we already trust its NUTS) but reach into the internals to expose the scan-step. BlackJAX is cleaner but adds a dependency.

### Hot spot 3: multi-cluster Pólya-urn step

`_polya_urn_reassign_all` is a per-feature `for i in range(n)` loop with data-dependent updates to `state.counts`. This is genuinely sequential — feature $i$'s reassignment uses the counts after feature $i-1$'s update. A direct scan over features is possible but `jax.lax.scan` with a string-indexed pytree (the per-cluster atoms) is awkward.

**Fix:** carry a fixed-size buffer of `T_max` cluster slots; mark inactive slots with a sentinel. Reassign features in a scan whose body updates the buffer.

## Scope of the refactor (W11)

Land in three commits:

1. **W11.1**: refactor `sweep_simple` (T=1 path only) to be fully
   JAX-traceable. Cell-level benchmark shows the speedup.
2. **W11.2**: refactor `multi_cluster_sweep` (T>1 path) — harder because
   of the variable-T cluster count. Use a `T_max`-sized buffer.
3. **W11.3**: rewire `run_chain_simple` / `run_chain_multi` to call
   `jax.lax.scan` over the post-warmup iterations. Warmup stays in
   Python (we adapt step size during warmup; that's hard to scan).

## Risks

- NumPyro internals are not API-stable. Reaching into
  `initialize_model` and the NUTS kernel may break across NumPyro
  releases. Add a pin in `pyproject.toml`.
- The `T_max`-buffer approach wastes some compute on empty cluster
  slots. Acceptable if `T_max` stays small (≤ 10 as in the current
  default).
- Reproducibility: the W8.11 divergence plumbing relies on a per-sweep
  diagnostics dict. Inside a scan we'd accumulate divergences as part
  of the carried state. Doable but the existing tests need updating.

## Acceptance criteria

- Per-cell wall time on `omids@a10-dev` at $K=2$, $n=2000$, $R=25$,
  `chains=4`, `n_warmup=100`, `n_samples=200` drops from ~2.5h to
  ≤ 15 min.
- The full S1–S5 production sweep at $R=100$ completes in ≤ 4h wall
  on a single A10G.
- Fast suite still passes (the scan-rewritten `sweep_simple` is
  numerically equivalent to the Python-loop version at fixed seed).
- Divergence count and split-$\widehat R$ unchanged.

## Why not now

This is a 3–5 day refactor with real risk of breaking the chain. The
W9 submission ships with a methodology-verification slice (CPU sweep
at $K=2$, $n=100$, $R=10$); the production-scale cells (W10.7) need
this refactor first. Doing it carelessly is worse than not doing it.
