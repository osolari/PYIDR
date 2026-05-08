---
name: mcmc-diagnostics-review
description: Review split-R-hat, ESS, divergences, and cluster-occupancy traces on a Zarr trace produced by Algorithm 1. Use when the user asks to "check chain mixing", "review MCMC diagnostics", "is the sampler converged", or "investigate divergences".
---

# Skill: MCMC Diagnostics Review

For PY-IDR, "the sampler converged" requires more than the usual split-$\widehat R \le 1.01$
on continuous parameters. Cluster-occupancy traces, atomic-type histograms, and per-cluster
correlation summaries are all part of the diagnostic story.

## When to invoke

Triggered by any of:

- "review MCMC diagnostics for <run-id>"
- "is the chain mixed?"
- "investigate divergences"
- "the trace looks suspicious"
- "show me the cluster-occupancy plot"

Do NOT use this skill to debug Algorithm 8 specifically — use `algorithm8-sanity` for
that.

## Prerequisites

- A Zarr trace at `runs/<run_id>/trace.zarr` containing the post-warmup chain.
- Run config at `runs/<run_id>/config.yaml`.

## Procedure

1. **Load the trace into arviz:**
   ```python
   import stick_idr.inference.mcmc.diagnostics as diag
   ix = diag.load_trace("runs/<run_id>/trace.zarr")
   ```
2. **Continuous-parameter diagnostics.**
   - Split-$\widehat R$ on $\pi$, $\sigma$, $\alpha$, $\alpha_F$: must be $\le 1.01$.
   - ESS on $\pi$: must be $\ge 400$.
   - Divergences across chains: must be $< 0.5\%$ of post-warmup samples.
   - Energy fraction of missing information (E-FMI): must be $\ge 0.3$.
3. **Discrete-parameter diagnostics.**
   - Number of active clusters $T$ trace: visually plateau after burn-in.
   - Atomic-type histogram across post-warmup samples: should not be degenerate
     (i.e., not all probability mass on one type unless the regime is intentionally
     homogeneous like S1).
   - Per-cluster occupancy stability: plot a heatmap of cluster index vs. iteration.
4. **Per-cluster diagnostics.**
   - For each cluster with > 5% of features, plot the off-diagonal trace of its
     correlation matrix. Sudden jumps suggest a label-switching artifact.
5. **Summarise.** Produce a report with:
   - Overall verdict: `converged | mixing-questionable | not-converged`.
   - Per-parameter $\widehat R$ and ESS table.
   - Cluster-occupancy plot (PNG).
   - Atomic-type histogram (PNG).
6. **If diagnostics are bad:** propose a corrective action:
   - High divergences → raise `target_accept_prob` to 0.99 in the NUTS step.
   - Low ESS on $\pi$ → run a longer chain or fix initialization.
   - Cluster-occupancy not plateaued → run more burn-in or check the auxiliary-atom count $H$.
   - Label switching → enable the deterministic relabeling post-processor in
     `inference/mcmc/diagnostics.py`.

## Expected outputs

- `runs/<run_id>/diagnostics/report.json`
- `runs/<run_id>/diagnostics/cluster_occupancy.png`
- `runs/<run_id>/diagnostics/atomic_types.png`
- `runs/<run_id>/diagnostics/cluster_correlations/cluster_<m>.png` per active cluster.

## Failure modes

- **Missing trace.** If `trace.zarr` is empty or partially written, the run was killed
  mid-flight. Don't try to diagnose; resume the run.
- **Cluster-relabeling instability.** Discrete-parameter diagnostics in PY mixtures are
  notoriously sensitive to label switching. The deterministic relabeling helper is
  the canonical fix; do not silently re-order in `report.json` without it.
- **Sampler mismatch.** If `config.yaml` says VI but the trace is Zarr (MCMC-only format),
  the user pointed the skill at the wrong run. Stop and clarify.
