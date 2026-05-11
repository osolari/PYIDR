# Reading diagnostics

Every PY-IDR fit produces an `arviz.InferenceData` object plus a
bundled `DiagnosticsReport`. This guide explains what to look at, in
what order, and which thresholds the report uses.

## The 30-second triage

After a chain, run:

```python
from py_idr.eval.diagnostics_report import build_diagnostics_report

report = build_diagnostics_report(result.idata, num_chains=result.num_chains)
print(report.verdict)  # "pass" | "warn" | "fail"
print(report.summary)
```

If `verdict == "pass"`, the chain is healthy; look at the posterior
summaries directly. If `warn`, read the per-variable detail. If `fail`,
re-run with longer warmup or more chains before trusting the posterior.

## The four diagnostics PY-IDR bundles

### 1. Split-$\hat R$ (chain convergence)

For each scalar parameter, $\hat R$ compares within-chain variance to
between-chain variance. The "split" variant chops each chain in half
and treats the halves as independent chains; this catches monotonic
drift that ordinary $\hat R$ misses.

Threshold: $\hat R \le 1.01$. The constant
`RHAT_PASS_THRESHOLD = 1.01` in `diagnostics_report.py` enforces this.

What to do if it fails:

- Longer warmup. The chain has not yet equilibrated.
- More chains. Two chains can luck into agreement; four is the minimum
  for $\hat R$ to be reliable. The report's pass criterion includes
  `num_chains >= MIN_CHAINS_PASS` (= 4).
- Check for multimodality. The cluster-count posterior $T$ is the
  usual suspect — if the chains land on different $T$, $\hat R$ on
  cluster-level summaries will be huge.

### 2. Effective sample size (ESS)

The ESS estimates how many *independent* samples are equivalent to the
observed (autocorrelated) chain. Two flavours are reported:

- **`ess_bulk`** — bulk of the distribution; controls the precision of
  posterior means.
- **`ess_tail`** — quantile estimates; controls the precision of
  posterior intervals.

Threshold: both $\ge 400$. The constant `ESS_PASS_THRESHOLD = 400`.

What to do if it fails:

- More samples. The chain has not run long enough.
- Re-parameterise. If `ess_bulk` is fine but `ess_tail` is bad, the
  chain is mixing in the bulk but stuck in the tail. The PY hyperparam
  $\alpha$ is the usual culprit; switching to a non-centered
  parameterisation can help.

### 3. Divergent transitions

NUTS reports a *divergence* whenever the leapfrog integrator's
Hamiltonian error exceeds a target threshold. A small fraction of
divergences is normal; many indicate the geometry is too sharp for the
current step size.

Threshold: divergence fraction $\le 0.005$ (half a percent). The
constant `DIVERGENCE_FRACTION_THRESHOLD = 0.005`.

What to do if it fails:

- Increase `nuts_warmup` so NumPyro has more time to adapt the step
  size and mass matrix.
- Reparameterise sharp coordinates. The PY discount $\sigma \in [0, 1)$
  is the usual culprit; the chain handles this via a logit
  reparameterisation.

### 4. Cluster-count traceplot

This is *not* a numeric diagnostic but the report attaches the per-chain
trace of $T$ (number of active clusters). Look for:

- Stable across chains: posterior on $T$ is consistent.
- Chain-specific bands: one chain is stuck in $T = 2$ while others are
  in $T = 1$ — multimodality.
- Monotonic growth: the chain is *still finding* clusters at the end of
  the run; you have not converged.

## The `DiagnosticsReport` object

The report is a Pydantic schema with these fields:

```python
class DiagnosticsReport(BaseModel):
    num_chains: int
    num_samples: int
    rhat_max: float
    ess_bulk_min: float
    ess_tail_min: float
    divergence_fraction: float
    cluster_count_mode: int  # most common T across draws
    verdict: Literal["pass", "warn", "fail"]
    failed_checks: list[str]  # names of any thresholds breached
    summary: str  # human-readable one-liner
```

`build_diagnostics_report(idata, num_chains=)` is the canonical
constructor. It does **not** decide which transformations to flag; that
is the user's call. The thresholds above are defaults.

## When to ignore a "fail"

Two cases warrant trusting the posterior despite a `fail`:

1. **You only care about one parameter and *its* diagnostics pass.**
   The report aggregates worst-case across every variable in the
   posterior. If `ess_bulk_min` is the only failing check and it's
   driven by an obscure cluster-level atom that you do not summarise
   downstream, the posterior on $\pi$ and $\rho$ might still be fine.
2. **You are running a smoke test.** Short chains with `n_samples = 20`
   will always fail ESS. The test suite under `tests/integration/`
   uses tiny chains and skips the diagnostic verdict.

## What the report does *not* check

- **Model misspecification.** $\hat R$ can be 1.00 and the model can
  still be wrong. Posterior-predictive checks (plotting the pilot CDF
  against the posterior-predictive CDF per replicate) are the standard
  remedy. PY-IDR does not bundle these yet.
- **Identifiability of cluster labels.** Label-switching makes cluster
  *identity* meaningless across chains. The report only summarises the
  *number* of clusters, not which cluster is which. This is the right
  thing for the local-idr decision pipeline because Sun-Cai operates on
  $\PP(Z_i = 0)$ which is label-invariant.
