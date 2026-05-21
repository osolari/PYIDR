<div class="saim-hero" markdown>
  <img src="assets/saim_logo.png" alt="sAIm Labs" class="saim-hero-logo">
  <h1>PY-IDR</h1>
  <p><strong>Hierarchical nonparametric Bayesian IDR via Pitman--Yor copula
  mixtures</strong> — arbitrary replicate count, heterogeneous non-Gaussian
  marginals, and an unknown number of reproducible-component dependence
  regimes, all in one model.</p>
  <p class="saim-hero-badges">
    <a href="https://github.com/osolari/PYIDR">GitHub</a>
    <a href="getting_started/quickstart/">Quickstart</a>
    <a href="getting_started/installation/">Install</a>
    <a href="api/">API</a>
    <a href="report_submission_readiness/">Submission readiness</a>
  </p>
</div>

# PY-IDR

**PY-IDR** is a hierarchical nonparametric Bayesian extension of the
*irreproducible-discovery-rate* (IDR) framework. It generalises the
two-component Gaussian-copula IDR of Li, Brown, Huang, and Bickel (2011) to:

- **A Pitman-Yor mixture over copula clusters**, so the reproducible
  signal does not have to come from a single Gaussian copula. Multiple
  copula *types* (Gaussian, Clayton, Gumbel, $t$) coexist in one model and
  one chain.
- **Bernstein-polynomial Dirichlet marginals**, so per-replicate
  distributional shape is learned alongside the copula instead of being
  baked into the pilot rank transform.
- **A Sun-Cai step-up decision rule** layered on top of the posterior
  *local idr*, giving exact mFDR control at any nominal $\alpha$ — not just
  the heuristic global-IDR threshold of the original method.

The package is a working research codebase that maps directly onto the
revised manuscript in `docs/report/`. Algorithm 1 (the 8-step Gibbs sweep
in §3) is implemented in [`py_idr.inference.mcmc.sweep_multi`][], with the
T = 1 "single-cluster" fast path in [`py_idr.inference.mcmc.sweep_simple`][];
the decision rule lives in [`py_idr.decision.sun_cai`][] and
[`py_idr.decision.local_idr`][]; the simulation grid from §4.1 is reproduced
in [`py_idr.simulation.scenarios`][].

## When you should reach for PY-IDR

You should use PY-IDR when:

- You have **two or more replicates** of the same noisy ranking experiment
  (ChIP-seq, scRNA-seq peak calls, large-scale prediction scores) and want
  to call features that *agree* across replicates with FDR control.
- The reproducible component is **plausibly non-Gaussian** — e.g. heavy-tail
  dependence at the top of the ranking, or asymmetry that the two-Gaussian
  copula cannot capture. PY-IDR's mixture lets multiple copula families
  share posterior mass.
- You care about **posterior uncertainty** on the reproducible mass
  $\pi$, the per-cluster dependence, and the FDR estimate — not just point
  estimates.

You should *not* use PY-IDR (yet) when:

- You only have a single replicate. The model has no signal to work with.
- You need millisecond-latency inference at production scale. The MCMC
  path costs several minutes per dataset; the variational path (§5) is on
  the roadmap.
- Your features are not exchangeable across the replicate axis (e.g.
  paired technical/biological replicates with known batch structure).
  PY-IDR's exchangeability assumption is load-bearing.

## How to read these docs

- **[Getting started](getting_started/installation.md)** — install the
  package, run a tiny end-to-end fit, and learn the few API entry-points
  you will actually call.
- **[Math primers](math/idr_recap.md)** — the conceptual chain from the
  IDR likelihood, through Sklar's theorem and the Pitman-Yor process, to
  the Sun-Cai step-up rule. Read these if you want to *understand* what
  the chain is doing, not just call it. The notation matches the report
  ($\pi^*$ is the reproducible mass, $\rho$ the Gaussian-copula
  correlation, $K$ the replicate count).
- **[Guides](guides/simulation_regimes.md)** — practical questions:
  how big should $K$ be, when ranks beat scores, how to read the chain
  diagnostics, how the seed ledger and manifest format work.
- **[Developer](developer/architecture.md)** — module layout, how to add
  a new copula family or a new simulation regime, the test taxonomy.

## A 30-second tour

```python
import jax
from py_idr.simulation.scenarios import simulate_S1
from py_idr.simulation.replicates import run_replicate_S1

# Synthetic data from regime S1: K=3 replicates, n=200 features,
# 30% reproducible, latent Gaussian copula with rho = 0.85.
sim = simulate_S1(K=3, n=200, seed=0)
print(sim.X.shape, sim.true_pi, sim.true_rho)
# (200, 3) 0.30 0.85

# End-to-end fit + Sun-Cai decision at alpha = 0.05.
row = run_replicate_S1(
    K=3, n=200,
    num_chains=2, n_samples=200,
    seed=0,
)
print(row.pi_posterior_mean, row.realized_fdr, row.power)
```

The `row` object is a flat `ReplicateRow` — one CSV row per run. Sweep
scripts under `scripts/` aggregate many of these into the Table 4
reproducer.

## Citing

If PY-IDR helped your work, please cite the report:

```bibtex
@techreport{pyidr2026,
  title  = {PY-IDR: Hierarchical nonparametric Bayesian IDR via
            Pitman-Yor copula mixtures},
  author = {Solari, Omid and contributors},
  year   = {2026},
}
```
