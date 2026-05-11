# Concepts

This page is a map from the moving parts of the model to where each one
lives in the code. It is meant as a reference you keep open while reading
the [math primers](../math/idr_recap.md) — every formula has a corresponding
module.

## The PY-IDR model in one paragraph

For each feature $i \in \{1, \dots, n\}$ we observe a $K$-vector of
replicate scores $X_i \in \mathbb R^K$. A latent indicator
$Z_i \in \{0, 1\}$ says whether the feature is *reproducible*. Conditional
on $Z_i = 0$ the score vector is drawn from the **independence copula**
(replicate scores are independent given their marginals). Conditional on
$Z_i = 1$ the score vector is drawn from one of $T$ **copula clusters**
whose number $T$ is *unknown* and inferred via a Pitman-Yor stick-breaking
prior. Each cluster carries a copula family and parameters (Gaussian
correlation $R$, Clayton $\theta$, etc.). Marginals are modelled by
Bernstein-Dirichlet polynomials, independently per replicate, so the
chain learns the marginal shape rather than relying on the input rank
transform.

The reproducible mass $\pi^* = \PP(Z = 1)$ is a Beta-prior scalar. The
Pitman-Yor hyperparameters $(\alpha, \sigma)$ are themselves given priors
and updated via NUTS. All of this fits in **one** Gibbs sweep —
Algorithm 1 in §3 of the report.

## Module map

| Module                                         | Role                                                                  |
|------------------------------------------------|-----------------------------------------------------------------------|
| [`py_idr.simulation.scenarios`][]              | Synthetic data generators (S1–S4 today, S5 next).                     |
| [`py_idr.marginals.transform`][]               | The empirical-rank pilot CDF $F_0$.                                   |
| [`py_idr.marginals.bernstein`][]               | Bernstein polynomial evaluation on the unit interval.                 |
| [`py_idr.marginals.dirichlet_weights`][]       | Conjugate Dirichlet update for the Bernstein weights.                 |
| [`py_idr.marginals.pilot`][]                   | Bernstein-Dirichlet marginal model glued to the pilot CDF.            |
| [`py_idr.marginals.degree_prior`][]            | Truncated geometric prior on the Bernstein degree $M$.                |
| [`py_idr.copulas.gaussian`][]                  | $K$-variate Gaussian copula density + draw.                           |
| [`py_idr.copulas.clayton`][]                   | Clayton (lower-tail) copula.                                          |
| [`py_idr.copulas.gumbel`][]                    | Gumbel (upper-tail) copula — $K = 2$ today.                           |
| [`py_idr.copulas.student_t`][]                 | $t_\nu$ elliptical copula.                                            |
| [`py_idr.copulas.independence`][]              | The Uniform-on-$[0,1]^K$ irreproducible component.                    |
| [`py_idr.copulas.registry`][]                  | Lookup table mapping family labels → constructors.                    |
| [`py_idr.pym.stickbreak`][]                    | Pitman-Yor stick-breaking weights.                                    |
| [`py_idr.pym.polya_urn`][]                     | Pólya urn predictive used in cluster reassignment.                    |
| [`py_idr.pym.eppf`][]                          | Pitman-Yor exchangeable partition probability function (EPPF).        |
| [`py_idr.algebra.plod`][]                      | Positive-lower-orthant-dependence projection on correlation matrices. |
| [`py_idr.algebra.correlation`][]               | Helpers for Cholesky + correlation reparameterisation.                |
| [`py_idr.inference.mcmc.sweep_simple`][]       | Single-cluster ($T = 1$) Gibbs sweep — the fast path.                 |
| [`py_idr.inference.mcmc.sweep_multi`][]        | Full Algorithm 1 with Pólya-urn cluster reassignment.                 |
| [`py_idr.inference.mcmc.nuts_atoms`][]         | NUTS update for cluster atom parameters via NumPyro.                  |
| [`py_idr.inference.mcmc.type_update`][]        | Atomic-type independence-Metropolis proposal across copula families.  |
| [`py_idr.inference.mcmc.py_hyperparam_update`][]| NUTS update on the PY EPPF for $(\alpha, \sigma)$.                    |
| [`py_idr.inference.mcmc.class_assign`][]       | Step 1 of the sweep — $Z_i$ resampling given everything else.         |
| [`py_idr.inference.mcmc.pi_update`][]          | Conjugate Beta-Bernoulli update for $\pi$.                            |
| [`py_idr.inference.mcmc.run_chain`][]          | Top-level chain driver; assembles arviz `InferenceData`.              |
| [`py_idr.decision.local_idr`][]                | Posterior local idr = $\PP(Z_i = 0 \mid \text{data})$.                |
| [`py_idr.decision.sun_cai`][]                  | Sun-Cai step-up FDR rule.                                             |
| [`py_idr.decision.bayes_mfdr`][]               | Bayesian marginal FDR + posterior expected discovery count.           |
| [`py_idr.eval.diagnostics_report`][]           | Bundled split-$\hat R$ / ESS / divergence diagnostic verdict.         |
| [`py_idr.data.schema`][]                       | Pydantic `DatasetManifest`, `RunManifest`, `SeedLedger`.              |
| [`py_idr.utils.run`][]                         | `new_run_id`, `now_utc`, `build_run_manifest` helpers.                |

## T = 1 vs T > 1

There are **two chain paths** that share most of the same model:

- **T = 1 path** (`sweep_simple` + `run_chain_simple` /
  `run_multi_chain_simple`). The reproducible component is *forced* to be
  a single Gaussian copula. Cheaper, more stable, and provably correct
  when the data really were generated by one copula. This is the right
  choice for S1–S4 because they are by construction single-copula.

- **T > 1 path** (`sweep_multi` + `run_chain_multi`). The Pitman-Yor
  mixture is active; the chain re-samples cluster assignments at every
  sweep via the Pólya urn predictive, dispatches per-atom NUTS via
  `nuts_atoms.py`, and proposes copula-family swaps via the
  independence-Metropolis step in `type_update.py`. Slower but necessary
  when the reproducible mass is *not* well-modelled by one Gaussian
  copula — heavy-tail dependence, asymmetry, mixture of two scales, etc.
  Regime S5 in the simulation grid exercises this path.

A useful rule of thumb: start with `T = 1`. If the chain diagnostics
flag systematic posterior misfit on the upper tail of the rank
distribution, switch to `T > 1`. The full grid in §4.1 of the report runs
both and reports the comparison.

## The Z-trace

The chain stores, at every retained draw, the full vector
$Z^{(t)} \in \{0, 1\}^n$. After the run we estimate the posterior local
idr by

$$
\widehat{\mathrm{idr}}_i = 1 - \frac{1}{S}\sum_{t=1}^{S} Z_i^{(t)},
$$

where $S$ is the total number of draws (across chains). This is the
*proper* Monte Carlo estimator. It avoids the bias of computing local idr
from the posterior mean of $\pi$ and the cluster atoms, which
double-counts the uncertainty.

The Z-trace lives at `ChainResult.z_samples` with shape
`(n_samples, n)` (single-chain) or `(num_chains, n_samples, n)`
(multi-chain). `decision.local_idr.from_mcmc` flattens chains and
returns the per-feature idr.

## What is and is not in this version

| Component                                   | Implemented           | Notes                                     |
|---------------------------------------------|-----------------------|-------------------------------------------|
| Simulations S1, S2, S3, S4                  | ✅                    | `py_idr.simulation.scenarios`             |
| Simulation S5 (heavy-tail mixture)          | ❌ (next)             | Plan 07 phase II                          |
| T = 1 fast path (Gaussian copula)           | ✅                    | `sweep_simple`, `run_chain_simple`        |
| T > 1 path (multi-cluster Pólya urn)        | ✅                    | `sweep_multi`, atom dispatch              |
| NUTS for atoms                              | ✅                    | NumPyro                                   |
| NUTS for $(\alpha, \sigma)$                 | ✅                    | `py_hyperparam_update`                    |
| Atomic-type IM (Gaussian/Clayton/Gumbel/$t$)| ✅                    | `type_update`; Gumbel restricted to $K=2$ |
| Bernstein-Dirichlet marginals               | ✅                    | `marginals.bernstein`, `dirichlet_weights`|
| Sun-Cai step-up rule                        | ✅                    | `decision.sun_cai`                        |
| Bayesian mFDR + expected discoveries        | ✅                    | `decision.bayes_mfdr`                     |
| Variational inference (mean-field NF)       | ❌ (roadmap)          | Plan 09                                   |
| Real-data F1 (ENCODE)                       | 🟡 (manifest only)    | Plan 11                                   |
| Verdict ledger / sweep CSV writer           | 🟡 (sweep only)       | Plan 13                                   |
