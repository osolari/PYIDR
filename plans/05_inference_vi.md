# Plan 05 — Inference: Structured Mean-Field VI

**Owner:** W4
**Depends on:** 04_inference_mcmc (uses the same `SamplerState` shape)
**Implements:** §3.3.2 of the revised report; Theorem 3.5 (**finite-truncation** variational consistency).

## Status snapshot (W8.1)

**TODO.** `py_idr.inference.vi` is still a stub package. The MCMC back-end is the only working inference path. This is the largest remaining piece of methodology work; everything else in the §3–§4 stack is operational.

## Goal

A structured mean-field variational family with **finite truncation** $T$ at the PY stick,
optimised by stochastic natural-gradient ascent (Hoffman, Blei, Wang & Paisley 2013). Default
truncation $T = 30$, mini-batch $B \in \{500, 2000\}$, 200 epochs.

> **Theorem 3.5 scope.** The revised manuscript states VI consistency for **fixed** finite
> truncation: under the truncated model, the variational posterior predictive density is
> Hellinger consistent if the structured mean-field family contains distributions within
> $o(n\epsilon_n^2)$ KL of the truncated posterior. Matching the Theorem 3.2 rate as
> $T = T_n \to \infty$ requires additional truncation-error and variational-approximation
> assumptions and is not asserted as a completed result for the unrestricted PY-copula
> model. Tests in this plan target the fixed-$T$ statement.

## Deliverables

### `src/py_idr/inference/vi/`

- `family.py` — variational distributions per §3.3.2 of the revised report:
  - $q(Z_i)$, $q(z_i)$ — categorical (vectorised)
  - $q(\theta_m)$ — Gaussian or transformed-Gaussian on the unconstrained atomic parameters (Cholesky for Gaussian/$t_5$ atoms; $\log\theta$ for Clayton/Gumbel)
  - $q(t_m)$ — categorical over the four atomic types
  - $q(V_m)$ — independent $\Beta$ on each stick-breaking ratio for $m \le T$ (per-cluster, not a joint $q(w)$ — this matches the revised manuscript's family)
  - $q(\pi)$ — $\Beta$
  - $q(F_j)$ — Dirichlet on Bernstein weights
  - $q(\alpha)$ — Gamma; $q(\sigma)$ — truncated Beta on $[0, 1)$
- `elbo.py` — composite ELBO with truncated stick. Closed-form expectations where possible (Beta entropies, Dirichlet expectations) and monte-carlo for the copula log-density expectations.
- `natgrad.py` — natural-gradient updates on conjugate blocks (Bernstein weights, $\pi$); reparam-gradient SGD on the others; Robbins–Monro learning-rate schedule with $\rho_t = (t + \tau)^{-\kappa}$, $\kappa \in (0.5, 1]$.
- `reparam.py` — reparam tricks for the continuous blocks (Cholesky free entries, $\log\alpha$, $\log\alpha_F$).
- `minibatch.py` — stochastic data passes; ensures $\E[\nabla\widetilde\ELBO] = \nabla\ELBO$ via the stochastic VI estimator of HBWP.

## Tests

- `tests/inference/test_vi_recover_pi.py` — fits VI to a small synthetic S1 cell; recovers $\pi^*$ within 0.02.
- `tests/inference/test_vi_truncation.py` — at $T = 1$ (collapsed to a single component), VI ELBO equals the closed-form expression for the bivariate Gaussian-copula model.
- `tests/integration/test_vi_matches_mcmc_means.py` — on a small problem, posterior means under VI agree with MCMC posterior means within tolerance (qualitative).

## Acceptance criteria

- VI per-epoch cost scales linearly in $n$ (verified empirically across $n \in \{1k, 10k, 100k\}$).
- VI default config trains stably (no NaN ELBO) on 100% of the simulation cells in a 10-cell smoke sweep.
- The truncated-VI ELBO at $T = 30$ is within 5% of the MCMC posterior log-evidence on a problem where MCMC mixed and the AIS-style log-evidence estimate is available.

## Notes

- The discrete $q(t_m)$ over atomic types means the type assignment is fully variational; no Gumbel-softmax surrogate is required because we can analytically marginalise.
- Stochastic natural gradients on conjugate blocks (Bernstein and $\pi$) reduce to the standard SVI updates; for the non-conjugate atom-parameter block we use the reparam-gradient with score-function variance reduction baselines as needed.
- VI is intended for $n \gtrsim 10^5$. For smaller $n$, defer to MCMC and use VI only for warm-starting.
