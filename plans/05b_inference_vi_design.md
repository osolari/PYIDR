# Plan 05b — VI back-end design

**Owner:** W12
**Status:** scaffolding only (W10.8 lays out the design; full implementation is W12+)
**Refines:** plan 05 (which was a one-line TODO).

## Why VI

The MCMC back-end is the default for $n \lesssim 10^4$ where posterior
uncertainty matters. For genomic-scale studies ($n \in [10^5, 10^7]$)
the MCMC walltime is prohibitive; the report (§3.5.2) commits to a
finite-truncation structured mean-field VI alternative.

The "PY-IDR (VI)" row in `tab:sim-fdr` and `tab:sim-power` is currently
marked `---` (not yet integrated). This plan is the bridge.

## Variational family

Following the structured mean-field family in §3.5.2 (Hoffman, Blei,
Wang, Paisley 2013 framework):

$$
q(Z, z, \theta, t, V, \pi, F, \alpha, \sigma) =
\prod_i q(Z_i) \prod_i q(z_i) \prod_{m \le T} q(\theta_m) q(t_m) q(V_m)
\cdot q(\pi) \prod_j q(F_j) q(\alpha) q(\sigma)
$$

with $T < \infty$ the truncation level.

Per-component variational distributions:

- $q(Z_i)$, $q(z_i)$: categorical (Concrete relaxation during training).
- $q(\theta_m)$: Gaussian on the unconstrained correlation parameterisation
  (Cholesky for the exchangeable / one-factor / two-factor classes).
- $q(t_m)$: categorical over $\{\text{Gauss}, t_5, \text{Clayton}, \text{Gumbel}\}$.
- $q(V_m)$: Beta on the stick-breaking ratios.
- $q(\pi)$: Beta.
- $q(F_j)$: Dirichlet on Bernstein weights (conjugate; closed-form).
- $q(\alpha)$: log-normal on $\R_+$.
- $q(\sigma)$: logit-normal on $[0, 1)$.

## ELBO

$$
\mathcal{L}(q) = \E_q[\log p(\mathbf{X}, Z, z, \theta, t, V, \pi, F, \alpha, \sigma)] - \E_q[\log q(\cdot)]
$$

Estimated by reparameterised Monte Carlo on the continuous parameters
plus pathwise / score-function estimators on the discrete $(Z_i, z_i, t_m)$
draws (Concrete relaxation is the cleanest path; Gumbel-Softmax for
$t_m$).

## Implementation surface

```
src/py_idr/inference/vi/
    family.py        # parameterise each q(·); pytree-friendly
    elbo.py          # build the ELBO + estimators
    runner.py        # run_chain_vi(...) → ChainResult-compatible output
    cli_hook.py      # wire `py-idr fit --inference vi` once ready
```

### `run_chain_vi`

Same signature shape as `run_chain_simple`:

```python
def run_chain_vi(
    *,
    F0: Float[Array, "n K"],
    M: int,
    key: jax.Array,
    T_max: int = 10,
    n_steps: int = 2000,
    batch_size: int = 256,
    lr: float = 1e-3,
    annealing_steps: int = 500,
) -> ChainResult:
    ...
```

Returns a `ChainResult` whose `idata` posterior group holds samples
**drawn from the variational distribution after training** rather than
MCMC draws. That way `from_mcmc` works unchanged: it averages the
class-indicator samples to get the local idr.

### Per-iteration step

```
state ← init from prior
for t in range(n_steps):
    minibatch ← sample(F0, batch_size)
    elbo, grads ← jax.value_and_grad(elbo_fn)(state, minibatch, key_t)
    state ← optax.adam(state, grads)
```

Run inside a `jax.lax.scan` over the training steps — this is
straightforward to vectorise because there's no Gibbs structure to
preserve.

## Theory hooks

The report's Theorem 3.5 (finite-truncation variational consistency)
commits to:

- At fixed $T$, the variational predictive density is Hellinger
  consistent if the variational family contains distributions within
  $o(n \epsilon_n^2)$ KL divergence of the truncated posterior.
- Matching the MCMC rate as $T \to \infty$ is an additional assumption
  (residual stick mass + structured mean-field projection).

The VI back-end's success criterion at production scale is the
"PY-IDR (VI)" row's calibration / power matching MCMC within a
documented slack — exactly the kind of validation the §4 experiments
section is set up for once the VI runner exists.

## Scope of the build-out (W12)

Three commits:

1. **W12.1**: `family.py` — parameterise each $q$, pytree-friendly,
   no ELBO yet. Unit tests for shape contracts.
2. **W12.2**: `elbo.py` + the score-function / pathwise estimator
   plumbing. Unit tests verify the ELBO on a toy fully-Gaussian
   special case matches the closed-form KL.
3. **W12.3**: `runner.py` end-to-end + a smoke test that the VI fit
   recovers $(\pi, \rho)$ on regime S1 within tolerance. Wire into
   `py-idr sim --method vi` and into `_REPORT_METHODS_ORDER`.

## Risks

- Concrete relaxation can have biased gradients near the discrete
  boundary; the discrete $(Z_i)$ trace decoded from $q$ at evaluation
  time must use `argmax`, not the relaxed soft assignment.
- The Pitman-Yor stick is fiddly: $V_m$'s Beta prior depends on
  $(\alpha, \sigma)$ which are themselves variational. Use a
  hierarchical reparameterisation; do not differentiate through
  finite-difference sampling.
- Compositional convergence: the joint $(Z, z, t)$ posterior is
  multi-modal (label-switching at the cluster level). VI tends to
  collapse to a single mode; this is a known limitation and is
  acceptable as long as the local idr (which is label-invariant)
  matches MCMC.

## Acceptance criteria

- `run_chain_vi` on regime S1, $K=2$, $n=2000$, $T_{\max}=4$:
  $\widehat \pi_{\rm VI} \in [\pi^* - 0.05, \pi^* + 0.05]$;
  $\widehat \rho_{\rm VI} \in [\rho^* - 0.05, \rho^* + 0.05]$.
- The PY-IDR (VI) row of `tab:sim-fdr` / `tab:sim-power` shows a
  realised FDR within $\pm 0.01$ of the PY-IDR (MCMC) row at $\alpha = 0.05$.
- Per-cell wall time at $n=10^4$ is < 10% of the MCMC equivalent at
  the same $n$.

## Why not now

The VI design is well-understood and not blocked on external work, but
a faithful implementation is multi-day work that benefits from being
done after the MCMC scan refactor (W11 / plan 04b) — both share the
same `state` pytree style and the same `optax` + `lax.scan`
infrastructure.
