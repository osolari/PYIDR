"""Single-cluster (T=1) Algorithm 1 sweep — the W2 minimum viable sampler.

This wires the four per-step updates that are independent of cluster structure:

1. Step 1 — class assignment $Z_i \\sim \\mathrm{Bernoulli}(\\widehat\\pi_i)$
2. Step 3 — atom-parameter NUTS update (single exchangeable Gaussian atom)
3. Step 5 — per-replicate Bernstein-Dirichlet weight conjugate update
4. Step 8 — $\\pi$ update

Steps that require multi-cluster bookkeeping (Pólya-urn cluster reassignment,
atomic-type IM, $(\\alpha, \\sigma)$ updates on a non-trivial partition) are deferred
to a follow-up. With $T = 1$ the PY partition is trivial, so $(\\alpha, \\sigma)$
are unidentifiable and we hold them fixed at the prior mode.

This minimum-viable sweep matches the **Gaussian-only ablation** of §4.6 and is
already enough to fit a PY-IDR mixture-of-two-components problem on small data.

See plans/04_inference_mcmc.md.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Integer

from py_idr.copulas.gaussian import GaussianCopula
from py_idr.inference.mcmc.class_assign import class_assign
from py_idr.inference.mcmc.nuts_atoms import update_exchangeable_gaussian
from py_idr.inference.mcmc.pi_update import update_pi
from py_idr.marginals.bernstein import cdf as bernstein_cdf
from py_idr.marginals.dirichlet_weights import sample_latent_S, update_weights


@dataclass(frozen=True)
class SimpleSweepState:
    """Minimal sampler state for the $T=1$ sweep.

    Attributes
    ----------
    pi
        Scalar mixing proportion.
    rho
        Scalar exchangeable-Gaussian correlation for the single atom.
    bernstein_weights
        Per-replicate Bernstein-Dirichlet weights; shape ``(K, M)``.
    Z
        Class indicators in $\\{0, 1\\}$; length $n$.

    Notes
    -----
    The Bernstein degree ``M`` and the per-replicate pilot-CDF values are
    fixed at run start and passed through ``Inputs`` rather than carried in
    state, since they are constants of the dataset.
    """

    pi: float
    rho: float
    bernstein_weights: Float[Array, "K M"]
    Z: Integer[Array, "n"]


@dataclass(frozen=True)
class SimpleSweepInputs:
    """Static inputs for the $T=1$ sweep.

    Attributes
    ----------
    F0
        Pre-computed pilot CDF values $F_j^{(0)}(X_{i,j})$; shape $(n, K)$.
        Fixed for the duration of the chain.
    M
        Bernstein degree (held fixed at run start in this minimal sweep).
    alpha_F
        Bernstein smoothing hyperparameter (held fixed).
    nuts_warmup
        NUTS warmup count for the atom step (default 50; the chain runs many
        such mini-NUTS calls so each must be cheap).
    """

    F0: Float[Array, "n K"]
    M: int
    alpha_F: float = 1.0
    nuts_warmup: int = 50


def _bernstein_U(
    F0: Float[Array, "n K"],
    weights: Float[Array, "K M"],
    M: int,
) -> Float[Array, "n K"]:
    """Apply each replicate's current Bernstein-Dirichlet CDF to its pilot values.

    Returns the $U \\in (0, 1)^{n \\times K}$ matrix consumed by the copula
    log-density. Clipping at $[10^{-6}, 1 - 10^{-6}]$ matches the convention
    used by :mod:`py_idr.marginals.transform`.
    """
    K = F0.shape[1]
    Us = [bernstein_cdf(F0[:, j], weights[j], M) for j in range(K)]
    U = jnp.stack(Us, axis=1)
    return jnp.clip(U, 1.0e-6, 1.0 - 1.0e-6)


def sweep_simple(
    state: SimpleSweepState,
    inputs: SimpleSweepInputs,
    key: jax.Array,
) -> SimpleSweepState:
    """One full $T=1$ sweep over class assignment, atom NUTS, Bernstein, $\\pi$.

    Parameters
    ----------
    state
        Current sampler state.
    inputs
        Fixed-dataset inputs (pilot CDF, Bernstein degree, smoothing).
    key
        JAX PRNG key. Internally split into per-step subkeys.

    Returns
    -------
    The new state after one sweep.
    """
    K = inputs.F0.shape[1]
    n = inputs.F0.shape[0]

    # Build per-step PRNG subkeys with explicit step names so a future seed
    # ledger can record them (plan 13 SeedLedger.derivations).
    k_class, k_atom, k_pi, *k_marginals = jax.random.split(key, 3 + 2 * K)
    k_S_per_j = k_marginals[:K]
    k_w_per_j = k_marginals[K:]

    # ---- compute U from the current Bernstein marginals ------------------------
    U = _bernstein_U(inputs.F0, state.bernstein_weights, inputs.M)

    # ---- Step 1: class assignment ---------------------------------------------
    R = (1.0 - state.rho) * jnp.eye(K) + state.rho * jnp.ones((K, K))
    L = jnp.linalg.cholesky(R)
    log_c1 = GaussianCopula(L).log_density(U)
    Z_new = class_assign(k_class, state.pi, log_c1)

    # ---- Step 3: atom NUTS update on the reproducible features ---------------
    # If no features are reproducible at this iteration, leave rho unchanged
    # (the NUTS step is undefined on an empty data set).
    reproducible_idx = jnp.where(Z_new == 1, size=n, fill_value=-1)[0]
    n_reproducible = int(jnp.sum(Z_new))
    if n_reproducible > 0:
        U_rep = U[reproducible_idx[:n_reproducible]]
        rho_new, _ = update_exchangeable_gaussian(
            k_atom,
            U_rep,
            rho_init=state.rho,
            n_warmup=inputs.nuts_warmup,
            n_samples=1,
        )
    else:
        rho_new = state.rho

    # ---- Step 5: Bernstein-Dirichlet conjugate update per replicate -----------
    new_weights_rows: list[Float[Array, "M"]] = []
    for j in range(K):
        S_j = sample_latent_S(
            k_S_per_j[j],
            inputs.F0[:, j],
            state.bernstein_weights[j],
            inputs.M,
        )
        w_j = update_weights(k_w_per_j[j], S_j, inputs.M, inputs.alpha_F)
        new_weights_rows.append(w_j)
    bernstein_weights_new = jnp.stack(new_weights_rows, axis=0)

    # ---- Step 8: π update -----------------------------------------------------
    pi_new = float(update_pi(k_pi, Z_new))

    return SimpleSweepState(
        pi=pi_new,
        rho=rho_new,
        bernstein_weights=bernstein_weights_new,
        Z=Z_new,
    )


def initial_simple_state(
    *,
    n: int,
    K: int,
    M: int,
    pi_init: float = 0.5,
    rho_init: float = 0.3,
) -> SimpleSweepState:
    """Construct a sensible default initial state.

    Bernstein weights are uniform on the first ``M`` columns (corresponds to a
    uniform marginal model — the pilot CDF is the active marginal at iteration 0).
    Z is initialised to ones (every feature is reproducible) so the first atom
    NUTS step has a non-trivial dataset.
    """
    if not (0.0 < pi_init < 1.0):
        raise ValueError(f"pi_init must be in (0, 1); got {pi_init}")
    if not (0.0 < rho_init < 1.0):
        raise ValueError(f"rho_init must be in (0, 1); got {rho_init}")
    if M <= 0:
        raise ValueError(f"M must be positive; got {M}")
    weights = jnp.ones((K, M)) / M
    return SimpleSweepState(
        pi=pi_init,
        rho=rho_init,
        bernstein_weights=weights,
        Z=jnp.ones(n, dtype=jnp.int32),
    )
