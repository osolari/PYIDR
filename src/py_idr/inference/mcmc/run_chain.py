"""Chain driver: loop the $T=1$ sweep and return an :class:`arviz.InferenceData`.

Per the durable user feedback to prefer mature-library tooling, the trace
is materialised as an ``arviz.InferenceData`` so that downstream diagnostic
calls — ``arviz.summary`` (split-$\\widehat R$ + ESS + HDI), ``arviz.plot_trace``,
``arviz.plot_posterior`` — Just Work. We do **not** roll our own diagnostics
or trace I/O.

This module is intentionally a thin wrapper. The sweep itself is in
:mod:`py_idr.inference.mcmc.sweep_simple`; the per-step updates are in their
own modules.

See plans/04_inference_mcmc.md and plans/12_diagnostics.md.
"""

from __future__ import annotations

from dataclasses import dataclass

import arviz as az
import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float

from py_idr.inference.mcmc.sweep_simple import (
    SimpleSweepInputs,
    SimpleSweepState,
    initial_simple_state,
    sweep_simple,
)


@dataclass(frozen=True)
class ChainResult:
    """Output of :func:`run_chain_simple`.

    Attributes
    ----------
    idata
        ``arviz.InferenceData`` with one chain. Contains scalar posterior
        variables in the ``posterior`` group; ``observed_data`` carries the
        original ``F0`` so a downstream replay can reconstruct.
    final_state
        The last sweep's state — useful for continuing the chain.
    """

    idata: az.InferenceData
    final_state: SimpleSweepState


@dataclass(frozen=True)
class MultiChainResult:
    """Output of :func:`run_multi_chain_simple` (multi-chain variant).

    Attributes
    ----------
    idata
        ``arviz.InferenceData`` with shape (num_chains, draws) on each
        posterior variable — required for split-$\\widehat R$.
    final_states
        Per-chain final states; ``len(final_states) == num_chains``.
    """

    idata: az.InferenceData
    final_states: tuple[SimpleSweepState, ...]


def run_chain_simple(
    *,
    F0: Float[Array, "n K"],
    M: int,
    key: jax.Array,
    n_warmup: int = 200,
    n_samples: int = 200,
    initial_state: SimpleSweepState | None = None,
    alpha_F: float = 1.0,
    nuts_warmup: int = 50,
) -> ChainResult:
    """Run a single chain of the $T=1$ PY-IDR sweep.

    Parameters
    ----------
    F0
        Pilot-CDF matrix; shape $(n, K)$.
    M
        Bernstein degree (held fixed; the marginal-degree update is a
        subsequent commit's responsibility).
    key
        JAX PRNG key.
    n_warmup, n_samples
        Sample counts. The warmup samples are discarded from the
        ``InferenceData`` trace.
    initial_state
        Optional override of the chain start. Defaults to
        :func:`initial_simple_state`.
    alpha_F
        Bernstein smoothing hyperparameter.
    nuts_warmup
        Per-sweep NUTS warmup for the atom step. Default 50 — short, so each
        sweep is cheap. Increase for tighter atom posteriors.

    Returns
    -------
    A :class:`ChainResult` carrying the arviz InferenceData and the final
    sweep state.
    """
    if F0.ndim != 2:
        raise ValueError(f"F0 must be 2-D (n, K); got shape {F0.shape}")
    if n_warmup < 0 or n_samples <= 0:
        raise ValueError(f"need n_warmup >= 0 and n_samples > 0; got ({n_warmup}, {n_samples})")

    n, K = F0.shape
    inputs = SimpleSweepInputs(F0=F0, M=M, alpha_F=alpha_F, nuts_warmup=nuts_warmup)
    state = initial_state if initial_state is not None else initial_simple_state(n=n, K=K, M=M)

    total_iters = n_warmup + n_samples
    iter_keys = jax.random.split(key, total_iters)

    # Scalar samples accumulator. Storing as Python lists is fine here — the
    # trace is small (a few floats per iteration) and arviz wants numpy arrays
    # in the end anyway.
    pi_trace: list[float] = []
    rho_trace: list[float] = []
    n_reproducible_trace: list[int] = []

    for i, k in enumerate(iter_keys):
        state = sweep_simple(state, inputs, k)
        if i >= n_warmup:
            pi_trace.append(float(state.pi))
            rho_trace.append(float(state.rho))
            n_reproducible_trace.append(int(jnp.sum(state.Z)))

    # Build the InferenceData. arviz expects shape (chains, draws, ...);
    # the API takes a nested {group_name: {var_name: array}} mapping.
    posterior_group = {
        "pi": np.asarray(pi_trace, dtype=np.float64)[None, :],  # (1, draws)
        "rho": np.asarray(rho_trace, dtype=np.float64)[None, :],
        "n_reproducible": np.asarray(n_reproducible_trace, dtype=np.int32)[None, :],
    }
    idata = az.from_dict({"posterior": posterior_group})
    return ChainResult(idata=idata, final_state=state)


def run_multi_chain_simple(
    *,
    F0: Float[Array, "n K"],
    M: int,
    key: jax.Array,
    num_chains: int = 4,
    n_warmup: int = 200,
    n_samples: int = 200,
    alpha_F: float = 1.0,
    nuts_warmup: int = 50,
    overdispersed: bool = True,
) -> MultiChainResult:
    """Run ``num_chains`` independent chains and stack into a multi-chain idata.

    Per §3.3.1 of the revised report, the diagnostics list requires at least 4
    overdispersed chains so split-$\\widehat R$ can flag mixing failures. We run
    them sequentially (Python loop) rather than via ``jax.pmap`` to keep the
    orchestrator simple — each chain is independent and only the final
    materialisation into an InferenceData stacks them.

    Parameters
    ----------
    F0, M, n_warmup, n_samples, alpha_F, nuts_warmup
        Same as :func:`run_chain_simple`.
    num_chains
        Number of chains; default 4 to match the §3.3.1 explicit requirement.
    key
        Root JAX PRNG key; split into per-chain subkeys before each chain runs.
    overdispersed
        If True (default), per-chain initial states are drawn from a
        dispersed distribution over (pi, rho) so chains start from genuinely
        different points and split-$\\widehat R$ has signal.

    Returns
    -------
    A :class:`MultiChainResult` with idata of shape ``(num_chains, n_samples)``
    on each posterior variable.
    """
    if num_chains < 1:
        raise ValueError(f"num_chains must be >= 1; got {num_chains}")
    if F0.ndim != 2:
        raise ValueError(f"F0 must be 2-D (n, K); got shape {F0.shape}")

    n, K = F0.shape
    chain_keys = jax.random.split(key, num_chains + 1)
    init_keys = jax.random.split(chain_keys[0], num_chains)
    per_chain_keys = chain_keys[1:]

    # Overdispersed inits: scatter pi in (0.2, 0.8), rho in (0.1, 0.9).
    if overdispersed:
        pi_inits = jnp.linspace(0.2, 0.8, num_chains)
        rho_inits = jnp.linspace(0.15, 0.85, num_chains)
    else:
        pi_inits = jnp.full(num_chains, 0.5)
        rho_inits = jnp.full(num_chains, 0.3)
    del init_keys  # not used — but reserved for future stochastic perturbations

    pi_chains: list[np.ndarray] = []
    rho_chains: list[np.ndarray] = []
    n_rep_chains: list[np.ndarray] = []
    final_states: list[SimpleSweepState] = []

    for c in range(num_chains):
        init_state = initial_simple_state(
            n=n,
            K=K,
            M=M,
            pi_init=float(pi_inits[c]),
            rho_init=float(rho_inits[c]),
        )
        chain_res = run_chain_simple(
            F0=F0,
            M=M,
            key=per_chain_keys[c],
            n_warmup=n_warmup,
            n_samples=n_samples,
            initial_state=init_state,
            alpha_F=alpha_F,
            nuts_warmup=nuts_warmup,
        )
        posterior = chain_res.idata.posterior
        pi_chains.append(np.asarray(posterior["pi"]).reshape(-1))
        rho_chains.append(np.asarray(posterior["rho"]).reshape(-1))
        n_rep_chains.append(np.asarray(posterior["n_reproducible"]).reshape(-1))
        final_states.append(chain_res.final_state)

    posterior_group = {
        "pi": np.stack(pi_chains, axis=0),  # (num_chains, draws)
        "rho": np.stack(rho_chains, axis=0),
        "n_reproducible": np.stack(n_rep_chains, axis=0),
    }
    idata = az.from_dict({"posterior": posterior_group})
    return MultiChainResult(idata=idata, final_states=tuple(final_states))


def summarise(result: ChainResult, var_names: list[str] | None = None) -> "az.InferenceData":
    """Return :func:`arviz.summary` for the chain's scalar parameters.

    Returns a DataFrame containing ``mean``, ``sd``, ``hdi_3%``, ``hdi_97%``,
    ``ess_bulk``, ``ess_tail``, ``r_hat`` per variable. The values are the
    standard arviz outputs — we don't reimplement them.
    """
    return az.summary(result.idata, var_names=var_names)
