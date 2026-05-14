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

from py_idr.inference.mcmc.sweep_multi import multi_cluster_sweep
from py_idr.inference.mcmc.sweep_simple import (
    SimpleSweepInputs,
    SimpleSweepState,
    initial_simple_state,
    sweep_simple,
)
from py_idr.inference.mcmc.type_update import prior_propose_atom
from py_idr.model.multi_state import (
    MultiClusterState,
    cluster_sizes,
    initial_multi_cluster_state,
)
from py_idr.utils.arviz_compat import idata_from_posterior as _idata_from_posterior


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
    z_samples
        Per-feature class indicators across post-warmup draws; shape
        ``(n_samples, n)`` with values in $\\{0, 1\\}$. Consumed by
        :func:`py_idr.decision.local_idr.from_mcmc` to estimate the local idr
        from the Monte Carlo average.
    num_divergent_transitions
        Total NUTS divergent transitions, summed across the per-sweep atom
        updates over the **post-warmup** portion of the chain. Plumbed into
        :func:`py_idr.eval.diagnostics_report.build_diagnostics_report` so
        the §3.3.1 divergence-fraction threshold has a real signal.
    """

    idata: az.InferenceData
    final_state: SimpleSweepState
    z_samples: np.ndarray | None = None
    num_divergent_transitions: int = 0


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
    z_samples
        Stacked per-chain class-indicator traces; shape
        ``(num_chains, n_samples, n)``. Consumed by the proper Monte Carlo
        idr estimator.
    num_divergent_transitions
        Total NUTS divergent transitions summed across **all chains** and
        across the post-warmup portion of each chain. The diagnostics report
        divides this by ``num_chains * num_draws_per_chain`` for the §3.3.1
        divergence-fraction threshold.
    """

    idata: az.InferenceData
    final_states: tuple[SimpleSweepState, ...]
    z_samples: np.ndarray | None = None
    num_divergent_transitions: int = 0


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
    em_init: bool = False,
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
    em_init
        If True (and ``initial_state`` is None), seed the chain via
        :func:`py_idr.inference.mcmc.em_init.compute_em_init` — pairwise
        Vanilla IDR EM on $F_0$ recovers an informed
        $(\\pi, \\rho, Z)$ start. Reduces warmup-budget waste; see the
        report's §3.3.1 and the W8.1 audit notes in the status doc.

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
    if initial_state is not None:
        state = initial_state
    elif em_init:
        from py_idr.inference.mcmc.em_init import compute_em_init  # local import to avoid cycle

        em = compute_em_init(np.asarray(F0))
        # Clamp rho into (0, 1) since the simple sweep's atom NUTS uses
        # Beta(2, 2) on rho.
        rho_clamped = float(np.clip(em.rho, 0.05, 0.95))
        state = initial_simple_state(n=n, K=K, M=M, pi_init=em.pi, rho_init=rho_clamped)
        # Inject the EM Z to skip the first sweep's bootstrap mis-classification.
        state = SimpleSweepState(
            pi=state.pi,
            rho=state.rho,
            bernstein_weights=state.bernstein_weights,
            Z=jnp.asarray(em.Z, dtype=jnp.int32),
        )
    else:
        state = initial_simple_state(n=n, K=K, M=M)

    total_iters = n_warmup + n_samples
    iter_keys = jax.random.split(key, total_iters)

    # Scalar samples accumulator. Storing as Python lists is fine here — the
    # trace is small (a few floats per iteration) and arviz wants numpy arrays
    # in the end anyway.
    pi_trace: list[float] = []
    rho_trace: list[float] = []
    n_reproducible_trace: list[int] = []
    z_trace: list[np.ndarray] = []
    # Aggregated over post-warmup sweeps only; warmup divergences don't count
    # toward the §3.3.1 divergence-fraction threshold (per Stan / NumPyro
    # convention).
    num_divergent_transitions = 0

    for i, k in enumerate(iter_keys):
        state, sweep_diag = sweep_simple(state, inputs, k)
        if i >= n_warmup:
            pi_trace.append(float(state.pi))
            rho_trace.append(float(state.rho))
            n_reproducible_trace.append(int(jnp.sum(state.Z)))
            z_trace.append(np.asarray(state.Z, dtype=np.int8))
            num_divergent_transitions += int(sweep_diag.get("num_divergences", 0))

    # Build the InferenceData. arviz expects shape (chains, draws, ...);
    # the API takes a nested {group_name: {var_name: array}} mapping.
    posterior_group = {
        "pi": np.asarray(pi_trace, dtype=np.float64)[None, :],  # (1, draws)
        "rho": np.asarray(rho_trace, dtype=np.float64)[None, :],
        "n_reproducible": np.asarray(n_reproducible_trace, dtype=np.int32)[None, :],
    }
    idata = _idata_from_posterior(posterior_group)
    z_arr = np.stack(z_trace, axis=0)  # (n_samples, n)
    return ChainResult(
        idata=idata,
        final_state=state,
        z_samples=z_arr,
        num_divergent_transitions=num_divergent_transitions,
    )


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
    em_init: bool = False,
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
    # When em_init=True, the per-chain inits scatter around the
    # EM-recovered (pi, rho) so split-Rhat still has signal.
    if em_init:
        from py_idr.inference.mcmc.em_init import compute_em_init  # local import to avoid cycle

        em = compute_em_init(np.asarray(F0))
        pi_center = float(np.clip(em.pi, 0.10, 0.90))
        rho_center = float(np.clip(em.rho, 0.10, 0.90))
        spread = 0.10
        pi_inits = jnp.linspace(
            max(0.02, pi_center - spread), min(0.98, pi_center + spread), num_chains
        )
        rho_inits = jnp.linspace(
            max(0.02, rho_center - spread), min(0.98, rho_center + spread), num_chains
        )
        em_Z = jnp.asarray(em.Z, dtype=jnp.int32)
    elif overdispersed:
        pi_inits = jnp.linspace(0.2, 0.8, num_chains)
        rho_inits = jnp.linspace(0.15, 0.85, num_chains)
        em_Z = None
    else:
        pi_inits = jnp.full(num_chains, 0.5)
        rho_inits = jnp.full(num_chains, 0.3)
        em_Z = None
    del init_keys  # not used — but reserved for future stochastic perturbations

    pi_chains: list[np.ndarray] = []
    rho_chains: list[np.ndarray] = []
    n_rep_chains: list[np.ndarray] = []
    z_chains: list[np.ndarray] = []
    final_states: list[SimpleSweepState] = []
    num_divergent_transitions = 0

    for c in range(num_chains):
        init_state = initial_simple_state(
            n=n,
            K=K,
            M=M,
            pi_init=float(pi_inits[c]),
            rho_init=float(rho_inits[c]),
        )
        if em_Z is not None:
            init_state = SimpleSweepState(
                pi=init_state.pi,
                rho=init_state.rho,
                bernstein_weights=init_state.bernstein_weights,
                Z=em_Z,
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
        z_chains.append(chain_res.z_samples)
        final_states.append(chain_res.final_state)
        num_divergent_transitions += chain_res.num_divergent_transitions

    posterior_group = {
        "pi": np.stack(pi_chains, axis=0),  # (num_chains, draws)
        "rho": np.stack(rho_chains, axis=0),
        "n_reproducible": np.stack(n_rep_chains, axis=0),
    }
    idata = _idata_from_posterior(posterior_group)
    z_arr = np.stack(z_chains, axis=0)  # (num_chains, n_samples, n)
    return MultiChainResult(
        idata=idata,
        final_states=tuple(final_states),
        z_samples=z_arr,
        num_divergent_transitions=num_divergent_transitions,
    )


def summarise(result: ChainResult, var_names: list[str] | None = None) -> "az.InferenceData":
    """Return :func:`arviz.summary` for the chain's scalar parameters.

    Returns a DataFrame containing ``mean``, ``sd``, ``hdi_3%``, ``hdi_97%``,
    ``ess_bulk``, ``ess_tail``, ``r_hat`` per variable. The values are the
    standard arviz outputs — we don't reimplement them.
    """
    return az.summary(result.idata, var_names=var_names)


# =========================================================================
# Multi-cluster drivers
# =========================================================================
#
# Same shape as the simple drivers above but routing each sweep through
# multi_cluster_sweep (Algorithm 1). The posterior trace adds the PY
# hyperparameters (alpha, sigma) and the cluster count T. The Z-trace
# stays for the proper-Monte-Carlo local idr estimator. A second trace
# array — cluster_assignments — records state.z so cluster-level
# diagnostics (per-cluster size, atom evolution) can be reconstructed
# post-hoc.


@dataclass(frozen=True)
class ChainResultMulti:
    """Output of :func:`run_chain_multi` (single-chain, multi-cluster path).

    Attributes
    ----------
    idata
        :class:`arviz.InferenceData` with the scalar posterior trace.
        Variables: ``pi``, ``alpha``, ``sigma``, ``T``, ``n_reproducible``.
    final_state
        Final :class:`MultiClusterState`. Carries the atom tuple at the
        last draw — useful for chain restarts and for inspecting which
        copula families ended up populated.
    z_samples
        Class-indicator trace ($Z$ in the state, **not** cluster label);
        shape ``(n_samples, n)`` int8 with values in $\\{0, 1\\}$.
        Consumed by :func:`py_idr.decision.local_idr.from_mcmc`.
    cluster_assignments
        Cluster-label trace ($z$ in the state); shape
        ``(n_samples, n)`` int32. Entries are ``-1`` for irreproducible
        features, ``0..T-1`` for reproducible. Useful for per-cluster
        size summaries and label-invariant diagnostics.
    T_trace
        Per-draw cluster count, shape ``(n_samples,)`` int32. Same data
        as ``idata.posterior["T"]`` but exposed as a plain array for
        ergonomic plotting.
    """

    idata: az.InferenceData
    final_state: MultiClusterState
    z_samples: np.ndarray | None = None
    cluster_assignments: np.ndarray | None = None
    T_trace: np.ndarray | None = None
    num_divergent_transitions: int = 0


@dataclass(frozen=True)
class MultiChainResultMulti:
    """Output of :func:`run_multi_chain_multi` (multi-chain, multi-cluster path).

    Attributes mirror :class:`ChainResultMulti` but with a leading
    ``num_chains`` axis on every trace array. ``idata`` has the standard
    ``(chain, draw)`` posterior layout that ``arviz.summary`` /
    ``arviz.rhat`` expect.
    """

    idata: az.InferenceData
    final_states: tuple[MultiClusterState, ...]
    z_samples: np.ndarray | None = None
    cluster_assignments: np.ndarray | None = None
    T_trace: np.ndarray | None = None
    num_divergent_transitions: int = 0


def _initial_multi_state(
    *,
    n: int,
    K: int,
    M: int,
    T_max: int,
    key: jax.Array,
    pi_init: float = 0.5,
    alpha_init: float = 1.0,
    sigma_init: float = 0.1,
    alpha_F: float = 1.0,
) -> MultiClusterState:
    """Build a default :class:`MultiClusterState` with one Gaussian-copula atom.

    The initial atom is a Gaussian copula with $\\rho$ drawn from the
    Beta(2, 2) prior — wide enough to be a sensible warm start for any
    regime; the chain's atom-NUTS step refines it.
    """
    init_atom = prior_propose_atom(key, "gauss", K)
    return initial_multi_cluster_state(
        n=n,
        K=K,
        M=M,
        T_max=T_max,
        initial_atom=init_atom,
        pi_init=pi_init,
        alpha_init=alpha_init,
        sigma_init=sigma_init,
        alpha_F=alpha_F,
    )


def run_chain_multi(
    *,
    F0: Float[Array, "n K"],
    M: int,
    key: jax.Array,
    T_max: int = 10,
    H: int = 5,
    n_warmup: int = 200,
    n_samples: int = 200,
    initial_state: MultiClusterState | None = None,
    pi_init: float = 0.5,
    alpha_init: float = 1.0,
    sigma_init: float = 0.1,
    alpha_F: float = 1.0,
    nuts_warmup: int = 30,
    py_hyperparam_warmup: int = 50,
    em_init: bool = False,
) -> ChainResultMulti:
    """Run a single chain of the multi-cluster PY-IDR sweep (Algorithm 1).

    Parameters
    ----------
    F0
        Pilot CDF matrix; shape $(n, K)$.
    M
        Bernstein degree (held fixed for this commit).
    key
        JAX PRNG key.
    T_max
        Hard cap on the number of active clusters. The chain raises if a
        Pólya-urn proposal would exceed it. Default 10 is generous for
        every regime in the report.
    H
        Auxiliary-atom count for the Pólya-urn step. Default 5 matches
        §3.3.1.
    n_warmup, n_samples
        Sample counts. Warmup draws are discarded from the
        ``InferenceData`` but their compute time is fully spent.
    initial_state
        Optional override of the chain start; defaults to a single
        Gaussian atom at $\\rho \\sim \\mathrm{Beta}(2, 2)$.
    pi_init, alpha_init, sigma_init, alpha_F
        Scalar hyperparameters used only when ``initial_state`` is None.
    nuts_warmup
        Per-cluster atom-NUTS warmup. Default 30.
    py_hyperparam_warmup
        NUTS warmup for the joint $(\\alpha, \\sigma)$ update. Default 50.

    Returns
    -------
    A :class:`ChainResultMulti` carrying the arviz InferenceData plus
    the per-draw cluster-label and class-indicator traces.
    """
    if F0.ndim != 2:
        raise ValueError(f"F0 must be 2-D (n, K); got shape {F0.shape}")
    if n_warmup < 0 or n_samples <= 0:
        raise ValueError(f"need n_warmup >= 0 and n_samples > 0; got ({n_warmup}, {n_samples})")
    if T_max < 1:
        raise ValueError(f"T_max must be >= 1; got {T_max}")
    if H < 1:
        raise ValueError(f"H (aux-atom count) must be >= 1; got {H}")

    n, K = F0.shape
    init_key, run_key = jax.random.split(key, 2)
    if initial_state is None:
        # Optional EM warm start: overrides pi_init and seeds the initial
        # atom's rho with the pairwise EM estimate.
        if em_init:
            from py_idr.inference.mcmc.em_init import compute_em_init

            em = compute_em_init(np.asarray(F0))
            pi_init = float(em.pi)
            # Build the initial state, then override its Gaussian atom
            # with one whose rho matches the EM estimate.
            rho_seed = float(np.clip(em.rho, 0.05, 0.95))
            seed_atom = prior_propose_atom(init_key, "gauss", K)
            # Rebuild the atom with the EM rho (the prior_propose_atom call
            # gave us the right structure; we just swap its parameter).
            from py_idr.copulas.gaussian import GaussianCopula
            from py_idr.inference.mcmc.type_update import TYPE_TO_INDEX, AtomDraw

            R = (1.0 - rho_seed) * jnp.eye(K) + rho_seed * jnp.ones((K, K))
            L = jnp.linalg.cholesky(R)
            seed_atom = AtomDraw(
                "gauss", TYPE_TO_INDEX["gauss"], GaussianCopula(L), {"rho": rho_seed}
            )
            state = initial_multi_cluster_state(
                n=n,
                K=K,
                M=M,
                T_max=T_max,
                initial_atom=seed_atom,
                pi_init=pi_init,
                alpha_init=alpha_init,
                sigma_init=sigma_init,
                alpha_F=alpha_F,
            )
            # Inject the EM Z so the first sweep doesn't start with
            # all-features-reproducible.
            from dataclasses import replace as _dc_replace

            state = _dc_replace(state, Z=jnp.asarray(em.Z, dtype=jnp.int32))
        else:
            state = _initial_multi_state(
                n=n,
                K=K,
                M=M,
                T_max=T_max,
                key=init_key,
                pi_init=pi_init,
                alpha_init=alpha_init,
                sigma_init=sigma_init,
                alpha_F=alpha_F,
            )
    else:
        state = initial_state

    total_iters = n_warmup + n_samples
    iter_keys = jax.random.split(run_key, total_iters)

    # Scalar trace accumulators.
    pi_trace: list[float] = []
    alpha_trace: list[float] = []
    sigma_trace: list[float] = []
    T_trace: list[int] = []
    n_reproducible_trace: list[int] = []
    # Z trace (class indicator) and z trace (cluster label).
    Z_trace: list[np.ndarray] = []
    z_trace: list[np.ndarray] = []
    num_divergent_transitions = 0

    for i, k in enumerate(iter_keys):
        state, sweep_diag = multi_cluster_sweep(
            state,
            jnp.asarray(F0),
            k,
            H=H,
            nuts_warmup=nuts_warmup,
            py_hyperparam_warmup=py_hyperparam_warmup,
        )
        if i >= n_warmup:
            pi_trace.append(float(state.pi))
            alpha_trace.append(float(state.alpha))
            sigma_trace.append(float(state.sigma))
            T_trace.append(int(state.T))
            n_reproducible_trace.append(int(jnp.sum(state.Z)))
            Z_trace.append(np.asarray(state.Z, dtype=np.int8))
            z_trace.append(np.asarray(state.z, dtype=np.int32))
            num_divergent_transitions += int(sweep_diag.get("num_divergences", 0))

    posterior_group = {
        "pi": np.asarray(pi_trace, dtype=np.float64)[None, :],
        "alpha": np.asarray(alpha_trace, dtype=np.float64)[None, :],
        "sigma": np.asarray(sigma_trace, dtype=np.float64)[None, :],
        "T": np.asarray(T_trace, dtype=np.int32)[None, :],
        "n_reproducible": np.asarray(n_reproducible_trace, dtype=np.int32)[None, :],
    }
    idata = _idata_from_posterior(posterior_group)
    Z_arr = np.stack(Z_trace, axis=0)  # (n_samples, n)
    z_arr = np.stack(z_trace, axis=0)  # (n_samples, n)
    T_arr = np.asarray(T_trace, dtype=np.int32)

    return ChainResultMulti(
        idata=idata,
        final_state=state,
        z_samples=Z_arr,
        cluster_assignments=z_arr,
        T_trace=T_arr,
        num_divergent_transitions=num_divergent_transitions,
    )


def run_multi_chain_multi(
    *,
    F0: Float[Array, "n K"],
    M: int,
    key: jax.Array,
    T_max: int = 10,
    H: int = 5,
    num_chains: int = 4,
    n_warmup: int = 200,
    n_samples: int = 200,
    pi_init: float = 0.5,
    alpha_init: float = 1.0,
    sigma_init: float = 0.1,
    alpha_F: float = 1.0,
    nuts_warmup: int = 30,
    py_hyperparam_warmup: int = 50,
    overdispersed: bool = True,
    em_init: bool = False,
) -> MultiChainResultMulti:
    """Run ``num_chains`` multi-cluster chains, stacked into a multi-chain idata.

    Parameters
    ----------
    F0, M, T_max, H, n_warmup, n_samples, alpha_F, nuts_warmup, py_hyperparam_warmup
        Same as :func:`run_chain_multi`.
    num_chains
        Number of chains. Default 4 — the minimum the diagnostics report
        accepts as a `pass` verdict for split-$\\widehat R$.
    key
        Root PRNG key; split into per-chain subkeys.
    pi_init, alpha_init, sigma_init
        Scalar hyperparameter starts. Per-chain inits are dispersed
        around these defaults when ``overdispersed=True``.
    overdispersed
        If True, per-chain initial states scatter $\\pi$, $\\alpha$,
        $\\sigma$ across a wide range so split-$\\widehat R$ has signal.
    """
    if num_chains < 1:
        raise ValueError(f"num_chains must be >= 1; got {num_chains}")
    if F0.ndim != 2:
        raise ValueError(f"F0 must be 2-D (n, K); got shape {F0.shape}")

    n, K = F0.shape
    chain_keys = jax.random.split(key, num_chains + 1)
    per_chain_keys = chain_keys[1:]

    if overdispersed:
        pi_inits = jnp.linspace(0.20, 0.80, num_chains)
        alpha_inits = jnp.linspace(0.5, 2.0, num_chains)
        sigma_inits = jnp.linspace(0.05, 0.40, num_chains)
    else:
        pi_inits = jnp.full(num_chains, pi_init)
        alpha_inits = jnp.full(num_chains, alpha_init)
        sigma_inits = jnp.full(num_chains, sigma_init)

    pi_chains: list[np.ndarray] = []
    alpha_chains: list[np.ndarray] = []
    sigma_chains: list[np.ndarray] = []
    T_chains: list[np.ndarray] = []
    n_rep_chains: list[np.ndarray] = []
    Z_chains: list[np.ndarray] = []
    z_chains: list[np.ndarray] = []
    final_states: list[MultiClusterState] = []
    num_divergent_transitions = 0

    for c in range(num_chains):
        chain_res = run_chain_multi(
            F0=F0,
            M=M,
            key=per_chain_keys[c],
            T_max=T_max,
            H=H,
            n_warmup=n_warmup,
            n_samples=n_samples,
            pi_init=float(pi_inits[c]),
            alpha_init=float(alpha_inits[c]),
            sigma_init=float(sigma_inits[c]),
            alpha_F=alpha_F,
            nuts_warmup=nuts_warmup,
            py_hyperparam_warmup=py_hyperparam_warmup,
            em_init=em_init,
        )
        posterior = chain_res.idata.posterior
        pi_chains.append(np.asarray(posterior["pi"]).reshape(-1))
        alpha_chains.append(np.asarray(posterior["alpha"]).reshape(-1))
        sigma_chains.append(np.asarray(posterior["sigma"]).reshape(-1))
        T_chains.append(np.asarray(posterior["T"]).reshape(-1))
        n_rep_chains.append(np.asarray(posterior["n_reproducible"]).reshape(-1))
        Z_chains.append(chain_res.z_samples)
        z_chains.append(chain_res.cluster_assignments)
        final_states.append(chain_res.final_state)
        num_divergent_transitions += chain_res.num_divergent_transitions

    posterior_group = {
        "pi": np.stack(pi_chains, axis=0),
        "alpha": np.stack(alpha_chains, axis=0),
        "sigma": np.stack(sigma_chains, axis=0),
        "T": np.stack(T_chains, axis=0),
        "n_reproducible": np.stack(n_rep_chains, axis=0),
    }
    idata = _idata_from_posterior(posterior_group)

    return MultiChainResultMulti(
        idata=idata,
        final_states=tuple(final_states),
        z_samples=np.stack(Z_chains, axis=0),
        cluster_assignments=np.stack(z_chains, axis=0),
        T_trace=np.stack(T_chains, axis=0),
        num_divergent_transitions=num_divergent_transitions,
    )


# Re-export for downstream consumers; cluster_sizes is convenient when
# the caller is summarising a single MultiClusterState.
__all__ = [
    "ChainResult",
    "ChainResultMulti",
    "MultiChainResult",
    "MultiChainResultMulti",
    "cluster_sizes",
    "run_chain_multi",
    "run_chain_simple",
    "run_multi_chain_multi",
    "run_multi_chain_simple",
    "summarise",
]
