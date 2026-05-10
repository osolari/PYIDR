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


def summarise(result: ChainResult, var_names: list[str] | None = None) -> "az.InferenceData":
    """Return :func:`arviz.summary` for the chain's scalar parameters.

    Returns a DataFrame containing ``mean``, ``sd``, ``hdi_3%``, ``hdi_97%``,
    ``ess_bulk``, ``ess_tail``, ``r_hat`` per variable. The values are the
    standard arviz outputs — we don't reimplement them.
    """
    return az.summary(result.idata, var_names=var_names)
