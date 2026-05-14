"""Sweep driver + long-format CSV writer.

A *sweep* is "run regime $R$ for $N$ independent replicates and evaluate
at $\\alpha \\in A$." Each (replicate, $\\alpha$) pair produces one
:class:`py_idr.simulation.replicates.ReplicateRow`; the rows concatenate
into a long-format DataFrame.

Two entry points:

- :func:`run_sweep` — fit the chain once per replicate, evaluate at every
  $\\alpha$ in ``alphas``, return the list of rows. The chain cost is
  paid once per (regime, seed) tuple; the alpha sweep is essentially
  free.
- :func:`write_sweep_csv` — pandas-based long-format CSV writer.

The output CSV columns match :class:`ReplicateRow.to_dict()` exactly, so
the F1 / Table 4 reproducers can read it directly.

Reproducibility
---------------

Seeds are derived from a single ``base_seed`` as
``seed = base_seed + replicate_index`` for ``replicate_index`` in
``range(num_replicates)``. The base seed appears in the optional
:class:`py_idr.data.schema.SeedLedger`; callers passing one through
:func:`run_sweep` get the ledger filled in automatically.

See plans/07_simulation_studies.md.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from py_idr.simulation.replicates import (
    _SIMULATORS,
    FitResult,
    ReplicateRow,
    _fit_to_idr,
    _make_row,
)


@dataclass(frozen=True)
class IdrTrace:
    """Per-replicate ``(idr, true_Z)`` arrays paired with regime + seed metadata.

    Stored in the NPZ sidecar produced by :func:`write_idr_sidecar`;
    consumed by F4 (ROC/PR figure) which needs the full per-feature
    posterior rather than the scalar summaries that fit the sweep CSV.

    Attributes
    ----------
    regime
        Regime label.
    replicate_seed
        Replicate seed for traceability.
    idr
        Posterior local idr, shape $(n,)$, values in $[0, 1]$.
    true_Z
        Ground-truth class indicators, shape $(n,)$, values in $\\{0, 1\\}$.
    """

    regime: str
    replicate_seed: int
    idr: np.ndarray
    true_Z: np.ndarray


@dataclass(frozen=True)
class SweepResult:
    """Bundled output of :func:`run_sweep_with_traces`.

    Attributes
    ----------
    rows
        Long-format :class:`ReplicateRow` list — same shape as
        :func:`run_sweep`'s return.
    idr_traces
        Per-replicate :class:`IdrTrace` (one per (regime, seed) tuple,
        **not** per alpha — the idr is the same across alphas).
    """

    rows: list[ReplicateRow]
    idr_traces: list[IdrTrace]


def run_sweep(
    regime: str,
    *,
    K: int,
    n: int,
    num_replicates: int,
    base_seed: int = 0,
    alphas: Sequence[float] = (0.01, 0.05, 0.10, 0.20),
    num_chains: int = 2,
    n_warmup: int = 50,
    n_samples: int = 100,
    nuts_warmup: int = 20,
    M: int = 5,
    chain_type: str = "auto",
    T_max: int = 10,
    H: int = 5,
    py_hyperparam_warmup: int = 50,
    em_init: bool = False,
    scenario_kwargs: dict[str, Any] | None = None,
) -> list[ReplicateRow]:
    """Run ``num_replicates`` independent replicates of one regime.

    Each replicate uses ``seed = base_seed + replicate_index``. For every
    replicate, the chain is fit once and the row is evaluated at every
    $\\alpha$ in ``alphas`` — so the output has
    ``num_replicates * len(alphas)`` rows total.

    Parameters
    ----------
    regime
        One of the keys in :data:`_SIMULATORS` (``"S1"``..``"S5-sparse"``).
    K, n
        Replicate count and feature count, forwarded to the simulator.
    num_replicates
        Number of independent replicates. The seed schedule is
        ``base_seed + i`` for ``i in range(num_replicates)``.
    base_seed
        Anchor seed. The same ``base_seed`` reproduces the entire sweep
        exactly.
    alphas
        Sun-Cai operating levels to evaluate at. Each emits one row per
        replicate.
    num_chains, n_warmup, n_samples, nuts_warmup, M
        Forwarded to the chain driver
        (:func:`py_idr.inference.mcmc.run_chain.run_multi_chain_simple` or
        :func:`py_idr.inference.mcmc.run_chain.run_multi_chain_multi`).
    chain_type
        ``"auto"`` (default; T>1 for S5/S5-sparse, T=1 otherwise),
        ``"T=1"``, or ``"T>1"``.
    T_max, H, py_hyperparam_warmup
        Multi-cluster driver knobs (ignored on the T=1 path).
    scenario_kwargs
        Regime-specific overrides forwarded to the simulator
        (e.g. ``{"pi_star": 0.30}`` for S1).

    Returns
    -------
    A flat list of :class:`ReplicateRow`, length
    ``num_replicates * len(alphas)``, ordered by replicate then alpha.

    Raises
    ------
    ValueError
        If ``regime`` is not in :data:`_SIMULATORS`, ``num_replicates``
        is non-positive, or ``alphas`` is empty.
    """
    if regime not in _SIMULATORS:
        raise ValueError(f"unknown regime {regime!r}; expected one of {sorted(_SIMULATORS)}")
    if num_replicates <= 0:
        raise ValueError(f"num_replicates must be > 0; got {num_replicates}")
    if not alphas:
        raise ValueError("alphas must be a non-empty sequence")

    simulator = _SIMULATORS[regime]
    extra = dict(scenario_kwargs or {})
    rows: list[ReplicateRow] = []

    for i in range(num_replicates):
        seed = base_seed + i
        sim = simulator(K=K, n=n, seed=seed, **extra)
        fit: FitResult = _fit_to_idr(
            sim,
            num_chains=num_chains,
            n_warmup=n_warmup,
            n_samples=n_samples,
            nuts_warmup=nuts_warmup,
            M=M,
            chain_type=chain_type,
            T_max=T_max,
            H=H,
            py_hyperparam_warmup=py_hyperparam_warmup,
            em_init=em_init,
        )
        for alpha in alphas:
            rows.append(_make_row(sim, fit, float(alpha)))

    return rows


def to_dataframe(rows: Sequence[ReplicateRow]) -> pd.DataFrame:
    """Convert a sweep's rows to a long-format pandas ``DataFrame``.

    Column layout matches :meth:`ReplicateRow.to_dict`. The dataframe is
    suitable for the figure / table reproducers and for filtering /
    aggregation in downstream notebooks.
    """
    if not rows:
        return pd.DataFrame(columns=list(ReplicateRow.__dataclass_fields__.keys()))
    return pd.DataFrame([r.to_dict() for r in rows])


def write_sweep_csv(rows: Sequence[ReplicateRow], path: str | Path) -> Path:
    """Write a sweep's rows to a CSV at ``path``.

    Creates the parent directory if needed. Returns the resolved path
    for downstream pipelines that need to record it in the
    :class:`py_idr.data.schema.RunManifest`.
    """
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    df = to_dataframe(rows)
    df.to_csv(out, index=False)
    return out


# ---- run_sweep_with_traces + NPZ sidecar (for F4 ROC/PR) -----------------------------


def run_sweep_with_traces(
    regime: str,
    *,
    K: int,
    n: int,
    num_replicates: int,
    base_seed: int = 0,
    alphas: Sequence[float] = (0.01, 0.05, 0.10, 0.20),
    num_chains: int = 2,
    n_warmup: int = 50,
    n_samples: int = 100,
    nuts_warmup: int = 20,
    M: int = 5,
    chain_type: str = "auto",
    T_max: int = 10,
    H: int = 5,
    py_hyperparam_warmup: int = 50,
    em_init: bool = False,
    scenario_kwargs: dict[str, Any] | None = None,
) -> SweepResult:
    """Like :func:`run_sweep`, but also returns per-replicate ``(idr, Z)`` arrays.

    Parameters are identical to :func:`run_sweep`. The extra
    :class:`IdrTrace` per replicate is what F4 (ROC/PR figure)
    consumes. Memory cost: $O(\\text{num\\_replicates} \\cdot n)$ on
    top of the rows.

    Returns
    -------
    A :class:`SweepResult` with ``rows`` matching :func:`run_sweep` and
    ``idr_traces`` carrying the per-replicate posterior + ground truth.

    Raises
    ------
    ValueError
        Same validation as :func:`run_sweep`.
    """
    if regime not in _SIMULATORS:
        raise ValueError(f"unknown regime {regime!r}; expected one of {sorted(_SIMULATORS)}")
    if num_replicates <= 0:
        raise ValueError(f"num_replicates must be > 0; got {num_replicates}")
    if not alphas:
        raise ValueError("alphas must be a non-empty sequence")

    simulator = _SIMULATORS[regime]
    extra = dict(scenario_kwargs or {})
    rows: list[ReplicateRow] = []
    traces: list[IdrTrace] = []

    for i in range(num_replicates):
        seed = base_seed + i
        sim = simulator(K=K, n=n, seed=seed, **extra)
        fit: FitResult = _fit_to_idr(
            sim,
            num_chains=num_chains,
            n_warmup=n_warmup,
            n_samples=n_samples,
            nuts_warmup=nuts_warmup,
            M=M,
            chain_type=chain_type,
            T_max=T_max,
            H=H,
            py_hyperparam_warmup=py_hyperparam_warmup,
            em_init=em_init,
        )
        for alpha in alphas:
            rows.append(_make_row(sim, fit, float(alpha)))
        traces.append(
            IdrTrace(
                regime=sim.regime,
                replicate_seed=seed,
                idr=np.asarray(fit.idr, dtype=np.float64),
                true_Z=np.asarray(sim.true_Z, dtype=np.int8),
            )
        )

    return SweepResult(rows=rows, idr_traces=traces)


def _trace_key(prefix: str, regime: str, seed: int) -> str:
    """Stable NPZ key combining prefix ('idr' or 'Z'), regime, and seed.

    NPZ keys may not contain certain characters, but ``__`` as a
    delimiter is fine. Regime labels in PY-IDR are alphanumeric + '-'
    (e.g. ``S5-sparse``), all NPZ-safe.
    """
    return f"{prefix}__{regime}__{seed}"


def write_idr_sidecar(
    traces: Sequence[IdrTrace],
    path: str | Path,
) -> Path:
    """Write per-replicate ``(idr, true_Z)`` arrays to a compressed NPZ sidecar.

    The file ends up next to the sweep ``results.csv`` and contains
    one ``idr__<regime>__<seed>`` array and one ``Z__<regime>__<seed>``
    array per replicate. Read it back with :func:`load_idr_sidecar`.

    Parameters
    ----------
    traces
        Output of :attr:`SweepResult.idr_traces`.
    path
        Output NPZ path. Parent directory is created if missing.

    Returns
    -------
    The resolved output path.

    Raises
    ------
    ValueError
        If ``traces`` is empty.
    """
    if not traces:
        raise ValueError("traces is empty; nothing to write")

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)

    arrays: dict[str, np.ndarray] = {}
    for tr in traces:
        arrays[_trace_key("idr", tr.regime, tr.replicate_seed)] = tr.idr
        arrays[_trace_key("Z", tr.regime, tr.replicate_seed)] = tr.true_Z
    np.savez_compressed(out, **arrays)
    return out


def load_idr_sidecar(
    path: str | Path,
) -> dict[str, list[tuple[np.ndarray, np.ndarray]]]:
    """Read an NPZ sidecar produced by :func:`write_idr_sidecar`.

    Returns the dict shape F4 expects: ``{regime: [(idr, Z), ...]}``.
    Replicates within each regime are returned in ascending seed order.

    Raises
    ------
    FileNotFoundError
        If ``path`` does not exist.
    ValueError
        If the NPZ contains an ``idr__*`` key without a matching
        ``Z__*`` key (corrupted sidecar).
    """
    npz_path = Path(path)
    if not npz_path.exists():
        raise FileNotFoundError(f"idr sidecar NPZ not found: {npz_path}")
    data = np.load(npz_path, allow_pickle=False)

    # Group keys by (regime, seed) and pair idr / Z.
    by_regime: dict[str, list[tuple[int, np.ndarray, np.ndarray]]] = {}
    for key in data.files:
        if not key.startswith("idr__"):
            continue
        # idr__<regime>__<seed>
        _, regime, seed_str = key.split("__", 2)
        seed = int(seed_str)
        z_key = _trace_key("Z", regime, seed)
        if z_key not in data.files:
            raise ValueError(f"sidecar is corrupted: missing {z_key!r} for {key!r}")
        idr = np.asarray(data[key])
        Z = np.asarray(data[z_key])
        by_regime.setdefault(regime, []).append((seed, idr, Z))

    out: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {}
    for regime, items in by_regime.items():
        items.sort(key=lambda t: t[0])  # ascending seed
        out[regime] = [(idr, Z) for _, idr, Z in items]
    return out


# ---- comparator sweep: Vanilla IDR (pairwise) ----------------------------------------


def run_sweep_vanilla_idr(
    regime: str,
    *,
    K: int,
    n: int,
    num_replicates: int,
    base_seed: int = 0,
    alphas: Sequence[float] = (0.01, 0.05, 0.10, 0.20),
    aggregation: str = "mean",
    pi_init: float = 0.5,
    rho_init: float = 0.5,
    max_iter: int = 200,
    tol: float = 1e-5,
    scenario_kwargs: dict[str, Any] | None = None,
) -> list[ReplicateRow]:
    """Vanilla IDR (Li et al. 2011) sweep — comparator companion to :func:`run_sweep`.

    Same `(regime, K, n, num_replicates, base_seed)` shape as
    :func:`run_sweep` so a sweep CSV can hold both PY-IDR rows and
    Vanilla IDR rows side by side. The EM is deterministic given the
    data, so each replicate produces a single fit; we still evaluate at
    every $\\alpha$ in ``alphas`` for table compatibility.

    Returns a flat list of :class:`ReplicateRow` with
    ``method = "Vanilla IDR (pairwise)"`` and
    ``len(rows) == num_replicates * len(alphas)``.
    """
    import time as _time

    from py_idr.comparators.vanilla_idr import fit_vanilla_idr_pairwise

    if regime not in _SIMULATORS:
        raise ValueError(f"unknown regime {regime!r}; expected one of {sorted(_SIMULATORS)}")
    if num_replicates <= 0:
        raise ValueError(f"num_replicates must be > 0; got {num_replicates}")
    if not alphas:
        raise ValueError("alphas must be a non-empty sequence")

    simulator = _SIMULATORS[regime]
    extra = dict(scenario_kwargs or {})
    rows: list[ReplicateRow] = []
    import jax.numpy as _jnp

    from py_idr.simulation.evaluation import evaluate_recovery as _evaluate_recovery

    for i in range(num_replicates):
        seed = base_seed + i
        sim = simulator(K=K, n=n, seed=seed, **extra)
        t0 = _time.perf_counter()
        idr = fit_vanilla_idr_pairwise(
            np.asarray(sim.X),
            aggregation=aggregation,  # type: ignore[arg-type]
            pi_init=pi_init,
            rho_init=rho_init,
            max_iter=max_iter,
            tol=tol,
        )
        walltime = float(_time.perf_counter() - t0)
        idr_jax = _jnp.asarray(idr)
        for alpha in alphas:
            recovery = _evaluate_recovery(idr_jax, sim.true_Z, alpha=float(alpha))
            rows.append(
                ReplicateRow(
                    regime=sim.regime,
                    K=K,
                    n=n,
                    replicate_seed=sim.seed,
                    method="Vanilla IDR (pairwise)",
                    alpha=float(alpha),
                    pi_true=sim.true_pi,
                    pi_posterior_mean=float("nan"),
                    rho_true=sim.true_rho,
                    rho_posterior_mean=float("nan"),
                    T_posterior_mean=float("nan"),
                    k_alpha=recovery.k_alpha,
                    realized_fdr=recovery.realized_fdr,
                    power=recovery.power,
                    num_chains=1,
                    num_samples_per_chain=max_iter,
                    walltime_s=walltime,
                )
            )

    return rows


def run_sweep_maxrank(
    regime: str,
    *,
    K: int,
    n: int,
    num_replicates: int,
    base_seed: int = 0,
    alphas: Sequence[float] = (0.01, 0.05, 0.10, 0.20),
    tie_method: str = "average",
    scenario_kwargs: dict[str, Any] | None = None,
) -> list[ReplicateRow]:
    """MaxRank sweep — comparator companion to :func:`run_sweep`.

    Same `(regime, K, n, num_replicates, base_seed)` shape as
    :func:`run_sweep`; rows carry ``method = "MaxRank"`` so they drop
    into the same CSV alongside PY-IDR and Vanilla IDR rows.

    The MaxRank statistic has no random component, so all replicates
    sharing a seed produce identical idr arrays — the per-alpha rows
    differ only in the Sun-Cai cutoff and the resulting decision
    metadata.
    """
    import time as _time

    from py_idr.comparators.maxrank import fit_maxrank

    if regime not in _SIMULATORS:
        raise ValueError(f"unknown regime {regime!r}; expected one of {sorted(_SIMULATORS)}")
    if num_replicates <= 0:
        raise ValueError(f"num_replicates must be > 0; got {num_replicates}")
    if not alphas:
        raise ValueError("alphas must be a non-empty sequence")

    simulator = _SIMULATORS[regime]
    extra = dict(scenario_kwargs or {})
    rows: list[ReplicateRow] = []
    import jax.numpy as _jnp

    from py_idr.simulation.evaluation import evaluate_recovery as _evaluate_recovery

    for i in range(num_replicates):
        seed = base_seed + i
        sim = simulator(K=K, n=n, seed=seed, **extra)
        t0 = _time.perf_counter()
        idr = fit_maxrank(np.asarray(sim.X), tie_method=tie_method)
        walltime = float(_time.perf_counter() - t0)
        idr_jax = _jnp.asarray(idr)
        for alpha in alphas:
            recovery = _evaluate_recovery(idr_jax, sim.true_Z, alpha=float(alpha))
            rows.append(
                ReplicateRow(
                    regime=sim.regime,
                    K=K,
                    n=n,
                    replicate_seed=sim.seed,
                    method="MaxRank",
                    alpha=float(alpha),
                    pi_true=sim.true_pi,
                    pi_posterior_mean=float("nan"),
                    rho_true=sim.true_rho,
                    rho_posterior_mean=float("nan"),
                    T_posterior_mean=float("nan"),
                    k_alpha=recovery.k_alpha,
                    realized_fdr=recovery.realized_fdr,
                    power=recovery.power,
                    num_chains=1,
                    num_samples_per_chain=1,
                    walltime_s=walltime,
                )
            )

    return rows
