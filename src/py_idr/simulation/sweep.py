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
from pathlib import Path
from typing import Any

import pandas as pd

from py_idr.simulation.replicates import (
    _SIMULATORS,
    FitResult,
    ReplicateRow,
    _fit_to_idr,
    _make_row,
)


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
