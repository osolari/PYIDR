"""One replicate of an S1–S5 simulation cell, end-to-end.

The driver ties together:

1. ``simulate_S1`` (or its sibling regime generators) — synthetic data + truth.
2. ``empirical_rank_transform`` — pilot CDF input to the chain.
3. ``run_multi_chain_simple`` — multi-chain PY-IDR fit (T = 1 path).
4. ``from_mcmc`` — posterior local idr from the Z trace.
5. ``evaluate_recovery`` — realised FDR + power at the Sun--Cai threshold.

For S1–S4 the T = 1 sampler is the right tool: the reproducible component is
a single exchangeable Gaussian copula and the multi-cluster machinery is
overkill. S5 will route to the multi-cluster sweep once added.

The output is a flat :class:`ReplicateRow` consumable as a single CSV row by
the figure F1 / table T4 reproducers.

The public surface:

- :func:`run_replicate` — generic dispatcher that takes the regime label as a
  string (``"S1"``…``"S4"``) and a regime-specific kwargs dict. This is the
  entrypoint for sweep scripts.
- :func:`run_replicate_S1`–:func:`run_replicate_S4` — thin wrappers for
  ergonomic single-regime use (defaults baked in).

See plans/07_simulation_studies.md.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from py_idr.decision.local_idr import from_mcmc
from py_idr.inference.mcmc.run_chain import (
    run_multi_chain_multi,
    run_multi_chain_simple,
)
from py_idr.marginals.transform import empirical_rank_transform
from py_idr.simulation.evaluation import evaluate_recovery
from py_idr.simulation.scenarios import (
    SimulationResult,
    simulate_S1,
    simulate_S2,
    simulate_S3,
    simulate_S4,
    simulate_S5,
    simulate_S5_sparse,
)


@dataclass(frozen=True)
class ReplicateRow:
    """Flat per-replicate record consumed by figure / table reproducers.

    The ``T_posterior_mean`` column is NaN under the T=1 fast path
    (where there is exactly one cluster by construction) and a finite
    real number for the multi-cluster path. F2 (cluster-count figure)
    consumes it directly.

    The ``walltime_s`` column is the wall-clock time spent in the
    chain (and only the chain — Sun-Cai is microseconds, so excluding
    it is a rounding error). F3 (runtime figure) consumes it directly.
    Every (replicate, alpha) row from the same fit carries the same
    ``walltime_s`` value.
    """

    regime: str
    K: int
    n: int
    replicate_seed: int
    method: str
    alpha: float
    pi_true: float
    pi_posterior_mean: float
    rho_true: float
    rho_posterior_mean: float
    T_posterior_mean: float
    k_alpha: int
    realized_fdr: float
    power: float
    num_chains: int
    num_samples_per_chain: int
    walltime_s: float

    def to_dict(self) -> dict[str, object]:
        """Return a plain dict (the long-format CSV layout)."""
        return asdict(self)


# Regime → simulator function. New scenarios plug in here.
_SIMULATORS: dict[str, Callable[..., SimulationResult]] = {
    "S1": simulate_S1,
    "S2": simulate_S2,
    "S3": simulate_S3,
    "S4": simulate_S4,
    "S5": simulate_S5,
    "S5-sparse": simulate_S5_sparse,
}


# Regimes that default to the multi-cluster chain driver. S1-S4 stay on
# the T=1 fast path because their reproducible component is a single
# Gaussian copula by construction; S5 / S5-sparse are the heavy-tail
# mixture, so we route them through run_multi_chain_multi.
_T_GT_1_REGIMES: frozenset[str] = frozenset({"S5", "S5-sparse"})


def _resolve_chain_type(regime: str, chain_type: str) -> str:
    """Resolve the chain-type request, handling ``"auto"`` as regime-aware default."""
    if chain_type == "auto":
        return "T>1" if regime in _T_GT_1_REGIMES else "T=1"
    if chain_type not in ("T=1", "T>1"):
        raise ValueError(f"chain_type must be 'auto', 'T=1', or 'T>1'; got {chain_type!r}")
    return chain_type


@dataclass(frozen=True)
class FitResult:
    """Output of :func:`_fit_to_idr` — the per-feature posterior plus summaries.

    Carries everything the downstream Sun-Cai / Bayes-mFDR steps need,
    so a single chain run can be evaluated at many $\\alpha$ levels
    without re-fitting.

    Attributes
    ----------
    idr
        Posterior local idr vector, shape $(n,)$, values in $[0, 1]$.
    pi_posterior_mean
        Posterior mean of the reproducible mass.
    rho_posterior_mean
        Posterior mean of the Gaussian-copula correlation. Set to
        :data:`math.nan` under the multi-cluster path, where no single
        $\\rho$ summarises the mixture.
    T_posterior_mean
        Posterior mean of the active cluster count $T$. NaN for the
        T=1 fast path; meaningful only for ``chain_type == "T>1"``.
    method
        Human-readable method label that downstream rows record. One of
        ``"PY-IDR (MCMC, T=1)"`` or ``"PY-IDR (MCMC, T>1)"``.
    num_chains, num_samples_per_chain
        Chain configuration — recorded so the eventual row reflects it.
    """

    idr: jnp.ndarray
    pi_posterior_mean: float
    rho_posterior_mean: float
    T_posterior_mean: float
    method: str
    num_chains: int
    num_samples_per_chain: int
    walltime_s: float


def _fit_to_idr(
    sim: SimulationResult,
    *,
    num_chains: int,
    n_warmup: int,
    n_samples: int,
    nuts_warmup: int,
    M: int,
    chain_type: str = "auto",
    T_max: int = 10,
    H: int = 5,
    py_hyperparam_warmup: int = 50,
    em_init: bool = False,
) -> FitResult:
    """Rank → chain → posterior local idr; no decision step yet.

    Pulled out so a sweep can fit one chain and evaluate at many
    $\\alpha$ values without paying the chain cost per $\\alpha$.

    Parameters
    ----------
    sim
        Simulation result. ``sim.regime`` determines the default chain
        when ``chain_type == "auto"``.
    chain_type
        One of ``"auto"``, ``"T=1"``, ``"T>1"``. ``"auto"`` picks the
        multi-cluster path for S5 / S5-sparse and the T=1 path for
        S1-S4.
    T_max, H, py_hyperparam_warmup
        Multi-cluster-only knobs. Ignored on the T=1 path.
    """
    chain_type_resolved = _resolve_chain_type(sim.regime, chain_type)
    K = sim.K

    # Build the pilot CDF from the simulated X via the empirical-rank transform.
    F0 = jnp.stack(
        [empirical_rank_transform(sim.X[:, j]) for j in range(K)],
        axis=1,
    )
    key = jax.random.PRNGKey(sim.seed)

    pi_mean: float
    rho_mean: float
    T_mean: float
    method: str
    z_samples: np.ndarray | None

    chain_start = time.perf_counter()
    if chain_type_resolved == "T=1":
        result_simple = run_multi_chain_simple(
            F0=F0,
            M=M,
            key=key,
            num_chains=num_chains,
            n_warmup=n_warmup,
            n_samples=n_samples,
            nuts_warmup=nuts_warmup,
            em_init=em_init,
        )
        # Block until JAX traces have materialised so the timing reflects
        # the full chain cost rather than just dispatch.
        jax.block_until_ready(result_simple.idata.posterior["pi"].values)
        pi_samples = np.asarray(result_simple.idata.posterior["pi"]).reshape(-1)
        rho_samples = np.asarray(result_simple.idata.posterior["rho"]).reshape(-1)
        pi_mean = float(np.mean(pi_samples))
        rho_mean = float(np.mean(rho_samples))
        T_mean = float("nan")
        method = "PY-IDR (MCMC, T=1)"
        z_samples = result_simple.z_samples
    else:
        result_multi = run_multi_chain_multi(
            F0=F0,
            M=M,
            key=key,
            num_chains=num_chains,
            n_warmup=n_warmup,
            n_samples=n_samples,
            T_max=T_max,
            H=H,
            nuts_warmup=nuts_warmup,
            py_hyperparam_warmup=py_hyperparam_warmup,
            em_init=em_init,
        )
        jax.block_until_ready(result_multi.idata.posterior["pi"].values)
        pi_samples = np.asarray(result_multi.idata.posterior["pi"]).reshape(-1)
        T_samples = np.asarray(result_multi.idata.posterior["T"]).reshape(-1)
        pi_mean = float(np.mean(pi_samples))
        rho_mean = float("nan")  # No single rho under the mixture.
        T_mean = float(np.mean(T_samples))
        method = "PY-IDR (MCMC, T>1)"
        z_samples = result_multi.z_samples
    walltime_s = float(time.perf_counter() - chain_start)

    if z_samples is None:
        raise RuntimeError(
            f"{chain_type_resolved} chain driver did not produce z_samples; "
            "this should not happen with the current run_multi_chain_* "
            "implementations."
        )

    # Proper Monte Carlo posterior local idr from the Z trace. The chain
    # drivers accumulate z_samples of shape (num_chains, n_samples, n);
    # we flatten the first two dims so decision.local_idr.from_mcmc
    # averages over all chains.
    z_flat = z_samples.reshape(-1, z_samples.shape[-1])
    idr = from_mcmc(jnp.asarray(z_flat))

    return FitResult(
        idr=idr,
        pi_posterior_mean=pi_mean,
        rho_posterior_mean=rho_mean,
        T_posterior_mean=T_mean,
        method=method,
        num_chains=num_chains,
        num_samples_per_chain=n_samples,
        walltime_s=walltime_s,
    )


def _make_row(sim: SimulationResult, fit: FitResult, alpha: float) -> ReplicateRow:
    """Apply Sun-Cai at one $\\alpha$ and package the per-replicate row."""
    recovery = evaluate_recovery(fit.idr, sim.true_Z, alpha=alpha)
    return ReplicateRow(
        regime=sim.regime,
        K=sim.K,
        n=sim.n,
        replicate_seed=sim.seed,
        method=fit.method,
        alpha=alpha,
        pi_true=sim.true_pi,
        pi_posterior_mean=fit.pi_posterior_mean,
        rho_true=sim.true_rho,
        rho_posterior_mean=fit.rho_posterior_mean,
        T_posterior_mean=fit.T_posterior_mean,
        k_alpha=recovery.k_alpha,
        realized_fdr=recovery.realized_fdr,
        power=recovery.power,
        num_chains=fit.num_chains,
        num_samples_per_chain=fit.num_samples_per_chain,
        walltime_s=fit.walltime_s,
    )


def _fit_and_evaluate(
    sim: SimulationResult,
    *,
    alpha: float,
    num_chains: int,
    n_warmup: int,
    n_samples: int,
    nuts_warmup: int,
    M: int,
    chain_type: str = "auto",
    T_max: int = 10,
    H: int = 5,
    py_hyperparam_warmup: int = 50,
    em_init: bool = False,
) -> ReplicateRow:
    """Single-alpha convenience wrapper around :func:`_fit_to_idr` + :func:`_make_row`."""
    fit = _fit_to_idr(
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
    return _make_row(sim, fit, alpha)


def run_replicate(
    regime: str,
    *,
    K: int,
    n: int,
    seed: int = 0,
    alpha: float = 0.05,
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
) -> ReplicateRow:
    """Run one end-to-end replicate of any S1–S5 regime.

    Dispatch order:

    1. Look up the regime simulator in :data:`_SIMULATORS`.
    2. Call it with ``K, n, seed`` plus any regime-specific kwargs.
    3. Resolve ``chain_type``: ``"auto"`` picks the multi-cluster path
       for S5 / S5-sparse and the T=1 fast path for S1-S4.
    4. Pipe through the shared rank → chain → idr → Sun--Cai pipeline.

    Parameters
    ----------
    regime
        One of the keys in :data:`_SIMULATORS` (``"S1"``..``"S5-sparse"``).
    K, n, seed
        Forwarded to the simulator. ``seed`` is also used as the chain
        PRNGKey so a replicate is fully reproducible from this one int.
    alpha
        Sun--Cai operating level. Default 0.05 matches §4.4.
    num_chains, n_warmup, n_samples, nuts_warmup
        Forwarded to the chain driver.
    M
        Bernstein degree (held fixed across the chain for this commit).
    chain_type
        ``"auto"`` (default), ``"T=1"``, or ``"T>1"``. ``"auto"``
        selects the multi-cluster path for S5 / S5-sparse and the T=1
        path for S1-S4.
    T_max, H, py_hyperparam_warmup
        Multi-cluster knobs. ``T_max`` caps the active cluster count
        (default 10). ``H`` is the auxiliary-atom count per Pólya-urn
        step (default 5). All three are ignored on the T=1 path.
    scenario_kwargs
        Regime-specific overrides forwarded to the simulator. Examples:
        ``{"pi_star": 0.30, "rho": 0.85}`` for S1/S2,
        ``{"sigma_grid": (1.0, 1.5, 2.0)}`` for S3,
        ``{"skewness_grid": (-1.0, 0.0, 1.0)}`` for S4,
        ``{"tau": 0.6, "sigma_grid": (1.0, 1.5, 2.0)}`` for S5/S5-sparse.

    Returns
    -------
    A :class:`ReplicateRow` ready to drop into a long-format CSV.

    Raises
    ------
    ValueError
        If ``regime`` is not one of the supported labels.
    """
    if regime not in _SIMULATORS:
        raise ValueError(f"unknown regime {regime!r}; expected one of {sorted(_SIMULATORS)}")
    simulator = _SIMULATORS[regime]
    extra = dict(scenario_kwargs or {})
    sim = simulator(K=K, n=n, seed=seed, **extra)
    return _fit_and_evaluate(
        sim,
        alpha=alpha,
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


def run_replicate_S1(
    *,
    K: int,
    n: int,
    pi_star: float = 0.30,
    rho: float = 0.85,
    seed: int = 0,
    alpha: float = 0.05,
    num_chains: int = 2,
    n_warmup: int = 50,
    n_samples: int = 100,
    nuts_warmup: int = 20,
    M: int = 5,
) -> ReplicateRow:
    """End-to-end S1 replicate (thin wrapper around :func:`run_replicate`)."""
    return run_replicate(
        "S1",
        K=K,
        n=n,
        seed=seed,
        alpha=alpha,
        num_chains=num_chains,
        n_warmup=n_warmup,
        n_samples=n_samples,
        nuts_warmup=nuts_warmup,
        M=M,
        scenario_kwargs={"pi_star": pi_star, "rho": rho},
    )


def run_replicate_S2(
    *,
    K: int,
    n: int,
    pi_star: float = 0.30,
    rho: float = 0.40,
    seed: int = 0,
    alpha: float = 0.05,
    num_chains: int = 2,
    n_warmup: int = 50,
    n_samples: int = 100,
    nuts_warmup: int = 20,
    M: int = 5,
) -> ReplicateRow:
    """End-to-end S2 replicate — weak signal (default ``rho = 0.40``)."""
    return run_replicate(
        "S2",
        K=K,
        n=n,
        seed=seed,
        alpha=alpha,
        num_chains=num_chains,
        n_warmup=n_warmup,
        n_samples=n_samples,
        nuts_warmup=nuts_warmup,
        M=M,
        scenario_kwargs={"pi_star": pi_star, "rho": rho},
    )


def run_replicate_S3(
    *,
    K: int,
    n: int,
    pi_star: float = 0.10,
    rho: float = 0.85,
    seed: int = 0,
    alpha: float = 0.05,
    num_chains: int = 2,
    n_warmup: int = 50,
    n_samples: int = 100,
    nuts_warmup: int = 20,
    M: int = 5,
    sigma_grid: tuple[float, ...] = (1.0, 1.5, 2.0),
) -> ReplicateRow:
    """End-to-end S3 replicate — sparse signal with per-replicate scale heterogeneity.

    The rank pilot collapses per-replicate scale to a uniform, so the chain
    only sees the latent copula; the marginal posterior summaries
    (``pi_posterior_mean``, ``rho_posterior_mean``) should remain near the
    truth even though the raw marginals differ across replicates.
    """
    return run_replicate(
        "S3",
        K=K,
        n=n,
        seed=seed,
        alpha=alpha,
        num_chains=num_chains,
        n_warmup=n_warmup,
        n_samples=n_samples,
        nuts_warmup=nuts_warmup,
        M=M,
        scenario_kwargs={"pi_star": pi_star, "rho": rho, "sigma_grid": sigma_grid},
    )


def run_replicate_S4(
    *,
    K: int,
    n: int,
    pi_star: float = 0.30,
    rho: float = 0.85,
    seed: int = 0,
    alpha: float = 0.05,
    num_chains: int = 2,
    n_warmup: int = 50,
    n_samples: int = 100,
    nuts_warmup: int = 20,
    M: int = 5,
    skewness_grid: tuple[float, ...] = (-1.0, 0.0, 1.0),
) -> ReplicateRow:
    """End-to-end S4 replicate — skewed marginal mismatch via skew-normal marginals.

    The latent copula structure matches S1 ($\\rho = 0.85$, $\\pi^* = 0.30$);
    only the observed marginals differ. The rank pilot strips marginal shape,
    so the chain's posterior on ($\\pi$, $\\rho$) should still be near truth
    even though the raw marginals are asymmetric across replicates.
    """
    return run_replicate(
        "S4",
        K=K,
        n=n,
        seed=seed,
        alpha=alpha,
        num_chains=num_chains,
        n_warmup=n_warmup,
        n_samples=n_samples,
        nuts_warmup=nuts_warmup,
        M=M,
        scenario_kwargs={
            "pi_star": pi_star,
            "rho": rho,
            "skewness_grid": skewness_grid,
        },
    )


def run_replicate_S5(
    *,
    K: int,
    n: int,
    pi_star: float = 0.30,
    tau: float = 0.6,
    seed: int = 0,
    alpha: float = 0.05,
    num_chains: int = 2,
    n_warmup: int = 50,
    n_samples: int = 100,
    nuts_warmup: int = 20,
    M: int = 5,
    sigma_grid: tuple[float, ...] = (1.0, 1.5, 2.0),
    nu: int = 5,
    chain_type: str = "auto",
    T_max: int = 10,
    H: int = 5,
    py_hyperparam_warmup: int = 50,
) -> ReplicateRow:
    """End-to-end S5 replicate — heavy-tail mixture of $t_5$ / Clayton / Gumbel atoms.

    Routes through the **multi-cluster** chain driver by default
    (``chain_type="auto"`` resolves to ``"T>1"`` for S5). The latent
    copula is an equal-weight mixture of three families calibrated to
    Kendall's $\\tau$ (default 0.6, matching the report's S5 design);
    marginals are per-replicate scaled $t_\\nu$. The chain learns the
    number of active clusters $T$ from the data via the Pitman-Yor
    prior.

    For ablation studies, pass ``chain_type="T=1"`` to force the
    deliberately misspecified single-Gaussian fit — useful for
    establishing the baseline that PY-IDR's multi-cluster atom is
    supposed to beat.

    Parameters
    ----------
    K, n, seed, alpha, num_chains, n_warmup, n_samples, nuts_warmup, M
        Standard sweep parameters; see :func:`run_replicate_S1`.
    pi_star
        Reproducible mass. Default 0.30 matches S5 in Table 3.
    tau
        Matched Kendall's $\\tau$ for the three families.
    sigma_grid
        Per-replicate $t_\\nu$ marginal scale; cycled mod $K$.
    nu
        Degrees of freedom for the elliptical copula and the marginals.
    chain_type
        ``"auto"`` (default; resolves to T>1 here), ``"T=1"``, or
        ``"T>1"``.
    T_max, H, py_hyperparam_warmup
        Multi-cluster driver knobs forwarded to
        :func:`run_multi_chain_multi`.
    """
    return run_replicate(
        "S5",
        K=K,
        n=n,
        seed=seed,
        alpha=alpha,
        num_chains=num_chains,
        n_warmup=n_warmup,
        n_samples=n_samples,
        nuts_warmup=nuts_warmup,
        M=M,
        chain_type=chain_type,
        T_max=T_max,
        H=H,
        py_hyperparam_warmup=py_hyperparam_warmup,
        scenario_kwargs={
            "pi_star": pi_star,
            "tau": tau,
            "sigma_grid": sigma_grid,
            "nu": nu,
        },
    )


def run_replicate_S5_sparse(
    *,
    K: int,
    n: int,
    pi_star: float = 0.10,
    tau: float = 0.6,
    seed: int = 0,
    alpha: float = 0.05,
    num_chains: int = 2,
    n_warmup: int = 50,
    n_samples: int = 100,
    nuts_warmup: int = 20,
    M: int = 5,
    sigma_grid: tuple[float, ...] = (1.0, 1.5, 2.0),
    nu: int = 5,
    chain_type: str = "auto",
    T_max: int = 10,
    H: int = 5,
    py_hyperparam_warmup: int = 50,
) -> ReplicateRow:
    """End-to-end S5-sparse replicate — S5 mixture with reduced reproducible mass.

    Identical to :func:`run_replicate_S5` except the default
    ``pi_star`` drops from 0.30 to 0.10. The report's S5-sparse cell
    is the planned ROC/PR stress test. Routes through the
    multi-cluster chain driver by default.
    """
    return run_replicate(
        "S5-sparse",
        K=K,
        n=n,
        seed=seed,
        alpha=alpha,
        num_chains=num_chains,
        n_warmup=n_warmup,
        n_samples=n_samples,
        nuts_warmup=nuts_warmup,
        M=M,
        chain_type=chain_type,
        T_max=T_max,
        H=H,
        py_hyperparam_warmup=py_hyperparam_warmup,
        scenario_kwargs={
            "pi_star": pi_star,
            "tau": tau,
            "sigma_grid": sigma_grid,
            "nu": nu,
        },
    )


# ---- comparator: Vanilla IDR (Li-Brown-Huang-Bickel 2011) ----------------------------


_VANILLA_IDR_METHOD_LABEL = "Vanilla IDR (pairwise)"


def run_replicate_vanilla_idr(
    *,
    regime: str,
    K: int,
    n: int,
    seed: int = 0,
    alpha: float = 0.05,
    aggregation: str = "mean",
    pi_init: float = 0.5,
    rho_init: float = 0.5,
    max_iter: int = 200,
    tol: float = 1e-5,
    scenario_kwargs: dict[str, Any] | None = None,
) -> ReplicateRow:
    """End-to-end Vanilla IDR (Li et al. 2011) replicate for any of the S1-S5 regimes.

    Drops into the same long-format CSV as PY-IDR rows with
    ``method = "Vanilla IDR (pairwise)"``. PY-IDR-specific columns
    (``T_posterior_mean``, ``rho_posterior_mean`` when the regime has
    no single rho, etc.) follow the same NaN-where-undefined convention.

    Parameters
    ----------
    regime
        Simulation regime label ("S1" .. "S5-sparse").
    K, n, seed
        Forwarded to the regime's simulator. ``seed`` also drives
        determinism — Vanilla IDR's EM is deterministic given the data.
    alpha
        Sun-Cai operating level.
    aggregation
        Pairwise aggregation rule for $K \\ge 3$: ``"mean"`` (default)
        or ``"max"``. See :func:`py_idr.comparators.vanilla_idr.fit_vanilla_idr_pairwise`.
    pi_init, rho_init, max_iter, tol
        EM hyperparameters.
    scenario_kwargs
        Regime-specific kwargs forwarded to the simulator.

    Returns
    -------
    A :class:`ReplicateRow` with ``method = "Vanilla IDR (pairwise)"``.

    Raises
    ------
    ValueError
        If ``regime`` is not a known simulation regime.
    """
    # Late import keeps the heavy comparators dep out of the simulation
    # top-level surface.
    from py_idr.comparators.vanilla_idr import fit_vanilla_idr_pairwise

    if regime not in _SIMULATORS:
        raise ValueError(f"unknown regime {regime!r}; expected one of {sorted(_SIMULATORS)}")
    simulator = _SIMULATORS[regime]
    extra = dict(scenario_kwargs or {})
    sim = simulator(K=K, n=n, seed=seed, **extra)

    t0 = time.perf_counter()
    idr = fit_vanilla_idr_pairwise(
        np.asarray(sim.X),
        aggregation=aggregation,  # type: ignore[arg-type]
        pi_init=pi_init,
        rho_init=rho_init,
        max_iter=max_iter,
        tol=tol,
    )
    walltime = float(time.perf_counter() - t0)

    recovery = evaluate_recovery(jnp.asarray(idr), sim.true_Z, alpha=alpha)

    # Vanilla IDR has no "chains" — fill the chain-config columns with 1 /
    # max_iter as a stand-in so the CSV schema stays consistent.
    return ReplicateRow(
        regime=sim.regime,
        K=K,
        n=n,
        replicate_seed=sim.seed,
        method=_VANILLA_IDR_METHOD_LABEL,
        alpha=alpha,
        pi_true=sim.true_pi,
        pi_posterior_mean=float("nan"),  # Vanilla IDR fits per pair; not a single pi.
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


# ---- comparator: MaxRank (rank-based baseline) ---------------------------------------


_MAXRANK_METHOD_LABEL = "MaxRank"


def run_replicate_maxrank(
    *,
    regime: str,
    K: int,
    n: int,
    seed: int = 0,
    alpha: float = 0.05,
    tie_method: str = "average",
    scenario_kwargs: dict[str, Any] | None = None,
) -> ReplicateRow:
    """End-to-end MaxRank replicate for any of the S1-S5 regimes.

    Same long-format ReplicateRow shape as PY-IDR / Vanilla IDR, with
    ``method = "MaxRank"`` and the parametric posterior summaries
    (``pi_posterior_mean``, ``rho_posterior_mean``, ``T_posterior_mean``)
    populated as NaN.

    Parameters
    ----------
    regime
        Simulation regime label ("S1" .. "S5-sparse").
    K, n, seed
        Forwarded to the regime's simulator.
    alpha
        Sun-Cai operating level.
    tie_method
        Tie-breaking rule for the per-replicate ranks. Forwarded to
        :func:`py_idr.comparators.maxrank.fit_maxrank`.
    scenario_kwargs
        Regime-specific kwargs forwarded to the simulator.

    Returns
    -------
    A :class:`ReplicateRow` with ``method = "MaxRank"``.
    """
    from py_idr.comparators.maxrank import fit_maxrank

    if regime not in _SIMULATORS:
        raise ValueError(f"unknown regime {regime!r}; expected one of {sorted(_SIMULATORS)}")
    simulator = _SIMULATORS[regime]
    extra = dict(scenario_kwargs or {})
    sim = simulator(K=K, n=n, seed=seed, **extra)

    t0 = time.perf_counter()
    idr = fit_maxrank(np.asarray(sim.X), tie_method=tie_method)
    walltime = float(time.perf_counter() - t0)

    recovery = evaluate_recovery(jnp.asarray(idr), sim.true_Z, alpha=alpha)

    return ReplicateRow(
        regime=sim.regime,
        K=K,
        n=n,
        replicate_seed=sim.seed,
        method=_MAXRANK_METHOD_LABEL,
        alpha=alpha,
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
