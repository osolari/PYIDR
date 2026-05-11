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

from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from py_idr.decision.local_idr import from_mcmc
from py_idr.inference.mcmc.run_chain import run_multi_chain_simple
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
    """Flat per-replicate record consumed by figure / table reproducers."""

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
    k_alpha: int
    realized_fdr: float
    power: float
    num_chains: int
    num_samples_per_chain: int

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


def _fit_and_evaluate(
    sim: SimulationResult,
    *,
    alpha: float,
    num_chains: int,
    n_warmup: int,
    n_samples: int,
    nuts_warmup: int,
    M: int,
) -> ReplicateRow:
    """Shared post-simulation pipeline: rank → chain → idr → Sun--Cai → row.

    Pulled out of the per-regime drivers so the inference path stays in one
    place. The regime label, K, n, and seed flow through ``sim`` unchanged.
    """
    K = sim.K
    n = sim.n

    # Build the pilot CDF from the simulated X via the empirical-rank transform.
    # The rank pilot is the same regardless of regime — the marginal model in
    # the chain learns the per-replicate distortion.
    F0 = jnp.stack(
        [empirical_rank_transform(sim.X[:, j]) for j in range(K)],
        axis=1,
    )

    # Multi-chain fit (T = 1 sampler — single Gaussian atom is the right model
    # for S1–S4; only S5 needs the multi-cluster sweep).
    key = jax.random.PRNGKey(sim.seed)
    result = run_multi_chain_simple(
        F0=F0,
        M=M,
        key=key,
        num_chains=num_chains,
        n_warmup=n_warmup,
        n_samples=n_samples,
        nuts_warmup=nuts_warmup,
    )

    # Posterior mean π, ρ from the chain (scalar summaries).
    pi_samples = np.asarray(result.idata.posterior["pi"]).reshape(-1)
    rho_samples = np.asarray(result.idata.posterior["rho"]).reshape(-1)
    pi_mean = float(np.mean(pi_samples))
    rho_mean = float(np.mean(rho_samples))

    # Proper Monte Carlo posterior local idr from the Z trace. The chain driver
    # accumulates z_samples of shape (num_chains, n_samples, n); we flatten the
    # first two dims so decision.local_idr.from_mcmc averages over all chains.
    z_samples = result.z_samples  # (num_chains, n_samples, n)
    if z_samples is None:
        raise RuntimeError(
            "chain driver did not produce z_samples; this should not happen with "
            "the current run_multi_chain_simple implementation."
        )
    z_flat = z_samples.reshape(-1, z_samples.shape[-1])  # (chains * samples, n)
    idr = from_mcmc(jnp.asarray(z_flat))

    recovery = evaluate_recovery(idr, sim.true_Z, alpha=alpha)

    return ReplicateRow(
        regime=sim.regime,
        K=K,
        n=n,
        replicate_seed=sim.seed,
        method="PY-IDR (MCMC, T=1)",
        alpha=alpha,
        pi_true=sim.true_pi,
        pi_posterior_mean=pi_mean,
        rho_true=sim.true_rho,
        rho_posterior_mean=rho_mean,
        k_alpha=recovery.k_alpha,
        realized_fdr=recovery.realized_fdr,
        power=recovery.power,
        num_chains=num_chains,
        num_samples_per_chain=n_samples,
    )


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
    scenario_kwargs: dict[str, Any] | None = None,
) -> ReplicateRow:
    """Run one end-to-end replicate of any S1–S4 regime.

    Dispatch order:

    1. Look up the regime simulator in :data:`_SIMULATORS`.
    2. Call it with ``K, n, seed`` plus any regime-specific kwargs (e.g.
       ``pi_star``, ``rho``, ``sigma_grid``, ``skewness_grid``).
    3. Pipe through the shared rank → chain → idr → Sun--Cai pipeline.

    Parameters
    ----------
    regime
        One of ``"S1"``, ``"S2"``, ``"S3"``, ``"S4"``. Selects the
        scenario generator.
    K, n, seed
        Forwarded to the simulator. ``seed`` is also used as the chain
        PRNGKey so a replicate is fully reproducible from this one int.
    alpha
        Sun--Cai operating level. Default 0.05 matches §4.4.
    num_chains, n_warmup, n_samples, nuts_warmup
        Forwarded to :func:`run_multi_chain_simple`.
    M
        Bernstein degree (held fixed across the chain for this commit).
    scenario_kwargs
        Regime-specific overrides forwarded to the simulator. Examples:
        ``{"pi_star": 0.30, "rho": 0.85}`` for S1/S2,
        ``{"sigma_grid": (1.0, 1.5, 2.0)}`` for S3,
        ``{"skewness_grid": (-1.0, 0.0, 1.0)}`` for S4.

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
) -> ReplicateRow:
    """End-to-end S5 replicate — heavy-tail mixture of $t_5$ / Clayton / Gumbel atoms.

    .. note::
       For S5 the T = 1 fast path is the *wrong* model — the
       reproducible data come from a three-family mixture and a single
       Gaussian-copula cluster cannot capture them. Running this driver
       today gives a *baseline* (deliberately misspecified) fit; the
       expected behaviour is biased posterior summaries with realised
       FDR drifting above the nominal $\\alpha$. The multi-cluster chain
       driver (planned in plan 04 follow-on) is what should consume S5.

    The function is still useful right now for: (a) sweep-CSV plumbing
    that needs *some* S5 row to wire end-to-end, and (b) ablation
    studies that explicitly compare T = 1 vs T > 1 on S5.

    The latent copula is parameterised by Kendall's $\\tau$ (default
    0.6, matching the report's S5 design); marginals are per-replicate
    scaled $t_\\nu$.

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
) -> ReplicateRow:
    """End-to-end S5-sparse replicate — S5 mixture with reduced reproducible mass.

    Identical to :func:`run_replicate_S5` except the default
    ``pi_star`` drops from 0.30 to 0.10. The report's S5-sparse cell
    is the planned ROC/PR stress test.
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
        scenario_kwargs={
            "pi_star": pi_star,
            "tau": tau,
            "sigma_grid": sigma_grid,
            "nu": nu,
        },
    )
