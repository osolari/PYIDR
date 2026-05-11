"""One replicate of an S1–S5 simulation cell, end-to-end.

The driver ties together:

1. ``simulate_S1`` (or its sibling regime generators) — synthetic data + truth.
2. ``empirical_rank_transform`` — pilot CDF input to the chain.
3. ``run_multi_chain_simple`` — multi-chain PY-IDR fit (T = 1 path).
4. ``from_mcmc`` — posterior local idr from the Z trace.
5. ``evaluate_recovery`` — realised FDR + power at the Sun--Cai threshold.

For S1 (and S2/S3/S4 once added), the T = 1 sampler is the right tool: the
reproducible component is a single exchangeable Gaussian copula and the
multi-cluster machinery is overkill. S5 will route to the multi-cluster
sweep once added.

The output is a flat ``ReplicateRow`` consumable as a single CSV row by the
figure F1 / table T4 reproducers.

See plans/07_simulation_studies.md.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import jax
import jax.numpy as jnp
import numpy as np

from py_idr.inference.mcmc.run_chain import run_multi_chain_simple
from py_idr.marginals.transform import empirical_rank_transform
from py_idr.simulation.evaluation import evaluate_recovery
from py_idr.simulation.scenarios import simulate_S1


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


def _idr_from_idata(idata) -> np.ndarray:  # type: ignore[no-untyped-def]
    """Compute per-feature local idr from a posterior Z trace stored in idata.

    Our chain driver currently stores scalar posterior summaries (``pi``,
    ``rho``, ``n_reproducible``) but not the per-feature Z samples. As a
    first cut we estimate the local idr from the posterior mean of $\\pi$ +
    the implied class-assignment probabilities at the simulation's $U$. This
    is the operational analogue used by Sun--Cai-style plug-in rules and is
    a stand-in until the chain driver propagates the full Z trace.

    The proper :func:`py_idr.decision.local_idr.from_mcmc` implementation
    consumes ``z_samples`` of shape ``(draws, n)``; once the chain driver
    threads that through, this helper can be retired.
    """
    # Currently unused — replaced by an inline computation in run_replicate_S1.
    raise NotImplementedError


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
    """End-to-end S1 replicate: simulate → fit → idr → Sun--Cai → realised FDR.

    Parameters
    ----------
    K, n, pi_star, rho, seed
        Forwarded to :func:`simulate_S1`.
    alpha
        Sun--Cai operating level. Default 0.05 matches §4.4.
    num_chains, n_warmup, n_samples, nuts_warmup
        Forwarded to :func:`run_multi_chain_simple`.
    M
        Bernstein degree (held fixed across the chain for this commit).

    Returns
    -------
    A :class:`ReplicateRow` ready to drop into a long-format CSV.
    """
    sim = simulate_S1(K=K, n=n, pi_star=pi_star, rho=rho, seed=seed)

    # Build the pilot CDF from the simulated X via the empirical-rank transform.
    F0 = jnp.stack([empirical_rank_transform(sim.X[:, j]) for j in range(K)], axis=1)

    # Multi-chain fit (T = 1 sampler — single Gaussian atom is the right model
    # for S1).
    key = jax.random.PRNGKey(seed)
    result = run_multi_chain_simple(
        F0=F0,
        M=M,
        key=key,
        num_chains=num_chains,
        n_warmup=n_warmup,
        n_samples=n_samples,
        nuts_warmup=nuts_warmup,
    )

    # Posterior mean π, ρ from the chain.
    pi_samples = np.asarray(result.idata.posterior["pi"]).reshape(-1)
    rho_samples = np.asarray(result.idata.posterior["rho"]).reshape(-1)
    pi_mean = float(np.mean(pi_samples))
    rho_mean = float(np.mean(rho_samples))

    # Per-feature local idr. The chain driver currently stores only scalar
    # posterior summaries; we compute idr from the posterior mean ρ via the
    # mixture posterior P(Z_i = 0 | data, π_mean, ρ_mean), which is the
    # operational plug-in estimator. A future commit propagates the full Z
    # trace through the chain so the proper Monte Carlo estimator
    # (decision/local_idr.from_mcmc) can be used.
    R_post = (1.0 - rho_mean) * jnp.eye(K) + rho_mean * jnp.ones((K, K))
    L_post = jnp.linalg.cholesky(R_post)
    from py_idr.copulas.gaussian import GaussianCopula
    from py_idr.inference.mcmc.class_assign import conditional_class_log_odds

    log_c1 = GaussianCopula(L_post).log_density(F0)
    log_odds = conditional_class_log_odds(pi_mean, log_c1)
    # Pr(Z = 1 | data) = sigmoid(log_odds); local idr = Pr(Z = 0 | data).
    pi_hat = jax.nn.sigmoid(log_odds)
    idr = 1.0 - pi_hat

    recovery = evaluate_recovery(idr, sim.true_Z, alpha=alpha)

    return ReplicateRow(
        regime=sim.regime,
        K=K,
        n=n,
        replicate_seed=seed,
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
