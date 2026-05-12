"""Tests for the Z trace plumbing through the chain driver."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from py_idr.decision.local_idr import from_mcmc
from py_idr.inference.mcmc.run_chain import (
    ChainResult,
    MultiChainResult,
    run_chain_simple,
    run_multi_chain_simple,
)
from py_idr.marginals.transform import empirical_rank_transform


def _build_F0(n: int, K: int, seed: int) -> jnp.ndarray:
    rng = np.random.default_rng(seed)
    U = rng.uniform(size=(n, K))
    return jnp.stack(
        [empirical_rank_transform(jnp.asarray(U[:, j])) for j in range(K)],
        axis=1,
    )


@pytest.mark.integration
def test_run_chain_simple_z_trace_shape() -> None:
    """Single-chain z_samples have shape (n_samples, n) with int values in {0, 1}."""
    n, K = 20, 2
    F0 = _build_F0(n, K, seed=0)
    result = run_chain_simple(
        F0=F0,
        M=5,
        key=jax.random.PRNGKey(0),
        n_warmup=2,
        n_samples=5,
        nuts_warmup=10,
    )
    assert isinstance(result, ChainResult)
    assert result.z_samples is not None
    assert result.z_samples.shape == (5, n)
    assert set(np.unique(result.z_samples)).issubset({0, 1})


@pytest.mark.integration
def test_run_multi_chain_simple_z_trace_shape() -> None:
    """Multi-chain z_samples have shape (num_chains, n_samples, n)."""
    n, K = 15, 2
    F0 = _build_F0(n, K, seed=1)
    num_chains = 3
    result = run_multi_chain_simple(
        F0=F0,
        M=5,
        key=jax.random.PRNGKey(0),
        num_chains=num_chains,
        n_warmup=2,
        n_samples=4,
        nuts_warmup=10,
    )
    assert isinstance(result, MultiChainResult)
    assert result.z_samples is not None
    assert result.z_samples.shape == (num_chains, 4, n)
    assert set(np.unique(result.z_samples)).issubset({0, 1})


@pytest.mark.integration
def test_z_trace_feeds_from_mcmc() -> None:
    """The Z trace's shape is exactly what decision.local_idr.from_mcmc expects."""
    n, K = 10, 2
    F0 = _build_F0(n, K, seed=2)
    result = run_chain_simple(
        F0=F0,
        M=5,
        key=jax.random.PRNGKey(0),
        n_warmup=2,
        n_samples=8,
        nuts_warmup=10,
    )
    # from_mcmc takes (draws, n) → returns (n,) of idr values.
    idr = from_mcmc(jnp.asarray(result.z_samples))
    assert idr.shape == (n,)
    assert bool(jnp.all((idr >= 0.0) & (idr <= 1.0)))


@pytest.mark.integration
def test_z_trace_idr_converges_to_per_iteration_mean() -> None:
    """from_mcmc on the Z trace equals 1 - mean(Z) feature-wise (Monte Carlo definition)."""
    n, K = 12, 2
    F0 = _build_F0(n, K, seed=3)
    result = run_chain_simple(
        F0=F0,
        M=5,
        key=jax.random.PRNGKey(0),
        n_warmup=2,
        n_samples=10,
        nuts_warmup=10,
    )
    z = result.z_samples  # (draws, n)
    idr_via_mc = from_mcmc(jnp.asarray(z))
    idr_via_mean = 1.0 - np.mean(z, axis=0)
    assert np.allclose(np.asarray(idr_via_mc), idr_via_mean, atol=1e-6)


@pytest.mark.integration
def test_z_trace_consistent_with_n_reproducible_counts() -> None:
    """sum(z[i, :]) per draw matches the n_reproducible value in idata."""
    n, K = 12, 2
    F0 = _build_F0(n, K, seed=4)
    result = run_chain_simple(
        F0=F0,
        M=5,
        key=jax.random.PRNGKey(0),
        n_warmup=2,
        n_samples=8,
        nuts_warmup=10,
    )
    per_draw_sum = np.sum(result.z_samples, axis=1)  # (draws,)
    posterior_n_rep = np.asarray(result.idata.posterior["n_reproducible"]).reshape(-1)
    assert np.array_equal(per_draw_sum, posterior_n_rep)


@pytest.mark.integration
def test_run_chain_simple_is_deterministic_in_key() -> None:
    """Same PRNG key produces bit-identical chains.

    Correctness guard: any subkey reuse or non-deterministic step
    would break this. After the W8.1 audit fixes (commit 4e98a63),
    determinism holds across pi / rho / Z trace and all derived
    quantities.
    """
    n, K = 12, 2
    F0 = _build_F0(n, K, seed=5)
    kwargs = dict(F0=F0, M=5, key=jax.random.PRNGKey(7), n_warmup=3, n_samples=8, nuts_warmup=10)
    r1 = run_chain_simple(**kwargs)  # type: ignore[arg-type]
    r2 = run_chain_simple(**kwargs)  # type: ignore[arg-type]
    pi_1 = np.asarray(r1.idata.posterior["pi"]).reshape(-1)
    pi_2 = np.asarray(r2.idata.posterior["pi"]).reshape(-1)
    rho_1 = np.asarray(r1.idata.posterior["rho"]).reshape(-1)
    rho_2 = np.asarray(r2.idata.posterior["rho"]).reshape(-1)
    assert np.array_equal(pi_1, pi_2)
    assert np.array_equal(rho_1, rho_2)
    assert np.array_equal(r1.z_samples, r2.z_samples)
