"""Tests for the multi-cluster chain driver (W6.1).

The driver wraps :func:`py_idr.inference.mcmc.sweep_multi.multi_cluster_sweep`
into a top-level entry point matching ``run_chain_simple``'s shape but
adding the PY-hyperparameter and cluster-count traces. We test the
structural contract (shapes, dtypes, value ranges) on tiny configs so
the suite runs in seconds rather than minutes.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from py_idr.decision.local_idr import from_mcmc
from py_idr.inference.mcmc.run_chain import (
    ChainResultMulti,
    MultiChainResultMulti,
    run_chain_multi,
    run_multi_chain_multi,
)
from py_idr.marginals.transform import empirical_rank_transform
from py_idr.simulation.scenarios import simulate_S1


def _build_F0(K: int, n: int, seed: int) -> jnp.ndarray:
    """Tiny pilot CDF from an S1 simulation."""
    sim = simulate_S1(K=K, n=n, seed=seed)
    return jnp.stack(
        [empirical_rank_transform(sim.X[:, j]) for j in range(K)],
        axis=1,
    )


# ---- input validation ----------------------------------------------------------------


@pytest.mark.unit
def test_run_chain_multi_rejects_bad_F0_shape() -> None:
    """F0 must be 2-D (n, K)."""
    with pytest.raises(ValueError, match="F0 must be 2-D"):
        run_chain_multi(
            F0=jnp.zeros(10),
            M=5,
            key=jax.random.PRNGKey(0),
            T_max=2,
            n_warmup=1,
            n_samples=1,
        )


@pytest.mark.unit
def test_run_chain_multi_rejects_bad_T_max() -> None:
    """T_max must be at least 1."""
    F0 = _build_F0(K=2, n=20, seed=0)
    with pytest.raises(ValueError, match="T_max"):
        run_chain_multi(F0=F0, M=5, key=jax.random.PRNGKey(0), T_max=0, n_warmup=1, n_samples=1)


@pytest.mark.unit
def test_run_chain_multi_rejects_bad_H() -> None:
    """Aux-atom count H must be >= 1."""
    F0 = _build_F0(K=2, n=20, seed=0)
    with pytest.raises(ValueError, match="H "):
        run_chain_multi(
            F0=F0, M=5, key=jax.random.PRNGKey(0), T_max=3, H=0, n_warmup=1, n_samples=1
        )


# ---- single-chain shape / dtype contract --------------------------------------------


@pytest.mark.integration
def test_run_chain_multi_returns_full_shape_contract() -> None:
    """The returned object has shapes the downstream pipeline expects."""
    n, K = 30, 2
    F0 = _build_F0(K=K, n=n, seed=0)
    result = run_chain_multi(
        F0=F0,
        M=5,
        key=jax.random.PRNGKey(0),
        T_max=4,
        H=3,
        n_warmup=2,
        n_samples=5,
        nuts_warmup=5,
        py_hyperparam_warmup=5,
    )
    assert isinstance(result, ChainResultMulti)
    # arviz InferenceData with the multi-cluster scalar trace.
    expected_vars = {"pi", "alpha", "sigma", "T", "n_reproducible"}
    assert expected_vars.issubset(result.idata.posterior.data_vars)
    assert result.idata.posterior["pi"].shape == (1, 5)
    # Z trace + cluster trace shapes match the n_samples × n contract.
    assert result.z_samples.shape == (5, n)
    assert result.cluster_assignments.shape == (5, n)
    assert result.T_trace.shape == (5,)
    # dtype contract.
    assert result.z_samples.dtype == np.int8
    assert result.cluster_assignments.dtype == np.int32


@pytest.mark.integration
def test_run_chain_multi_z_samples_are_binary() -> None:
    """The class-indicator trace contains only {0, 1}."""
    n, K = 20, 2
    F0 = _build_F0(K=K, n=n, seed=0)
    result = run_chain_multi(
        F0=F0,
        M=5,
        key=jax.random.PRNGKey(0),
        T_max=3,
        H=3,
        n_warmup=2,
        n_samples=5,
        nuts_warmup=5,
        py_hyperparam_warmup=5,
    )
    assert set(np.unique(result.z_samples)).issubset({0, 1})


@pytest.mark.integration
def test_run_chain_multi_cluster_labels_invariants() -> None:
    """Cluster labels: -1 for irreproducible, in [0, T-1] for reproducible."""
    n, K = 20, 2
    F0 = _build_F0(K=K, n=n, seed=0)
    result = run_chain_multi(
        F0=F0,
        M=5,
        key=jax.random.PRNGKey(0),
        T_max=5,
        H=3,
        n_warmup=2,
        n_samples=5,
        nuts_warmup=5,
        py_hyperparam_warmup=5,
    )
    for d in range(result.cluster_assignments.shape[0]):
        Z = result.z_samples[d]
        z = result.cluster_assignments[d]
        T = int(result.T_trace[d])
        # Irreproducible features → -1.
        assert np.all(z[Z == 0] == -1)
        # Reproducible features → 0..T-1.
        if (Z == 1).any():
            assert z[Z == 1].min() >= 0
            assert z[Z == 1].max() <= T - 1


@pytest.mark.integration
def test_run_chain_multi_z_trace_feeds_from_mcmc() -> None:
    """Z trace shape is exactly what decision.local_idr.from_mcmc expects."""
    n, K = 20, 2
    F0 = _build_F0(K=K, n=n, seed=0)
    result = run_chain_multi(
        F0=F0,
        M=5,
        key=jax.random.PRNGKey(0),
        T_max=3,
        H=3,
        n_warmup=2,
        n_samples=8,
        nuts_warmup=5,
        py_hyperparam_warmup=5,
    )
    idr = from_mcmc(jnp.asarray(result.z_samples))
    assert idr.shape == (n,)
    assert bool(jnp.all((idr >= 0.0) & (idr <= 1.0)))


@pytest.mark.integration
def test_run_chain_multi_T_trace_matches_idata() -> None:
    """T_trace field and idata.posterior['T'] carry the same numbers."""
    F0 = _build_F0(K=2, n=20, seed=0)
    result = run_chain_multi(
        F0=F0,
        M=5,
        key=jax.random.PRNGKey(0),
        T_max=4,
        H=3,
        n_warmup=1,
        n_samples=5,
        nuts_warmup=5,
        py_hyperparam_warmup=5,
    )
    posterior_T = np.asarray(result.idata.posterior["T"]).reshape(-1)
    assert np.array_equal(posterior_T, result.T_trace)


@pytest.mark.integration
def test_run_chain_multi_T_respects_T_max() -> None:
    """Cluster count never exceeds T_max."""
    F0 = _build_F0(K=2, n=20, seed=0)
    T_max = 3
    result = run_chain_multi(
        F0=F0,
        M=5,
        key=jax.random.PRNGKey(0),
        T_max=T_max,
        H=5,
        n_warmup=2,
        n_samples=10,
        nuts_warmup=5,
        py_hyperparam_warmup=5,
    )
    assert int(result.T_trace.max()) <= T_max


@pytest.mark.integration
def test_run_chain_multi_pi_alpha_sigma_in_range() -> None:
    """Posterior traces of (pi, alpha, sigma) respect their support."""
    F0 = _build_F0(K=2, n=20, seed=0)
    result = run_chain_multi(
        F0=F0,
        M=5,
        key=jax.random.PRNGKey(0),
        T_max=4,
        H=3,
        n_warmup=2,
        n_samples=5,
        nuts_warmup=5,
        py_hyperparam_warmup=5,
    )
    pi = np.asarray(result.idata.posterior["pi"])
    alpha = np.asarray(result.idata.posterior["alpha"])
    sigma = np.asarray(result.idata.posterior["sigma"])
    assert (pi > 0).all() and (pi < 1).all()
    assert (alpha > 0).all()
    assert (sigma >= 0).all() and (sigma < 1).all()


# ---- multi-chain wrapper --------------------------------------------------------------


@pytest.mark.integration
def test_run_multi_chain_multi_stacks_chains() -> None:
    """num_chains independent runs stack into a single InferenceData."""
    F0 = _build_F0(K=2, n=20, seed=0)
    num_chains = 3
    result = run_multi_chain_multi(
        F0=F0,
        M=5,
        key=jax.random.PRNGKey(0),
        num_chains=num_chains,
        T_max=3,
        H=3,
        n_warmup=2,
        n_samples=5,
        nuts_warmup=5,
        py_hyperparam_warmup=5,
    )
    assert isinstance(result, MultiChainResultMulti)
    assert result.idata.posterior["pi"].shape == (num_chains, 5)
    assert result.z_samples.shape == (num_chains, 5, 20)
    assert result.cluster_assignments.shape == (num_chains, 5, 20)
    assert result.T_trace.shape == (num_chains, 5)
    assert len(result.final_states) == num_chains


@pytest.mark.integration
def test_run_multi_chain_multi_overdispersed_inits_differ() -> None:
    """Chains start from different (pi, alpha, sigma) — first-draw posteriors must differ."""
    F0 = _build_F0(K=2, n=20, seed=0)
    result = run_multi_chain_multi(
        F0=F0,
        M=5,
        key=jax.random.PRNGKey(0),
        num_chains=4,
        T_max=3,
        H=3,
        n_warmup=0,
        n_samples=3,
        nuts_warmup=5,
        py_hyperparam_warmup=5,
        overdispersed=True,
    )
    # With n_warmup=0 the first draw reflects the init scatter.
    first_alphas = np.asarray(result.idata.posterior["alpha"])[:, 0]
    assert len(set(first_alphas.tolist())) >= 2  # at least two distinct values
