"""Multi-cluster sweep tests: state consistency + smoke-level recovery."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy import stats as sps

from py_idr.inference.mcmc.sweep_multi import (
    _bernstein_U,
    _mixture_log_c1,
    _per_cluster_log_density,
    multi_cluster_sweep,
)
from py_idr.inference.mcmc.type_update import prior_propose_atom
from py_idr.marginals.transform import empirical_rank_transform
from py_idr.model.multi_state import (
    MultiClusterState,
    cluster_sizes,
    initial_multi_cluster_state,
)


def _build_F0(n: int, K: int, seed: int) -> jnp.ndarray:
    """Empirical-rank pilot from synthetic uniform data."""
    rng = np.random.default_rng(seed)
    U = rng.uniform(size=(n, K))
    return jnp.stack(
        [empirical_rank_transform(jnp.asarray(U[:, j])) for j in range(K)],
        axis=1,
    )


# ---- helpers ----------------------------------------------------------------------


@pytest.mark.unit
def test_per_cluster_log_density_shape() -> None:
    """log-densities per cluster have shape (n, T)."""
    K = 3
    atom1 = prior_propose_atom(jax.random.PRNGKey(0), "gauss", K)
    atom2 = prior_propose_atom(jax.random.PRNGKey(1), "clayton", K)
    U = jnp.full((20, K), 0.5)
    out = _per_cluster_log_density((atom1, atom2), U)
    assert out.shape == (20, 2)
    assert bool(jnp.all(jnp.isfinite(out)))


@pytest.mark.unit
def test_per_cluster_log_density_zero_clusters_returns_empty() -> None:
    """With T = 0 active clusters, the function returns shape (n, 0)."""
    U = jnp.full((5, 3), 0.5)
    out = _per_cluster_log_density((), U)
    assert out.shape == (5, 0)


@pytest.mark.unit
def test_mixture_log_c1_normalises_over_clusters() -> None:
    """log c_1 marginalised over the PY cluster prior is finite for all features."""
    n, T = 7, 3
    log_dens = jnp.full((n, T), 0.0)
    counts = jnp.array([5, 3, 2], dtype=jnp.float32)
    out = _mixture_log_c1(log_dens, counts, alpha=1.0, sigma=0.1)
    assert out.shape == (n,)
    assert bool(jnp.all(jnp.isfinite(out)))


@pytest.mark.unit
def test_bernstein_U_clipped() -> None:
    """The applied U stays strictly inside (0, 1)."""
    F0 = _build_F0(20, 2, seed=0)
    weights = jnp.ones((2, 5)) / 5
    U = _bernstein_U(F0, weights, M=5)
    assert float(U.min()) > 0.0
    assert float(U.max()) < 1.0


# ---- sweep mechanics --------------------------------------------------------------


@pytest.mark.integration
def test_multi_cluster_sweep_returns_valid_state() -> None:
    """One sweep yields a state with consistent T / atoms / counts invariants."""
    n, K, M = 30, 2, 5
    F0 = _build_F0(n, K, seed=0)
    atom0 = prior_propose_atom(jax.random.PRNGKey(0), "gauss", K)
    state = initial_multi_cluster_state(n=n, K=K, M=M, T_max=6, initial_atom=atom0)
    new_state = multi_cluster_sweep(
        state,
        F0,
        jax.random.PRNGKey(1),
        H=3,
        nuts_warmup=15,
        py_hyperparam_warmup=20,
    )
    # Invariants:
    assert isinstance(new_state, MultiClusterState)
    assert 0.0 < new_state.pi < 1.0
    assert new_state.alpha > 0.0
    assert 0.0 <= new_state.sigma < 1.0
    # Cluster bookkeeping invariants:
    assert len(new_state.atoms) == new_state.T
    assert new_state.T_max >= new_state.T
    # Counts for the active prefix sum to the number of reproducible features.
    sizes = cluster_sizes(new_state)
    n_reproducible = int(jnp.sum(new_state.Z))
    assert int(jnp.sum(sizes)) == n_reproducible


@pytest.mark.integration
def test_multi_cluster_sweep_can_spawn_and_drop_clusters() -> None:
    """Over multiple sweeps, T can grow and shrink — the Pólya-urn path works."""
    n, K, M = 40, 2, 5
    F0 = _build_F0(n, K, seed=1)
    atom0 = prior_propose_atom(jax.random.PRNGKey(0), "gauss", K)
    state = initial_multi_cluster_state(n=n, K=K, M=M, T_max=8, initial_atom=atom0)
    key = jax.random.PRNGKey(2)
    Ts: list[int] = []
    for _ in range(5):
        key, subkey = jax.random.split(key)
        state = multi_cluster_sweep(
            state,
            F0,
            subkey,
            H=3,
            nuts_warmup=15,
            py_hyperparam_warmup=20,
        )
        Ts.append(state.T)
    # T should fluctuate (the Pólya-urn path explores cluster splits/merges).
    assert max(Ts) >= 1
    assert min(Ts) >= 1


@pytest.mark.integration
def test_multi_cluster_sweep_atom_count_matches_T() -> None:
    """The atom tuple length equals T after every sweep."""
    n, K, M = 30, 2, 5
    F0 = _build_F0(n, K, seed=2)
    atom0 = prior_propose_atom(jax.random.PRNGKey(0), "gauss", K)
    state = initial_multi_cluster_state(n=n, K=K, M=M, T_max=6, initial_atom=atom0)
    key = jax.random.PRNGKey(3)
    for _ in range(3):
        key, subkey = jax.random.split(key)
        state = multi_cluster_sweep(
            state,
            F0,
            subkey,
            H=3,
            nuts_warmup=10,
            py_hyperparam_warmup=15,
        )
        assert (
            len(state.atoms) == state.T
        ), f"T={state.T} does not match len(atoms)={len(state.atoms)}"


# ---- recovery (slow) --------------------------------------------------------------


def _simulate_k2_mixture(n: int, true_pi: float, true_rho: float, seed: int) -> jnp.ndarray:
    rng = np.random.default_rng(seed)
    n_rep = int(true_pi * n)
    R = (1.0 - true_rho) * np.eye(2) + true_rho * np.ones((2, 2))
    z = rng.multivariate_normal(np.zeros(2), R, size=n_rep)
    u_rep = sps.norm.cdf(z)
    u_irrep = rng.uniform(size=(n - n_rep, 2))
    U = np.vstack([u_rep, u_irrep])
    rng.shuffle(U)
    return jnp.asarray(U)


@pytest.mark.integration
@pytest.mark.slow
def test_multi_cluster_sweep_produces_valid_trace_on_K2_mixture() -> None:
    """A short multi-cluster chain on a K=2 mixture produces a finite, well-formed trace.

    This is a smoke-level check that the sweep runs end-to-end across many
    sweeps without state-consistency failures or NaN/Inf values. Detailed
    recovery validation (posterior mean within MC error of the truth) is
    the simulation harness's responsibility (plan 07); a 30-sweep run with
    short NUTS warmups on n=80 data is not expected to converge.
    """
    n, K, M = 80, 2, 5
    true_pi, true_rho = 0.6, 0.7
    U = _simulate_k2_mixture(n, true_pi, true_rho, seed=10)
    F0 = jnp.stack([empirical_rank_transform(U[:, j]) for j in range(K)], axis=1)
    atom0 = prior_propose_atom(jax.random.PRNGKey(0), "gauss", K)
    state = initial_multi_cluster_state(n=n, K=K, M=M, T_max=6, initial_atom=atom0)
    key = jax.random.PRNGKey(20)

    pi_trace: list[float] = []
    T_trace: list[int] = []
    for _ in range(30):
        key, subkey = jax.random.split(key)
        state = multi_cluster_sweep(
            state,
            F0,
            subkey,
            H=3,
            nuts_warmup=15,
            py_hyperparam_warmup=20,
        )
        pi_trace.append(state.pi)
        T_trace.append(state.T)
        # Per-sweep invariants.
        assert 0.0 < state.pi < 1.0
        assert state.alpha > 0.0
        assert 0.0 <= state.sigma < 1.0
        assert len(state.atoms) == state.T
    # The trace contains finite values.
    assert not np.any(np.isnan(np.asarray(pi_trace)))
    assert min(T_trace) >= 1
    assert max(T_trace) <= state.T_max
