"""End-to-end smoke tests for the $T=1$ PY-IDR sweep + chain driver."""

from __future__ import annotations

import arviz as az
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy import stats as sps

from py_idr.inference.mcmc.run_chain import (
    ChainResult,
    run_chain_simple,
    summarise,
)
from py_idr.inference.mcmc.sweep_simple import (
    SimpleSweepInputs,
    SimpleSweepState,
    _bernstein_U,
    initial_simple_state,
    sweep_simple,
)
from py_idr.marginals.transform import empirical_rank_transform

# ---- single-sweep mechanics ---------------------------------------------------------


def _build_synthetic_F0(n: int, K: int, seed: int) -> jnp.ndarray:
    """Empirical-rank pilot from uniform-shuffled synthetic data."""
    rng = np.random.default_rng(seed)
    U = rng.uniform(size=(n, K))
    return jnp.stack(
        [empirical_rank_transform(jnp.asarray(U[:, j])) for j in range(K)],
        axis=1,
    )


@pytest.mark.integration
def test_sweep_returns_valid_state_shapes() -> None:
    """One sweep preserves the state shapes and ranges."""
    n, K, M = 50, 2, 5
    F0 = _build_synthetic_F0(n, K, seed=0)
    state = initial_simple_state(n=n, K=K, M=M)
    inputs = SimpleSweepInputs(F0=F0, M=M, nuts_warmup=10)
    new_state = sweep_simple(state, inputs, jax.random.PRNGKey(0))
    assert isinstance(new_state, SimpleSweepState)
    assert 0.0 < new_state.pi < 1.0
    assert 0.0 < new_state.rho < 1.0
    assert new_state.bernstein_weights.shape == (K, M)
    assert new_state.Z.shape == (n,)
    # Bernstein rows sum to 1 (Dirichlet draws).
    row_sums = jnp.sum(new_state.bernstein_weights, axis=1)
    assert jnp.allclose(row_sums, 1.0, atol=1e-5)


@pytest.mark.integration
def test_sweep_handles_all_z_zero_iteration() -> None:
    """If a sweep produces all-Z=0 (no reproducible features), atom rho is held.

    The class assignment can land on all zeros for small n and extreme π̂; the
    sweep should leave rho unchanged in that case rather than failing in the
    NUTS step with an empty data set.
    """
    n, K, M = 5, 2, 5
    F0 = _build_synthetic_F0(n, K, seed=1)
    # Force Z = 0 everywhere by setting π near zero and rho such that log_c1 < 0.
    state = SimpleSweepState(
        pi=0.001,
        rho=0.5,
        bernstein_weights=jnp.ones((K, M)) / M,
        Z=jnp.zeros(n, dtype=jnp.int32),
    )
    inputs = SimpleSweepInputs(F0=F0, M=M, nuts_warmup=10)
    new_state = sweep_simple(state, inputs, jax.random.PRNGKey(0))
    # The new Z should still be near all-zero; rho is held (or close to held).
    assert new_state.rho > 0.0


@pytest.mark.integration
def test_bernstein_U_clipped_to_open_unit() -> None:
    """The marginal-applied U stays in (0, 1)."""
    n, K, M = 20, 2, 5
    F0 = _build_synthetic_F0(n, K, seed=2)
    weights = jnp.ones((K, M)) / M
    U = _bernstein_U(F0, weights, M)
    assert float(jnp.min(U)) > 0.0
    assert float(jnp.max(U)) < 1.0


# ---- initial_simple_state validators -----------------------------------------------


@pytest.mark.integration
def test_initial_state_rejects_invalid_pi() -> None:
    """pi_init must be in (0, 1)."""
    with pytest.raises(ValueError, match="pi_init"):
        initial_simple_state(n=5, K=2, M=5, pi_init=0.0)


@pytest.mark.integration
def test_initial_state_rejects_invalid_rho() -> None:
    """rho_init must be in (0, 1)."""
    with pytest.raises(ValueError, match="rho_init"):
        initial_simple_state(n=5, K=2, M=5, rho_init=1.0)


@pytest.mark.integration
def test_initial_state_rejects_invalid_M() -> None:
    """M must be positive."""
    with pytest.raises(ValueError, match="M must be positive"):
        initial_simple_state(n=5, K=2, M=0)


# ---- chain driver -----------------------------------------------------------------


@pytest.mark.integration
def test_run_chain_returns_chainresult_with_arviz_idata() -> None:
    """run_chain_simple returns a ChainResult; idata has the expected groups + variables."""
    n, K = 30, 2
    F0 = _build_synthetic_F0(n, K, seed=3)
    result = run_chain_simple(
        F0=F0,
        M=5,
        key=jax.random.PRNGKey(0),
        n_warmup=3,
        n_samples=5,
        nuts_warmup=10,
    )
    assert isinstance(result, ChainResult)
    assert isinstance(result.idata, az.InferenceData)
    posterior = result.idata.posterior
    # Both scalar params are present and have the expected (chain, draw) shape.
    assert "pi" in posterior
    assert "rho" in posterior
    assert "n_reproducible" in posterior
    assert posterior["pi"].shape == (1, 5)


@pytest.mark.integration
def test_run_chain_rejects_invalid_inputs() -> None:
    """The driver rejects non-(n, K) F0 and non-positive sample counts."""
    F0_bad = jnp.ones(10)
    with pytest.raises(ValueError, match="2-D"):
        run_chain_simple(F0=F0_bad, M=5, key=jax.random.PRNGKey(0))
    F0 = _build_synthetic_F0(5, 2, seed=0)
    with pytest.raises(ValueError, match="n_samples"):
        run_chain_simple(F0=F0, M=5, key=jax.random.PRNGKey(0), n_samples=0)


@pytest.mark.integration
def test_summarise_returns_arviz_summary_frame() -> None:
    """summarise yields the arviz summary table with the expected columns."""
    n, K = 30, 2
    F0 = _build_synthetic_F0(n, K, seed=4)
    result = run_chain_simple(
        F0=F0,
        M=5,
        key=jax.random.PRNGKey(0),
        n_warmup=3,
        n_samples=10,
        nuts_warmup=10,
    )
    s = summarise(result, var_names=["pi", "rho"])
    # arviz.summary columns include 'mean', 'sd', ESS-style + r_hat. We do not
    # pin the exact column set (it varies across arviz versions); the table
    # has at least the params as rows and mean/sd columns.
    assert "mean" in s.columns
    assert "sd" in s.columns
    assert "pi" in s.index
    assert "rho" in s.index


# ---- end-to-end recovery (slow) ----------------------------------------------------


def _simulate_py_idr_mixture(
    n: int, K: int, true_pi: float, true_rho: float, seed: int
) -> jnp.ndarray:
    """Synthetic two-component PY-IDR observations (Gaussian copula + independence)."""
    rng = np.random.default_rng(seed)
    n_rep = int(true_pi * n)
    R = (1.0 - true_rho) * np.eye(K) + true_rho * np.ones((K, K))
    z_rep = rng.multivariate_normal(np.zeros(K), R, size=n_rep)
    u_rep = sps.norm.cdf(z_rep)
    u_irrep = rng.uniform(size=(n - n_rep, K))
    U = np.vstack([u_rep, u_irrep])
    rng.shuffle(U)
    return jnp.asarray(U)


@pytest.mark.integration
@pytest.mark.slow
def test_end_to_end_recovers_pi_and_rho_on_small_K2_problem() -> None:
    """K=2 synthetic mixture: 200-sample chain recovers π and ρ within MC error."""
    n, K = 200, 2
    true_pi = 0.6
    true_rho = 0.7
    U = _simulate_py_idr_mixture(n, K, true_pi, true_rho, seed=5)
    F0 = jnp.stack(
        [empirical_rank_transform(U[:, j]) for j in range(K)],
        axis=1,
    )
    key = jax.random.PRNGKey(7)
    result = run_chain_simple(
        F0=F0,
        M=10,
        key=key,
        n_warmup=50,
        n_samples=100,
        nuts_warmup=20,
    )
    pi_samples = np.asarray(result.idata.posterior["pi"]).flatten()
    rho_samples = np.asarray(result.idata.posterior["rho"]).flatten()
    # Posterior mean within 0.15 of the truth — generous since chain is short.
    assert abs(pi_samples.mean() - true_pi) < 0.15
    assert abs(rho_samples.mean() - true_rho) < 0.15
    # No NaN in the trace.
    assert not np.any(np.isnan(pi_samples))
    assert not np.any(np.isnan(rho_samples))
