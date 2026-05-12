"""Tests for compute_em_init + the run_chain_*(em_init=True) wiring."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from py_idr.inference.mcmc.em_init import EMInit, compute_em_init
from py_idr.inference.mcmc.run_chain import run_chain_multi, run_chain_simple
from py_idr.marginals.transform import empirical_rank_transform
from py_idr.simulation.scenarios import simulate_S1


def _build_F0(K: int, n: int, seed: int) -> np.ndarray:
    sim = simulate_S1(K=K, n=n, pi_star=0.30, rho=0.85, seed=seed)
    F0 = jnp.stack([empirical_rank_transform(sim.X[:, j]) for j in range(K)], axis=1)
    return np.asarray(F0)


# ---- compute_em_init: validation ----------------------------------------------------


@pytest.mark.unit
def test_compute_em_init_rejects_1d_F0() -> None:
    """F0 must be 2-D (n, K)."""
    with pytest.raises(ValueError, match="2-D"):
        compute_em_init(np.zeros(20))


@pytest.mark.unit
def test_compute_em_init_rejects_K_lt_2() -> None:
    """K must be at least 2."""
    with pytest.raises(ValueError, match="K must be >= 2"):
        compute_em_init(np.zeros((20, 1)))


@pytest.mark.unit
def test_compute_em_init_rejects_bad_pi_clamps() -> None:
    """pi_floor / pi_ceiling must satisfy 0 < floor < ceiling < 1."""
    F0 = _build_F0(K=2, n=20, seed=0)
    with pytest.raises(ValueError, match="pi_floor"):
        compute_em_init(F0, pi_floor=0.5, pi_ceiling=0.5)
    with pytest.raises(ValueError, match="pi_floor"):
        compute_em_init(F0, pi_floor=-0.1)


@pytest.mark.unit
def test_compute_em_init_rejects_bad_rho_clamp() -> None:
    """rho_clamp must be in (0, 1)."""
    F0 = _build_F0(K=2, n=20, seed=0)
    with pytest.raises(ValueError, match="rho_clamp"):
        compute_em_init(F0, rho_clamp=1.5)


# ---- compute_em_init: correctness ---------------------------------------------------


@pytest.mark.unit
def test_compute_em_init_returns_dataclass_with_expected_shapes() -> None:
    """EMInit dataclass has scalar pi/rho and (n,) Z + idr arrays."""
    F0 = _build_F0(K=3, n=200, seed=0)
    init = compute_em_init(F0)
    assert isinstance(init, EMInit)
    assert 0.0 < init.pi < 1.0
    assert -1.0 < init.rho < 1.0
    assert init.Z.shape == (200,)
    assert init.idr_per_feature.shape == (200,)
    assert set(np.unique(init.Z)).issubset({0, 1})


@pytest.mark.unit
def test_compute_em_init_recovers_S1_truth_at_K3() -> None:
    """At K=3, n=500, the EM init lands close to pi*=0.30 and rho*=0.85."""
    F0 = _build_F0(K=3, n=500, seed=0)
    init = compute_em_init(F0)
    assert abs(init.pi - 0.30) < 0.10
    assert abs(init.rho - 0.85) < 0.10


@pytest.mark.unit
def test_compute_em_init_pi_clamps_apply() -> None:
    """pi is clamped into [pi_floor, pi_ceiling]."""
    F0 = _build_F0(K=2, n=100, seed=0)
    init = compute_em_init(F0, pi_floor=0.40, pi_ceiling=0.45)
    assert 0.40 <= init.pi <= 0.45


@pytest.mark.unit
def test_compute_em_init_deterministic() -> None:
    """Same F0 → same EM init (the EM is data-deterministic)."""
    F0 = _build_F0(K=3, n=150, seed=0)
    a = compute_em_init(F0)
    b = compute_em_init(F0)
    assert a.pi == b.pi
    assert a.rho == b.rho
    assert np.array_equal(a.Z, b.Z)


# ---- run_chain_simple(em_init=True) -------------------------------------------------


@pytest.mark.integration
def test_run_chain_simple_em_init_runs() -> None:
    """The simple chain accepts em_init=True and produces a valid result.

    Doesn't check recovery — short warmup makes that flaky — only that
    the new code path executes end-to-end and produces the standard
    output shape.
    """
    F0 = _build_F0(K=3, n=60, seed=0)
    result = run_chain_simple(
        F0=jnp.asarray(F0),
        M=5,
        key=jax.random.PRNGKey(0),
        n_warmup=2,
        n_samples=5,
        nuts_warmup=5,
        em_init=True,
    )
    assert result.z_samples is not None
    assert result.z_samples.shape == (5, 60)


@pytest.mark.integration
def test_run_chain_simple_em_init_starts_near_em_estimate() -> None:
    """First-draw pi sits near the EM-recovered pi, not at the default 0.5."""
    F0_np = _build_F0(K=3, n=200, seed=0)
    em = compute_em_init(F0_np)
    result = run_chain_simple(
        F0=jnp.asarray(F0_np),
        M=5,
        key=jax.random.PRNGKey(0),
        n_warmup=0,
        n_samples=3,
        nuts_warmup=5,
        em_init=True,
    )
    # First post-warmup draw of pi: not at default 0.5.
    pi0 = float(np.asarray(result.idata.posterior["pi"])[0, 0])
    # Should be much closer to em.pi than to 0.5.
    assert abs(pi0 - em.pi) < abs(pi0 - 0.5) + 0.05


# ---- run_chain_multi(em_init=True) --------------------------------------------------


@pytest.mark.integration
def test_run_chain_multi_em_init_runs() -> None:
    """The multi-cluster chain accepts em_init=True and produces a valid result."""
    F0 = _build_F0(K=2, n=40, seed=0)
    result = run_chain_multi(
        F0=jnp.asarray(F0),
        M=5,
        key=jax.random.PRNGKey(0),
        T_max=3,
        H=3,
        n_warmup=2,
        n_samples=5,
        nuts_warmup=5,
        py_hyperparam_warmup=5,
        em_init=True,
    )
    assert result.z_samples is not None
    assert result.z_samples.shape == (5, 40)
