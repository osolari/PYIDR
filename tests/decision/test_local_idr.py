"""Local idr tests: from_mcmc Monte Carlo averaging and the δ_n error scale."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from py_idr.decision.local_idr import from_mcmc, from_vi, local_idr_error_scale


@pytest.mark.decision
def test_from_mcmc_marginalisation() -> None:
    """from_mcmc averages (1 - Z_i) across draws — matches a brute-force enumeration."""
    z_samples = jnp.array(
        [
            [0, 1, 1, 0],  # draw 1: features 0, 3 are irreproducible (Z=0)
            [0, 1, 1, 0],
            [1, 1, 1, 0],  # feature 0 occasionally reproducible
            [0, 1, 1, 0],
        ]
    )
    idr = from_mcmc(z_samples)
    # Feature 0: Z=0 in 3/4 draws -> idr = 0.75
    # Feature 1: Z=1 in 4/4 -> idr = 0
    # Feature 2: Z=1 in 4/4 -> idr = 0
    # Feature 3: Z=0 in 4/4 -> idr = 1
    assert jnp.allclose(idr, jnp.array([0.75, 0.0, 0.0, 1.0]))


@pytest.mark.decision
def test_from_mcmc_rejects_wrong_shape() -> None:
    """from_mcmc requires (draws, n) — 1-D or 3-D inputs raise."""
    with pytest.raises(ValueError, match="must be"):
        from_mcmc(jnp.array([0, 1, 0, 1]))
    with pytest.raises(ValueError, match="must be"):
        from_mcmc(jnp.zeros((2, 3, 4)))


@pytest.mark.decision
def test_from_vi_passes_through_when_valid() -> None:
    """from_vi is the identity on valid q(Z=0) inputs (provided for API symmetry)."""
    q = jnp.array([0.0, 0.3, 0.7, 1.0])
    out = from_vi(q)
    assert jnp.allclose(out, q)


@pytest.mark.decision
def test_from_vi_rejects_out_of_range_values() -> None:
    """from_vi flags values outside [0, 1] (a common bug when feeding raw logits)."""
    with pytest.raises(ValueError, match="\\[0, 1\\]"):
        from_vi(jnp.array([0.5, 1.2, 0.3]))
    with pytest.raises(ValueError, match="\\[0, 1\\]"):
        from_vi(jnp.array([-0.1, 0.5, 0.3]))


@pytest.mark.decision
def test_delta_n_zero_for_identical_chains() -> None:
    """If two chain estimates agree exactly, δ_n is 0."""
    chain = jnp.array([0.1, 0.2, 0.3, 0.4])
    delta = local_idr_error_scale(jnp.stack([chain, chain]))
    assert delta == 0.0


@pytest.mark.decision
def test_delta_n_captures_max_disagreement() -> None:
    """δ_n is the largest per-feature spread across chains."""
    chain1 = jnp.array([0.10, 0.20, 0.30, 0.40])
    chain2 = jnp.array([0.12, 0.18, 0.31, 0.50])  # max spread is at feature 3 (0.10)
    delta = local_idr_error_scale(jnp.stack([chain1, chain2]))
    assert pytest.approx(delta, abs=1e-6) == 0.10


@pytest.mark.decision
def test_delta_n_shrinks_with_chain_length() -> None:
    """On a fixed problem, δ_n decreases as we draw more samples per chain.

    Synthetic test: simulate two independent chains drawing Bernoulli(0.3); the
    Monte Carlo variance shrinks as 1/√D, so δ_n should fall below the short-chain
    value when we lengthen.
    """
    n = 50
    pi_true = 0.3
    rng = jax.random.PRNGKey(0)

    def estimate_delta(chain_len: int, key: jax.Array) -> float:
        keys = jax.random.split(key, 2)
        # Two independent chains.
        chain_zs_a = (jax.random.uniform(keys[0], (chain_len, n)) > pi_true).astype(jnp.int32)
        chain_zs_b = (jax.random.uniform(keys[1], (chain_len, n)) > pi_true).astype(jnp.int32)
        idr_a = from_mcmc(chain_zs_a)
        idr_b = from_mcmc(chain_zs_b)
        return local_idr_error_scale(jnp.stack([idr_a, idr_b]))

    short = estimate_delta(50, jax.random.fold_in(rng, 0))
    long = estimate_delta(5000, jax.random.fold_in(rng, 1))
    assert long < short
