"""SamplerState PyTree tests."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from py_idr.model.state import SamplerState, initial_state


@pytest.mark.unit
def test_initial_state_default_construction() -> None:
    """initial_state with valid args produces a SamplerState with the expected shapes."""
    s = initial_state(n=20, K=3, T_max=8, M_max=20, M_init=10)
    assert s.n == 20
    assert s.K == 3
    assert s.T_max == 8
    assert s.M_max == 20
    assert s.Z.shape == (20,)
    assert s.counts.shape == (8,)
    assert s.bernstein_weights.shape == (3, 20)
    # All features start in cluster 0; counts[0] = n.
    assert int(s.counts[0]) == 20
    # T = 1 — one occupied cluster.
    assert int(s.T) == 1


@pytest.mark.unit
def test_initial_state_uniform_bernstein_first_M_init_columns() -> None:
    """initial_state seeds uniform Bernstein weights on the first M_init columns."""
    s = initial_state(n=10, K=2, T_max=4, M_max=20, M_init=5)
    row_sum = float(jnp.sum(s.bernstein_weights[0]))
    assert pytest.approx(row_sum, abs=1e-5) == 1.0
    # Columns past M_init should be zero.
    assert bool(jnp.all(s.bernstein_weights[:, 5:] == 0.0))


@pytest.mark.unit
def test_initial_state_rejects_invalid_pi() -> None:
    """π must be in (0, 1)."""
    with pytest.raises(ValueError, match="pi must"):
        initial_state(n=10, K=2, T_max=4, M_max=20, pi=0.0)
    with pytest.raises(ValueError, match="pi must"):
        initial_state(n=10, K=2, T_max=4, M_max=20, pi=1.5)


@pytest.mark.unit
def test_initial_state_rejects_invalid_alpha_sigma() -> None:
    """alpha > 0 and sigma in [0, 1)."""
    with pytest.raises(ValueError, match="alpha"):
        initial_state(n=10, K=2, T_max=4, M_max=20, alpha=0.0)
    with pytest.raises(ValueError, match="sigma"):
        initial_state(n=10, K=2, T_max=4, M_max=20, sigma=1.0)
    with pytest.raises(ValueError, match="sigma"):
        initial_state(n=10, K=2, T_max=4, M_max=20, sigma=-0.1)


@pytest.mark.unit
def test_initial_state_rejects_invalid_M_init() -> None:
    """M_init must be in DEGREE_SUPPORT and ≤ M_max."""
    with pytest.raises(ValueError, match="degree-prior"):
        initial_state(n=10, K=2, T_max=4, M_max=20, M_init=7)
    with pytest.raises(ValueError, match="degree-prior"):
        initial_state(n=10, K=2, T_max=4, M_max=10, M_init=80)


@pytest.mark.unit
def test_sampler_state_pytree_round_trip() -> None:
    """SamplerState is a registered JAX PyTree — flatten + unflatten round-trips."""
    s = initial_state(n=15, K=3, T_max=6, M_max=20, M_init=10)
    leaves, treedef = jax.tree_util.tree_flatten(s)
    s2 = jax.tree_util.tree_unflatten(treedef, leaves)
    assert s2.n == s.n
    assert s2.K == s.K
    assert bool(jnp.all(s2.Z == s.Z))
    assert bool(jnp.all(s2.counts == s.counts))


@pytest.mark.unit
def test_sampler_state_pytree_jit_compatible() -> None:
    """SamplerState works as a jit input — proves the PyTree registration is correct."""
    s = initial_state(n=10, K=2, T_max=4, M_max=20, M_init=5)

    @jax.jit
    def increment_pi(state: SamplerState, delta: float) -> SamplerState:
        from dataclasses import replace

        return replace(state, pi=state.pi + delta)

    s2 = increment_pi(s, 0.1)
    assert pytest.approx(float(s2.pi), abs=1e-5) == float(s.pi) + 0.1


@pytest.mark.unit
def test_sampler_state_pytree_has_no_static_aux() -> None:
    """All SamplerState fields are leaves; no Python ints stored as state."""
    s = initial_state(n=5, K=2, T_max=4, M_max=20, M_init=5)
    leaves, treedef = jax.tree_util.tree_flatten(s)
    # 10 fields, each a JAX array.
    assert len(leaves) == 10
    for leaf in leaves:
        assert hasattr(leaf, "shape")
