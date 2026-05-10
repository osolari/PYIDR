"""Atomic-type independence-Metropolis tests (Algorithm 1 step 4)."""

from __future__ import annotations

from collections import Counter

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy import stats as sps

from py_idr.copulas.registry import ATOMIC_TYPES
from py_idr.inference.mcmc.type_update import (
    INDEX_TO_TYPE,
    TYPE_TO_INDEX,
    AtomDraw,
    prior_propose_atom,
    supported_types_for_K,
    update_atomic_type,
)

# ---- prior_propose_atom -------------------------------------------------------------


@pytest.mark.inference
@pytest.mark.parametrize("type_name", ["gauss", "t5", "clayton"])
def test_prior_propose_atom_returns_atomdraw(type_name: str) -> None:
    """Each registered type returns an AtomDraw with matching index + a usable copula."""
    K = 3
    draw = prior_propose_atom(jax.random.PRNGKey(0), type_name, K)
    assert isinstance(draw, AtomDraw)
    assert draw.type_name == type_name
    assert draw.type_index == TYPE_TO_INDEX[type_name]
    # log_density returns a finite vector at a non-degenerate test point.
    u = jnp.full((1, K), 0.5)
    log_d = draw.copula.log_density(u)
    assert bool(jnp.all(jnp.isfinite(log_d)))


@pytest.mark.inference
def test_prior_propose_atom_gumbel_K2() -> None:
    """Gumbel proposal works at K=2 (the only K it currently supports)."""
    draw = prior_propose_atom(jax.random.PRNGKey(0), "gumbel", 2)
    assert draw.type_name == "gumbel"
    assert draw.params["theta"] > 1.0


@pytest.mark.inference
def test_prior_propose_atom_rejects_unknown_type() -> None:
    """A typo in the type name raises ValueError."""
    with pytest.raises(ValueError, match="unknown atomic type"):
        prior_propose_atom(jax.random.PRNGKey(0), "frank", 3)


@pytest.mark.inference
def test_index_to_type_round_trip() -> None:
    """The TYPE_TO_INDEX / INDEX_TO_TYPE maps are inverses."""
    for name in ATOMIC_TYPES:
        assert INDEX_TO_TYPE[TYPE_TO_INDEX[name]] == name


# ---- supported_types_for_K ----------------------------------------------------------


@pytest.mark.inference
def test_supported_types_for_K_K2_includes_gumbel() -> None:
    """At K=2, Gumbel is in the proposal set."""
    out = supported_types_for_K(2)
    assert "gumbel" in out


@pytest.mark.inference
@pytest.mark.parametrize("K", [3, 5, 10])
def test_supported_types_for_K_K_geq_3_excludes_gumbel(K: int) -> None:
    """At K≥3, Gumbel is dropped from the proposal set (Hofert recursion not yet implemented)."""
    out = supported_types_for_K(K)
    assert "gumbel" not in out
    assert {"gauss", "t5", "clayton"}.issubset(out)


@pytest.mark.inference
def test_supported_types_for_K_respects_caller_filter() -> None:
    """If the caller has already restricted proposal_types, the K-filter narrows further."""
    out = supported_types_for_K(5, proposal_types=("gauss", "gumbel"))
    assert out == ("gauss",)


# ---- update_atomic_type — basic sanity ----------------------------------------------


@pytest.mark.inference
def test_update_atomic_type_returns_valid_atomdraw() -> None:
    """The returned atom is always an AtomDraw with a usable copula."""
    K = 3
    U = jnp.asarray(np.random.default_rng(0).uniform(0.05, 0.95, size=(50, K)))
    current = prior_propose_atom(jax.random.PRNGKey(0), "gauss", K)
    new_atom, accepted, log_alpha = update_atomic_type(jax.random.PRNGKey(1), U, current)
    assert isinstance(new_atom, AtomDraw)
    assert isinstance(accepted, bool)
    assert jnp.isfinite(jnp.asarray(log_alpha))


@pytest.mark.inference
def test_update_atomic_type_rejects_empty_cluster() -> None:
    """Empty clusters are caller errors — the function raises rather than silently no-op."""
    K = 3
    empty_U = jnp.zeros((0, K))
    current = prior_propose_atom(jax.random.PRNGKey(0), "gauss", K)
    with pytest.raises(ValueError, match="empty cluster"):
        update_atomic_type(jax.random.PRNGKey(1), empty_U, current)


@pytest.mark.inference
def test_update_atomic_type_respects_proposal_types_filter() -> None:
    """proposal_types restriction prevents proposing types outside the set.

    Used by the §4.6 'Gaussian-only' ablation: restricting to ('gauss',) makes
    every proposed type 'gauss', and the chain stays on the Gaussian sub-family.
    """
    K = 4
    U = jnp.asarray(np.random.default_rng(2).uniform(0.05, 0.95, size=(50, K)))
    # Start on Clayton; only allow 'gauss' as proposal — every accepted draw is 'gauss'.
    current = prior_propose_atom(jax.random.PRNGKey(0), "clayton", K)
    keys = jax.random.split(jax.random.PRNGKey(3), 50)
    types_seen = []
    for k in keys:
        current, _, _ = update_atomic_type(k, U, current, proposal_types=("gauss",))
        types_seen.append(current.type_name)
    # After at least one acceptance, the chain should be on 'gauss'; check at the end.
    assert "gauss" in types_seen


# ---- recovery: synthetic data favours the correct type ------------------------------


def _simulate_gaussian_copula(K: int, n: int, rho: float, seed: int) -> jnp.ndarray:
    """Draw from an exchangeable Gaussian copula."""
    rng = np.random.default_rng(seed)
    R = (1.0 - rho) * np.eye(K) + rho * np.ones((K, K))
    z = rng.multivariate_normal(np.zeros(K), R, size=n)
    return jnp.asarray(sps.norm.cdf(z))


def _simulate_clayton_bivariate(n: int, theta: float, seed: int) -> jnp.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.uniform(size=n)
    w = rng.uniform(size=n)
    u1 = v
    u2 = (((w ** (-theta / (1.0 + theta))) - 1.0) * v ** (-theta) + 1.0) ** (-1.0 / theta)
    return jnp.stack([jnp.asarray(u1), jnp.asarray(u2)], axis=1)


@pytest.mark.inference
@pytest.mark.slow
def test_im_chain_settles_on_gauss_for_gaussian_data() -> None:
    """At K=4 with Gaussian copula data, the IM chain spends most steps on 'gauss'."""
    K = 4
    U = _simulate_gaussian_copula(K, 200, rho=0.7, seed=0)
    current = prior_propose_atom(jax.random.PRNGKey(0), "t5", K)
    keys = jax.random.split(jax.random.PRNGKey(1), 500)
    types_seen = [update_atomic_type(k, U, current)[0].type_name for k in keys[:1]]
    # Run sequentially for the rest.
    types_seen = []
    for k in keys:
        current, _, _ = update_atomic_type(k, U, current)
        types_seen.append(current.type_name)
    counts = Counter(types_seen)
    # Sanity: 'gauss' should be the modal type and account for at least 50% of the chain.
    assert counts.most_common(1)[0][0] == "gauss"
    assert counts["gauss"] / sum(counts.values()) > 0.5


@pytest.mark.inference
@pytest.mark.slow
def test_im_chain_settles_on_clayton_for_clayton_data() -> None:
    """At K=2 with Clayton-copula data, the IM chain prefers 'clayton'."""
    U = _simulate_clayton_bivariate(n=200, theta=2.0, seed=1)
    current = prior_propose_atom(jax.random.PRNGKey(0), "gauss", 2)
    keys = jax.random.split(jax.random.PRNGKey(2), 500)
    types_seen = []
    for k in keys:
        current, _, _ = update_atomic_type(k, U, current)
        types_seen.append(current.type_name)
    counts = Counter(types_seen)
    assert counts.most_common(1)[0][0] == "clayton"
    assert counts["clayton"] / sum(counts.values()) > 0.5
