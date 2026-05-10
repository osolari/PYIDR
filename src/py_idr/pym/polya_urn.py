"""Pitman-Yor Pólya-urn auxiliary-atom cluster reassignment (§3.3.1 of the revised report).

Implements the predictive of Eqs. (3.5)–(3.6):

.. math::
   \\Pr(z_i = m \\mid \\cdot)
       \\propto (n_{m,-i} - \\sigma)\\, c_{\\theta_m, t_m}(\\bU_i),
       \\quad m = 1, \\ldots, T_{-i},

.. math::
   \\Pr(z_i = \\widetilde m_h \\mid \\cdot)
       \\propto \\frac{\\alpha + \\sigma\\, T_{-i}}{H}\\,
                  c_{\\widetilde\\theta_h, \\widetilde t_h}(\\bU_i),
       \\quad h = 1, \\ldots, H.

Both branches use the **post-removal** counts $n_{m,-i}$ and $T_{-i}$ — i.e., the
counts after temporarily removing observation $i$ from its current cluster. This
matters most in the singleton-deletion path: when $i$ is the only member of its
current cluster, that cluster is dropped from the occupied list and $T_{-i} = T - 1$.
The full posterior step instantiates a new cluster from the chosen auxiliary atom
when ``new_index >= T_{-i}``.

The functions in this module operate on a minimal :class:`ClusterState` that
tracks just the cluster assignments and per-cluster atom log-density evaluations;
the full :class:`py_idr.model.state.SamplerState` (W2, plan 04) wraps this with
$\\pi$, $\\alpha$, $\\sigma$, marginals, etc.

See plans/04_inference_mcmc.md.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Integer


@dataclass(frozen=True)
class ClusterState:
    """Minimal Pólya-urn state for a single cluster reassignment step.

    Attributes
    ----------
    z
        Length-$n$ array of current cluster assignments. Reproducible features
        ($Z_i = 1$) hold an integer in ``[0, T)``; irreproducible features hold
        any sentinel and are skipped by :func:`step`.
    counts
        Length-``T_max`` array of cluster occupancy counts; entries beyond the
        currently-active prefix are 0. Carried explicitly to avoid recomputation.
    T
        Number of currently-occupied clusters (the active prefix length).
    alpha
        PY concentration parameter, $\\alpha > 0$.
    sigma
        PY discount parameter, $\\sigma \\in [0, 1)$.
    """

    z: Integer[Array, "n"]
    counts: Integer[Array, "T_max"]
    T: int
    alpha: float
    sigma: float


def predictive_log_weights(
    n_minus_i: Integer[Array, "T_minus_i"],
    T_minus_i: int,
    alpha: float,
    sigma: float,
    H: int,
    occupied_log_density: Float[Array, "T_minus_i"],
    aux_log_density: Float[Array, "H"],
) -> Float[Array, "T_minus_i_plus_H"]:
    """Return the un-normalised log-weights of Eqs. (3.5) and (3.6).

    Parameters
    ----------
    n_minus_i
        Post-removal counts $n_{m,-i}$ for the $T_{-i}$ occupied clusters (each
        strictly positive — empty clusters are dropped before this call).
    T_minus_i
        Post-removal occupied-cluster count.
    alpha, sigma
        PY hyperparameters; require $\\alpha > 0$ and $\\sigma \\in [0, 1)$.
    H
        Auxiliary-atom count (the report uses $H = 5$ by default).
    occupied_log_density
        Length-$T_{-i}$ array of $\\log c_{\\theta_m, t_m}(\\bU_i)$ at each occupied cluster.
    aux_log_density
        Length-$H$ array of $\\log c_{\\widetilde\\theta_h, \\widetilde t_h}(\\bU_i)$
        at each auxiliary atom.

    Returns
    -------
    Length-$(T_{-i} + H)$ vector of un-normalised log-weights, with the first
    $T_{-i}$ entries from Eq. (3.5) and the last $H$ entries from Eq. (3.6).
    The caller normalises via ``jax.nn.log_softmax`` if it needs probabilities.

    Notes
    -----
    The auxiliary multiplier is $(\\alpha + \\sigma\\, T_{-i}) / H$ (Eq. 3.6) — the
    crucial PY correction that earlier drafts missed by writing $(\\alpha + \\sigma\\, T)/H$.
    With $T_{-i} = T$ in non-singleton paths the two are identical, but they
    differ on singleton deletion where $T_{-i} = T - 1$.
    """
    # Occupied weights: log(n_{m,-i} - σ) + log c_{θ_m}(U_i)
    occupied_log_w = jnp.log(n_minus_i - sigma) + occupied_log_density
    # Auxiliary weights: log((α + σ T_{-i}) / H) + log c_{aux_h}(U_i)
    aux_constant = jnp.log((alpha + sigma * T_minus_i) / H)
    aux_log_w = aux_constant + aux_log_density
    return jnp.concatenate([occupied_log_w, aux_log_w])


def sample_assignment(
    log_weights: Float[Array, "T_minus_i_plus_H"],
    key: jax.Array,
) -> int:
    """Sample a single index from the categorical given by un-normalised log-weights.

    Returned indices in ``[0, T_{-i})`` denote occupied clusters; indices in
    ``[T_{-i}, T_{-i} + H)`` denote auxiliary atoms (which the caller instantiates
    as a new occupied cluster).
    """
    return int(jax.random.categorical(key, log_weights))


def post_removal_counts(
    counts: Integer[Array, "T"],
    z_i_old: int,
) -> tuple[Integer[Array, "T_minus_i"], Integer[Array, "T_minus_i"], int]:
    """Compute $n_{m,-i}$, the surviving cluster indices, and $T_{-i}$ after removing $i$.

    Returns
    -------
    n_minus_i_compact
        $n_{m,-i}$ restricted to surviving clusters (positive entries only), in
        the broader cluster-index order.
    surviving_indices
        Indices into the original ``counts`` array of the surviving clusters
        (length $T_{-i}$).
    T_minus_i
        Number of surviving clusters.
    """
    n_minus_i = counts.at[z_i_old].add(-1)
    survives = n_minus_i > 0
    T_minus_i = int(jnp.sum(survives))
    surviving_indices = jnp.where(survives, size=counts.shape[0])[0][:T_minus_i]
    n_minus_i_compact = n_minus_i[surviving_indices]
    return n_minus_i_compact, surviving_indices, T_minus_i


def step(
    state: ClusterState,
    i: int,
    occupied_log_density_at_U_i: Float[Array, "T"],
    aux_log_density_at_U_i: Float[Array, "H"],
    H: int,
    key: jax.Array,
) -> tuple[int, int]:
    """Run one Pólya-urn auxiliary-atom reassignment for feature $i$.

    Parameters
    ----------
    state
        Current cluster state. Caller must guarantee that feature $i$ is currently
        assigned to a reproducible cluster.
    i
        Feature index being reassigned.
    occupied_log_density_at_U_i
        Length-$T$ array of $\\log c_{\\theta_m, t_m}(\\bU_i)$ at every currently
        occupied cluster.
    aux_log_density_at_U_i
        Length-$H$ array of log-densities at $H$ auxiliary atoms drawn from $P_0$.
    H
        Auxiliary-atom count.
    key
        JAX PRNG key for the categorical sample.

    Returns
    -------
    A pair ``(chosen_index, T_minus_i)`` — the index sampled from the
    $(T_{-i} + H)$-way categorical, and the post-removal $T_{-i}$. The caller
    interprets ``chosen_index < T_minus_i`` as an occupied-cluster reassignment
    (mapping back to the broader cluster numbering via the surviving indices)
    and ``chosen_index >= T_minus_i`` as an auxiliary-atom instantiation.

    Notes
    -----
    This function deliberately avoids materialising new state: the cluster
    instantiation, count update, and atom-array growth are bookkeeping that
    belongs in the broader :class:`SamplerState` (plan 04). Returning the bare
    chosen index keeps the unit tests focused on the predictive math.
    """
    z_i_old = int(state.z[i])
    occupied_log_dens_compact, _, T_minus_i = _project_to_survivors(
        state.counts, z_i_old, state.T, occupied_log_density_at_U_i
    )
    n_minus_i_compact = _n_minus_i_compact(state.counts, z_i_old, state.T)

    log_w = predictive_log_weights(
        n_minus_i=n_minus_i_compact,
        T_minus_i=T_minus_i,
        alpha=state.alpha,
        sigma=state.sigma,
        H=H,
        occupied_log_density=occupied_log_dens_compact,
        aux_log_density=aux_log_density_at_U_i,
    )

    chosen = sample_assignment(log_w, key)
    return chosen, T_minus_i


def _n_minus_i_compact(
    counts: Integer[Array, "T_max"],
    z_i_old: int,
    T: int,
) -> Integer[Array, "T_minus_i"]:
    """Compute the compact post-removal counts on surviving clusters."""
    active = counts[:T]
    n_minus_i = active.at[z_i_old].add(-1)
    survives = n_minus_i > 0
    T_minus_i = int(jnp.sum(survives))
    surviving_indices = jnp.where(survives, size=T)[0][:T_minus_i]
    return n_minus_i[surviving_indices]


def _project_to_survivors(
    counts: Integer[Array, "T_max"],
    z_i_old: int,
    T: int,
    occupied_log_density: Float[Array, "T"],
) -> tuple[Float[Array, "T_minus_i"], Integer[Array, "T_minus_i"], int]:
    """Project an occupied-cluster log-density vector onto the surviving clusters."""
    active = counts[:T]
    n_minus_i = active.at[z_i_old].add(-1)
    survives = n_minus_i > 0
    T_minus_i = int(jnp.sum(survives))
    surviving_indices = jnp.where(survives, size=T)[0][:T_minus_i]
    return occupied_log_density[surviving_indices], surviving_indices, T_minus_i
