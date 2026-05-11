"""Full Algorithm 1 sweep with Pólya-urn cluster reassignment (multi-cluster).

This is the production multi-cluster orchestrator that wires every step of
Algorithm 1 from the revised report:

Step 1  — class assignment $Z_i \\sim \\mathrm{Bernoulli}(\\widehat\\pi_i)$
Step 2  — Pólya-urn auxiliary-atom cluster reassignment (Eqs. 3.5–3.6)
Step 3  — per-cluster NUTS atom-parameter update
Step 4  — per-cluster atomic-type IM proposal
Step 5  — Bernstein-Dirichlet conjugate update per replicate
Step 7  — (α, σ) joint NUTS update on the partition's EPPF
Step 8  — π update

The marginal-degree update (Step 6) is held fixed at run start for this commit;
it lands in a follow-up.

The orchestrator is intentionally Python-loop-heavy because the heterogeneous
atom state (variable atomic types, variable parameter shapes per cluster) does
not fit cleanly into a single jax-traced PyTree. The per-step numerical kernels
are jit-compatible; the outer loop is bookkeeping. For 100-feature problems
the cluster-reassignment loop dominates and is sequential by construction
(the partition state changes after each feature).

See plans/04_inference_mcmc.md.
"""

from __future__ import annotations

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float

from py_idr.inference.mcmc.atom_dispatch import update_atom_via_nuts
from py_idr.inference.mcmc.class_assign import class_assign
from py_idr.inference.mcmc.pi_update import update_pi
from py_idr.inference.mcmc.py_hyperparam_update import update_py_hyperparams
from py_idr.inference.mcmc.type_update import (
    AtomDraw,
    prior_propose_atom,
    supported_types_for_K,
    update_atomic_type,
)
from py_idr.marginals.bernstein import cdf as bernstein_cdf
from py_idr.marginals.dirichlet_weights import sample_latent_S, update_weights
from py_idr.model.multi_state import (
    MultiClusterState,
    cluster_sizes,
)


def _bernstein_U(
    F0: Float[Array, "n K"],
    weights: Float[Array, "K M"],
    M: int,
) -> Float[Array, "n K"]:
    """Apply each replicate's current Bernstein-Dirichlet CDF to its pilot values."""
    K = F0.shape[1]
    Us = [bernstein_cdf(F0[:, j], weights[j], M) for j in range(K)]
    return jnp.clip(jnp.stack(Us, axis=1), 1.0e-6, 1.0 - 1.0e-6)


def _counts_from_z(z, Z, T, T_max):  # type: ignore[no-untyped-def]  # noqa: ANN001
    """Recompute cluster occupancy counts from the (Z, z) arrays.

    Used after Step 1 (class assignment) before Step 2 (Pólya-urn): features
    transitioning between classes leave the stale ``counts`` array
    inconsistent with the new ``(Z, z)`` partition. The Pólya-urn step
    assumes counts match exactly.
    """
    counts = jnp.zeros(T_max, dtype=jnp.int32)
    # Only Z=1 features contribute; their cluster index is z[i].
    contributions = jnp.where(Z == 1, z, -1)
    for m in range(T):
        counts = counts.at[m].set(int(jnp.sum(contributions == m)))
    return counts


def _per_cluster_log_density(
    atoms: tuple[AtomDraw, ...],
    U: Float[Array, "n K"],
) -> Float[Array, "n T"]:
    """Evaluate $\\log c_m(\\bU_i)$ at every cluster for every feature."""
    if not atoms:
        return jnp.zeros((U.shape[0], 0))
    cols = [atom.copula.log_density(U) for atom in atoms]
    return jnp.stack(cols, axis=1)


def _mixture_log_c1(
    log_dens_per_cluster: Float[Array, "n T"],
    counts: Float[Array, "T"],
    alpha: float,
    sigma: float,
) -> Float[Array, "n"]:
    r"""Marginalise the per-cluster log-densities over the PY cluster prior.

    For class assignment, we use the predictive Pr(cluster_m | partition) ∝
    (n_m - σ) and ignore the small contribution of the auxiliary-atom mass
    (it averages to the same density mixture in expectation across sweeps and
    its effect on the local Z assignment is second-order).

    Returns log c_1(U_i) per feature.
    """
    T = log_dens_per_cluster.shape[1]
    if T == 0:
        return jnp.full((log_dens_per_cluster.shape[0],), -jnp.inf)
    weights = (counts - sigma).astype(jnp.float32)
    weights = weights / jnp.sum(weights)
    log_weights = jnp.log(weights + 1e-300)
    # log-sum-exp over clusters.
    weighted = log_dens_per_cluster + log_weights[None, :]
    return jax.scipy.special.logsumexp(weighted, axis=1)


def _polya_urn_reassign_all(
    state: MultiClusterState,
    U: Float[Array, "n K"],
    H: int,
    key: jax.Array,
) -> MultiClusterState:
    r"""Step 2: per-feature Pólya-urn cluster reassignment using Eqs. (3.5)–(3.6).

    Loops over reproducible features (``Z_i == 1``) sequentially because the
    partition state mutates between steps. Auxiliary atoms are drawn fresh
    per feature. Empty clusters are dropped as soon as they vacate.
    """
    K = U.shape[1]
    reproducible_idx = np.asarray(jnp.where(state.Z == 1)[0])
    # Active proposal type set excludes Gumbel at K ≥ 3 by construction.
    types_supported = supported_types_for_K(K)

    for i in reproducible_idx:
        i = int(i)
        key, subkey, aux_key = jax.random.split(key, 3)

        # Compute occupied + aux log-densities at this U_i.
        U_i = U[i : i + 1]  # shape (1, K)
        T = state.T
        occupied_log_dens = jnp.array(
            [float(atom.copula.log_density(U_i)[0]) for atom in state.atoms]
        )
        # Pre-removal counts.
        counts_active = state.counts[:T]

        # Compute post-removal counts and surviving cluster indices.
        z_i_old = int(state.z[i])
        n_minus_i = counts_active.at[z_i_old].add(-1)
        survives_mask = np.asarray(n_minus_i > 0)
        surviving = np.where(survives_mask)[0]
        T_minus_i = int(len(surviving))
        if T_minus_i == 0:
            # Pathological: only the removed feature occupied a singleton. Skip
            # this feature's reassignment; the broader sweep handles it next round.
            continue

        n_minus_i_compact = jnp.asarray(np.asarray(n_minus_i)[surviving])
        occupied_compact = occupied_log_dens[surviving]

        # Draw H aux atoms from P_0 (uniform over admissible types).
        aux_atoms: list[AtomDraw] = []
        type_keys = jax.random.split(aux_key, H + 1)
        for h in range(H):
            # Uniformly choose a type from the supported set.
            type_pick = type_keys[h]
            type_idx = int(jax.random.randint(type_pick, (), 0, len(types_supported)))
            type_name = types_supported[type_idx]
            aux_atoms.append(prior_propose_atom(type_keys[h + 1], type_name, K))
        aux_log_dens = jnp.array([float(atom.copula.log_density(U_i)[0]) for atom in aux_atoms])

        # Build the (T_-i + H) log-weight vector per Eqs. (3.5)–(3.6).
        sigma = state.sigma
        alpha = state.alpha
        log_w_occupied = jnp.log(n_minus_i_compact - sigma) + occupied_compact
        log_w_aux = jnp.log((alpha + sigma * T_minus_i) / H) + aux_log_dens
        log_w = jnp.concatenate([log_w_occupied, log_w_aux])

        chosen = int(jax.random.categorical(subkey, log_w))

        # Apply the chosen assignment.
        if chosen < T_minus_i:
            # Re-attach to a surviving cluster (in the broader cluster numbering).
            new_z_i = int(surviving[chosen])
            new_counts = state.counts.at[z_i_old].add(-1).at[new_z_i].add(1)
            new_z = state.z.at[i].set(new_z_i)
            state = replace(state, z=new_z, counts=new_counts)
        else:
            # Instantiate a new cluster from the auxiliary atom.
            h_idx = chosen - T_minus_i
            new_atom = aux_atoms[h_idx]
            if state.T_max < state.T + 1:
                # Truncation hit — stay put.
                continue
            new_counts = state.counts.at[z_i_old].add(-1).at[state.T].set(1)
            new_z = state.z.at[i].set(state.T)
            state = replace(
                state,
                z=new_z,
                counts=new_counts,
                atoms=state.atoms + (new_atom,),
                T=state.T + 1,
            )

        # Drop the old cluster if it vacated and isn't the new one.
        if int(state.counts[z_i_old]) == 0 and z_i_old != int(state.z[i]):
            state = state.with_cluster_dropped(z_i_old)

    return state


def multi_cluster_sweep(
    state: MultiClusterState,
    F0: Float[Array, "n K"],
    key: jax.Array,
    *,
    H: int = 5,
    nuts_warmup: int = 30,
    py_hyperparam_warmup: int = 50,
) -> MultiClusterState:
    """One full multi-cluster Algorithm 1 sweep.

    Parameters
    ----------
    state
        Current :class:`MultiClusterState`.
    F0
        Pilot CDF matrix of shape $(n, K)$. Fixed for the duration of the chain.
    key
        JAX PRNG key.
    H
        Number of auxiliary atoms per feature in the Pólya-urn step. Default 5
        matches §3.3.1 of the report.
    nuts_warmup
        Per-cluster NUTS warmup for the atom step. Short by design — each
        cluster gets a short chain each sweep, with the chain initialised at
        the previous sweep's value.
    py_hyperparam_warmup
        NUTS warmup for the (α, σ) joint update.

    Returns
    -------
    The new :class:`MultiClusterState`.
    """
    n, K = F0.shape
    keys = jax.random.split(key, 6)
    k_class, k_polya, k_atoms, k_types, k_marginals, k_hyperparam = keys

    # ---- compute U from the current Bernstein marginals --------------------
    U = _bernstein_U(F0, state.bernstein_weights, state.M)

    # ---- Step 1: class assignment using the current cluster mixture --------
    log_dens = _per_cluster_log_density(state.atoms, U)
    log_c1 = _mixture_log_c1(log_dens, cluster_sizes(state), state.alpha, state.sigma)
    Z_new = class_assign(k_class, state.pi, log_c1)

    # After Z changes, any feature that newly went Z=1 needs a starting cluster
    # (the Pólya-urn step will then reassign properly). Default newly-reproducible
    # features to cluster 0; mark Z=0 features with -1 to be safe.
    z_masked = jnp.where(Z_new == 1, jnp.where(state.z >= 0, state.z, 0), -1)

    # Recompute occupancy counts from (Z_new, z_masked) so the active prefix
    # matches the reproducible-feature count. Without this the Pólya-urn step
    # would operate on stale counts when features transition between classes.
    counts_new = _counts_from_z(z_masked, Z_new, state.T, state.T_max)
    state = replace(state, Z=Z_new, z=z_masked, counts=counts_new)

    # ---- Step 2: Pólya-urn cluster reassignment ---------------------------
    state = _polya_urn_reassign_all(state, U, H, k_polya)

    # ---- Step 3: per-cluster NUTS atom update -----------------------------
    new_atoms: list[AtomDraw] = []
    atom_keys = jax.random.split(k_atoms, max(state.T, 1))
    for m, atom in enumerate(state.atoms):
        mask = jnp.asarray(np.asarray(state.z == m) & np.asarray(state.Z == 1))
        n_m = int(jnp.sum(mask))
        if n_m == 0:
            new_atoms.append(atom)
            continue
        # Compact the cluster's U via numpy indexing (small data).
        idx = np.asarray(np.where(np.asarray(mask))[0])
        cluster_U = U[jnp.asarray(idx)]
        atom_new, _ = update_atom_via_nuts(
            atom_keys[m], atom, cluster_U, n_warmup=nuts_warmup, n_samples=1
        )
        new_atoms.append(atom_new)
    state = replace(state, atoms=tuple(new_atoms))

    # ---- Step 4: per-cluster atomic-type IM -------------------------------
    new_atoms_type: list[AtomDraw] = []
    type_keys = jax.random.split(k_types, max(state.T, 1))
    for m, atom in enumerate(state.atoms):
        mask = jnp.asarray(np.asarray(state.z == m) & np.asarray(state.Z == 1))
        n_m = int(jnp.sum(mask))
        if n_m == 0:
            new_atoms_type.append(atom)
            continue
        idx = np.asarray(np.where(np.asarray(mask))[0])
        cluster_U = U[jnp.asarray(idx)]
        atom_new, _, _ = update_atomic_type(type_keys[m], cluster_U, atom)
        new_atoms_type.append(atom_new)
    state = replace(state, atoms=tuple(new_atoms_type))

    # ---- Step 5: Bernstein-Dirichlet conjugate update per replicate -------
    new_weights_rows: list[Float[Array, "M"]] = []
    marg_keys = jax.random.split(k_marginals, 2 * K)
    k_S_per_j = marg_keys[:K]
    k_w_per_j = marg_keys[K:]
    for j in range(K):
        S_j = sample_latent_S(k_S_per_j[j], F0[:, j], state.bernstein_weights[j], state.M)
        w_j = update_weights(k_w_per_j[j], S_j, state.M, state.alpha_F)
        new_weights_rows.append(w_j)
    new_weights = jnp.stack(new_weights_rows, axis=0)
    state = replace(state, bernstein_weights=new_weights)

    # ---- Step 7: (α, σ) joint update on the current partition -------------
    sizes_active = cluster_sizes(state)
    # The Pólya-urn dropout in step 2 should already have evacuated empty clusters,
    # but a safety filter here makes the update robust to any stragglers; the EPPF
    # update_py_hyperparams requires strictly positive cluster sizes.
    sizes = sizes_active[sizes_active > 0]
    n_reproducible = int(jnp.sum(state.Z))
    if int(sizes.shape[0]) >= 2 and n_reproducible >= 2:
        (alpha_new, sigma_new), _ = update_py_hyperparams(
            k_hyperparam,
            sizes,
            n_reproducible,
            alpha_init=state.alpha,
            sigma_init=max(state.sigma, 0.01),
            n_warmup=py_hyperparam_warmup,
            n_samples=1,
        )
        state = replace(state, alpha=alpha_new, sigma=sigma_new)

    # ---- Step 8: π update --------------------------------------------------
    pi_new = float(update_pi(k_class, state.Z))
    state = replace(state, pi=pi_new)

    return state
