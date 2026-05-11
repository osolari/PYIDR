"""Multi-cluster sampler state for the full Algorithm 1 sweep.

Combines JAX arrays (the numerical state — class indicators, cluster
assignments, occupancy counts, Bernstein weights, hyperparameters) with a
Python tuple of :class:`py_idr.inference.mcmc.type_update.AtomDraw` (the
heterogeneous atom state — variable atomic types, variable parameter
shapes). This mixed-state design avoids forcing every atom into a fixed-shape
array; the orchestrator threads both pieces, with the JAX arrays used by
jit-compiled per-step kernels and the Python atom list used at the sweep
boundary.

The single-cluster sampler in :mod:`py_idr.inference.mcmc.sweep_simple`
remains the production fast-path for the §4.6 Gaussian-only ablation; this
module is the full-PY-mixture path.

See plans/04_inference_mcmc.md.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

import jax.numpy as jnp
from jaxtyping import Array, Float, Integer

if TYPE_CHECKING:
    from py_idr.inference.mcmc.type_update import AtomDraw


@dataclass(frozen=True)
class MultiClusterState:
    """Full PY-IDR sampler state across $T$ active clusters.

    Attributes
    ----------
    pi
        Reproducible-component mixing proportion.
    Z
        Length-$n$ class indicators in $\\{0, 1\\}$.
    z
        Length-$n$ cluster assignments. ``z[i]`` is meaningful only when
        ``Z[i] == 1``; irreproducible entries hold ``-1``.
    counts
        Length-``T_max`` occupancy counts; entries beyond ``T`` are 0.
    bernstein_weights
        Per-replicate Bernstein-Dirichlet weights; shape $(K, M)$.
    atoms
        Tuple of length ``T`` — one ``AtomDraw`` per active cluster.
    T
        Active cluster count (a Python int because the atom list length must
        match; not a JAX-traced quantity).
    M
        Bernstein degree (held fixed at run start in the current sweep).
    alpha, sigma, alpha_F
        Hyperparameters.

    Notes
    -----
    ``T_max`` is implicit in ``counts.shape[0]``. The orchestrator grows ``T``
    when an auxiliary atom is selected; if the active prefix would exceed
    ``T_max`` the sweep raises (caller chooses a larger truncation).
    """

    pi: float
    Z: Integer[Array, "n"]
    z: Integer[Array, "n"]
    counts: Integer[Array, "T_max"]
    bernstein_weights: Float[Array, "K M"]
    atoms: tuple["AtomDraw", ...]
    T: int
    M: int
    alpha: float
    sigma: float
    alpha_F: float = 1.0

    @property
    def n(self) -> int:
        """Feature count, derived from the ``Z`` array."""
        return int(self.Z.shape[0])

    @property
    def K(self) -> int:
        """Replicate count, derived from ``bernstein_weights``."""
        return int(self.bernstein_weights.shape[0])

    @property
    def T_max(self) -> int:
        """Cluster truncation, derived from ``counts``."""
        return int(self.counts.shape[0])

    def with_new_cluster(self, new_atom: "AtomDraw") -> "MultiClusterState":
        """Return a copy with one extra active cluster carrying ``new_atom``.

        The new cluster is appended at index ``T``. ``counts[T]`` is initialised
        to 0; the caller is expected to update ``counts`` and the feature's
        ``z`` entry. Raises if ``T + 1 > T_max``.
        """
        if self.T_max < self.T + 1:
            raise ValueError(
                f"cluster truncation T_max={self.T_max} reached; cannot add a new cluster."
            )
        return replace(self, atoms=self.atoms + (new_atom,), T=self.T + 1)

    def with_cluster_dropped(self, cluster_idx: int) -> "MultiClusterState":
        """Return a copy with cluster ``cluster_idx`` removed (its count is 0).

        Used after the Pólya-urn step when a singleton cluster is vacated. The
        function compacts the atom tuple and re-indexes the ``z`` and ``counts``
        arrays so the active prefix has no holes.
        """
        if not (0 <= cluster_idx < self.T):
            raise IndexError(f"cluster_idx {cluster_idx} out of range [0, {self.T}).")
        new_atoms = tuple(a for i, a in enumerate(self.atoms) if i != cluster_idx)
        # Compact counts and update z indices: any z > cluster_idx shifts down by 1.
        new_counts = jnp.zeros_like(self.counts)
        new_active_counts = jnp.concatenate(
            [self.counts[:cluster_idx], self.counts[cluster_idx + 1 : self.T]]
        )
        new_counts = new_counts.at[: self.T - 1].set(new_active_counts)
        # Re-index z. Features at the dropped cluster keep their value (-1 or whatever
        # the caller set), since the dropped cluster has 0 members.
        shifted_z = jnp.where(
            (self.Z == 1) & (self.z > cluster_idx),
            self.z - 1,
            self.z,
        )
        return replace(
            self,
            atoms=new_atoms,
            counts=new_counts,
            z=shifted_z,
            T=self.T - 1,
        )


def cluster_sizes(state: MultiClusterState) -> Integer[Array, "T"]:
    """Return the active-prefix occupancy counts as a length-$T$ array."""
    return state.counts[: state.T]


def initial_multi_cluster_state(
    *,
    n: int,
    K: int,
    M: int,
    T_max: int,
    initial_atom: "AtomDraw",
    pi_init: float = 0.5,
    alpha_init: float = 1.0,
    sigma_init: float = 0.1,
    alpha_F: float = 1.0,
) -> MultiClusterState:
    """Build a default initial state with a single active cluster."""
    if T_max < 1:
        raise ValueError(f"T_max must be >= 1; got {T_max}.")
    if not (0.0 < pi_init < 1.0):
        raise ValueError(f"pi_init must be in (0, 1); got {pi_init}.")
    if not (0.0 <= sigma_init < 1.0):
        raise ValueError(f"sigma_init must be in [0, 1); got {sigma_init}.")
    if alpha_init <= 0.0:
        raise ValueError(f"alpha_init must be > 0; got {alpha_init}.")
    counts = jnp.zeros(T_max, dtype=jnp.int32).at[0].set(n)
    weights = jnp.ones((K, M)) / M
    return MultiClusterState(
        pi=pi_init,
        Z=jnp.ones(n, dtype=jnp.int32),
        z=jnp.zeros(n, dtype=jnp.int32),
        counts=counts,
        bernstein_weights=weights,
        atoms=(initial_atom,),
        T=1,
        M=M,
        alpha=alpha_init,
        sigma=sigma_init,
        alpha_F=alpha_F,
    )


def update_counts_from_z(
    z: Integer[Array, "n"],
    Z: Integer[Array, "n"],
    T: int,
    T_max: int,
) -> Integer[Array, "T_max"]:
    """Recompute occupancy counts from the assignment / class arrays."""
    counts = jnp.zeros(T_max, dtype=jnp.int32)
    # Mask out irreproducible features.
    reproducible_z = jnp.where(Z == 1, z, -1)
    # bincount over the active prefix.
    valid_counts = jnp.bincount(jnp.maximum(reproducible_z, 0), length=T_max)
    # Subtract the implicit 0-count from the "-1 → 0" mapping.
    n_irrep = int(jnp.sum(Z == 0))
    valid_counts = valid_counts.at[0].add(-n_irrep)
    counts = counts.at[:T_max].set(valid_counts)
    return counts


del TYPE_CHECKING  # only needed at type-check time
