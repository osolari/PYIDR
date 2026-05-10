"""SamplerState — the PyTree carried through Algorithm 1.

State layout follows §3.3.1 of the revised report. The container is a frozen
dataclass registered as a JAX PyTree so the sweep orchestrator can ``jit`` /
``vmap`` over it. All shape-relevant constants ($n$, $K$, ``T_max``) are
implicit in the array shapes — there are no Python integers stored as state.

The atom parameter and atomic-type arrays are sized at the truncation
``T_max``; only the prefix of length ``T`` (also stored as an array) is the
"active" prefix at any given sweep. The Pólya-urn step adjusts ``T`` and
projects onto active entries before computing the predictive (see
:mod:`py_idr.pym.polya_urn`).

This module is intentionally **forward-compatible**: a few fields are
placeholders that will be elaborated in subsequent commits (NUTS atom step
needs a full atom-parameter array; the marginal block needs per-replicate
Bernstein-weight matrices). The fields below capture what the sweep needs
*now*, without committing to a final layout for those subsequent pieces.

See plans/04_inference_mcmc.md.
"""

from __future__ import annotations

from dataclasses import dataclass, fields

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Integer


@dataclass(frozen=True)
class SamplerState:
    """Pólya-urn auxiliary-atom MCMC state for Algorithm 1.

    Attributes
    ----------
    pi
        Scalar reproducible-component mixing proportion $\\pi \\in (0, 1)$.
    Z
        Length-$n$ class indicators $Z_i \\in \\{0, 1\\}$.
    z
        Length-$n$ cluster assignments. ``z[i]`` is meaningful only when
        ``Z[i] == 1``; the irreproducible-class entries hold any sentinel
        (we use ``-1``) and are masked out by the sweep.
    counts
        Length-``T_max`` cluster occupancy counts; entries beyond ``T`` are 0.
    T
        Scalar (0-D) active cluster count, stored as a JAX integer scalar so
        the whole state remains a single PyTree. The integer value is read
        out via ``int(state.T)`` when needed by the orchestrator.
    alpha, sigma
        Pitman--Yor hyperparameters. ``alpha > 0``, ``sigma in [0, 1)``.
    alpha_F
        Bernstein-Dirichlet smoothing hyperparameter (``alpha_F > 0``).
    M
        Length-$K$ vector of Bernstein degrees for each replicate.
        Stored as integer scalars to keep the PyTree shape stable across
        sweeps; the discrete prior support is enforced by the marginal-degree
        update.
    bernstein_weights
        Shape ``(K, M_max)`` array of per-replicate Bernstein weights.
        Only the first ``M[j]`` entries of row $j$ are meaningful; trailing
        entries are zeros.

    Notes
    -----
    ``T_max`` and ``M_max`` are implicit in the arrays' shapes. They are
    chosen at run start (``T_max`` is the cluster truncation; ``M_max = 80``
    is the largest Bernstein degree from the discrete prior support).
    """

    pi: Float[Array, ""]
    Z: Integer[Array, "n"]
    z: Integer[Array, "n"]
    counts: Integer[Array, "T_max"]
    T: Integer[Array, ""]
    alpha: Float[Array, ""]
    sigma: Float[Array, ""]
    alpha_F: Float[Array, ""]
    M: Integer[Array, "K"]
    bernstein_weights: Float[Array, "K M_max"]

    @property
    def n(self) -> int:
        """Feature count $n$ — derived from the ``Z`` array length."""
        return int(self.Z.shape[0])

    @property
    def K(self) -> int:
        """Replicate count $K$ — derived from the ``M`` array length."""
        return int(self.M.shape[0])

    @property
    def T_max(self) -> int:
        """Cluster truncation — derived from ``counts.shape[0]``."""
        return int(self.counts.shape[0])

    @property
    def M_max(self) -> int:
        """Max Bernstein degree — derived from ``bernstein_weights.shape[1]``."""
        return int(self.bernstein_weights.shape[1])


def _flatten(state: SamplerState) -> tuple[tuple, tuple]:
    """JAX PyTree flatten: every field is a leaf; no static aux data."""
    leaves = tuple(getattr(state, f.name) for f in fields(state))
    return leaves, ()


def _unflatten(_: tuple, leaves: tuple) -> SamplerState:
    """JAX PyTree unflatten: rebuild from the leaves in field order."""
    return SamplerState(*leaves)


jax.tree_util.register_pytree_node(SamplerState, _flatten, _unflatten)


def initial_state(
    *,
    n: int,
    K: int,
    T_max: int,
    M_max: int = 80,
    pi: float = 0.5,
    alpha: float = 1.0,
    sigma: float = 0.0,
    alpha_F: float = 1.0,
    M_init: int = 10,
) -> SamplerState:
    """Build a default-initial :class:`SamplerState` of the requested shape.

    Used at the start of a fit; the sweep then mutates the state. Feature
    classes default to ``Z = 1`` (everything reproducible), cluster assignments
    to 0, and Bernstein weights to uniform on the first ``M_init`` columns.
    """
    if not (0.0 < pi < 1.0):
        raise ValueError(f"pi must be in (0, 1); got {pi}")
    if alpha <= 0.0:
        raise ValueError(f"alpha must be > 0; got {alpha}")
    if not (0.0 <= sigma < 1.0):
        raise ValueError(f"sigma must be in [0, 1); got {sigma}")
    if alpha_F <= 0.0:
        raise ValueError(f"alpha_F must be > 0; got {alpha_F}")
    if M_init not in (5, 10, 20, 40, 80) or M_init > M_max:
        raise ValueError(f"M_init must be a degree-prior support point ≤ M_max; got {M_init}")

    counts = jnp.zeros(T_max, dtype=jnp.int32).at[0].set(n)
    bernstein_weights = jnp.zeros((K, M_max), dtype=jnp.float32)
    bernstein_weights = bernstein_weights.at[:, :M_init].set(1.0 / M_init)

    return SamplerState(
        pi=jnp.asarray(pi, dtype=jnp.float32),
        Z=jnp.ones(n, dtype=jnp.int32),
        z=jnp.zeros(n, dtype=jnp.int32),
        counts=counts,
        T=jnp.asarray(1, dtype=jnp.int32),
        alpha=jnp.asarray(alpha, dtype=jnp.float32),
        sigma=jnp.asarray(sigma, dtype=jnp.float32),
        alpha_F=jnp.asarray(alpha_F, dtype=jnp.float32),
        M=jnp.full(K, M_init, dtype=jnp.int32),
        bernstein_weights=bernstein_weights,
    )
