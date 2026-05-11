"""Per-atomic-type NUTS dispatch for the multi-cluster sweep.

Each :class:`py_idr.inference.mcmc.type_update.AtomDraw` carries enough
context (``type_name``, ``params``) to route to the right per-type NUTS
updater. This module is the single switchboard so the multi-cluster sweep
doesn't need to know the per-type calling conventions.

See plans/04_inference_mcmc.md.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from py_idr.copulas.clayton import ClaytonCopula
from py_idr.copulas.gaussian import GaussianCopula
from py_idr.copulas.gumbel import GumbelCopula
from py_idr.copulas.student_t import StudentTCopula
from py_idr.inference.mcmc.nuts_atoms import (
    update_clayton,
    update_exchangeable_gaussian,
    update_gumbel_k2,
)
from py_idr.inference.mcmc.type_update import TYPE_TO_INDEX, AtomDraw


def _exchangeable_R(rho: float, K: int) -> Float[Array, "K K"]:
    """Build $R = (1-\\rho) I + \\rho \\mathbf{1}\\mathbf{1}^\\top$ given a scalar $\\rho$."""
    return (1.0 - rho) * jnp.eye(K) + rho * jnp.ones((K, K))


def update_atom_via_nuts(
    key: jax.Array,
    atom: AtomDraw,
    cluster_U: Float[Array, "n_m K"],
    *,
    n_warmup: int = 50,
    n_samples: int = 1,
) -> tuple[AtomDraw, dict[str, int]]:
    """NUTS update for one cluster's atom; dispatches on ``atom.type_name``.

    Parameters
    ----------
    key
        JAX PRNG key for the NUTS step.
    atom
        Current atom. The function reads ``atom.type_name`` and the relevant
        scalar parameter (``params["rho"]`` for Gauss / $t_5$; ``params["theta"]``
        for Clayton / Gumbel).
    cluster_U
        Length-$(n_m, K)$ pseudo-uniform data for the cluster.
    n_warmup, n_samples
        NUTS warmup and post-warmup samples; defaults are short so each per-
        cluster update is cheap.

    Returns
    -------
    (new_atom, diagnostics)
        ``new_atom`` is a fresh :class:`AtomDraw` with the same atomic type
        and the updated parameter; ``diagnostics`` carries ``num_divergences``.

    Notes
    -----
    Only ``gauss`` (exchangeable scalar $\\rho$), ``clayton`` (scalar $\\theta$),
    and bivariate ``gumbel`` (scalar $\\theta$) are dispatched here. ``t5`` is
    falls back to no-op (with a divergence count of 0) since the corresponding
    NUTS updater is the next follow-on (a t-marginal helper around the existing
    NUTS exchangeable-correlation pattern); adding it is a single new ``update_t5``
    in :mod:`py_idr.inference.mcmc.nuts_atoms`.
    """
    if cluster_U.shape[0] == 0:
        raise ValueError("update_atom_via_nuts called on an empty cluster.")
    K = cluster_U.shape[1]

    if atom.type_name == "gauss":
        rho_new, diag = update_exchangeable_gaussian(
            key, cluster_U, rho_init=atom.params["rho"], n_warmup=n_warmup, n_samples=n_samples
        )
        R = _exchangeable_R(rho_new, K)
        L = jnp.linalg.cholesky(R)
        return (
            AtomDraw(
                type_name="gauss",
                type_index=TYPE_TO_INDEX["gauss"],
                copula=GaussianCopula(L),
                params={"rho": rho_new},
            ),
            diag,
        )

    if atom.type_name == "t5":
        # No dedicated NUTS update yet; leave the atom unchanged with 0 divergences.
        # The follow-on adds update_student_t mirroring update_exchangeable_gaussian.
        return atom, {"num_divergences": 0, "n_samples": 0}

    if atom.type_name == "clayton":
        theta_new, diag = update_clayton(
            key, cluster_U, theta_init=atom.params["theta"], n_warmup=n_warmup, n_samples=n_samples
        )
        return (
            AtomDraw(
                type_name="clayton",
                type_index=TYPE_TO_INDEX["clayton"],
                copula=ClaytonCopula(theta=theta_new, K=K),
                params={"theta": theta_new},
            ),
            diag,
        )

    if atom.type_name == "gumbel":
        if K != 2:
            # Multivariate Gumbel (K ≥ 3) density not yet implemented; the atomic-type
            # IM step already excludes Gumbel proposals for K ≥ 3, so this branch
            # should never execute in practice. Defensive return.
            return atom, {"num_divergences": 0, "n_samples": 0}
        theta_new, diag = update_gumbel_k2(
            key, cluster_U, theta_init=atom.params["theta"], n_warmup=n_warmup, n_samples=n_samples
        )
        return (
            AtomDraw(
                type_name="gumbel",
                type_index=TYPE_TO_INDEX["gumbel"],
                copula=GumbelCopula(theta=theta_new, K=K),
                params={"theta": theta_new},
            ),
            diag,
        )

    raise ValueError(f"unknown atomic type {atom.type_name!r}.")


# StudentTCopula isn't currently invoked by update_atom_via_nuts (no t5 NUTS), but
# we re-export the constructor so a downstream consumer can build t_5 atoms from
# the same dispatch surface.
__all__ = [
    "GaussianCopula",
    "StudentTCopula",
    "ClaytonCopula",
    "GumbelCopula",
    "update_atom_via_nuts",
]
