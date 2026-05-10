"""Positive lower-orthant dependence (PLOD) probe.

The PLOD condition (Joe 1997, Definition 2.1) is

.. math::
   C(u, \\ldots, u) > u^K \\qquad \\text{for all } u \\in (0, 1).

It is the hinge of the report's Theorem 3.1: the independence copula $c_0$ has
$C_0(u,\\ldots,u) = u^K$ exactly, so any PLOD atom is distinguishable from $c_0$ in
the diagonal section, and the linear-independence criterion of Yakowitz–Spragins
(1968) closes the identifiability argument.

The probe in this module is a runtime guard: copula families are *registered* into
the PY-IDR base measure only after they pass :func:`assert_plod` on a representative
parameter draw, and the simulation harness re-checks PLOD on every newly instantiated
atom to catch numerical edge cases.

See plans/02_algebra_and_copulas.md.
"""

from __future__ import annotations

import jax.numpy as jnp
from jaxtyping import Array, Bool, Float

from stick_idr.copulas.base import Copula


def default_grid(n_grid: int = 99) -> Float[Array, "n_grid"]:
    """Return the default open-cube grid $u \\in (0, 1)$ used by :func:`holds`.

    Endpoints are excluded since PLOD is required on the open interval and the
    diagonal section equals $u^K$ at the boundaries by definition.
    """
    return jnp.linspace(1.0 / (n_grid + 1), n_grid / (n_grid + 1), n_grid)


def holds(
    copula: Copula,
    K: int,
    u_grid: Float[Array, "n"] | None = None,
    *,
    tol: float = 1e-6,
) -> Bool[Array, "n"]:
    """Return a boolean array testing $C(u,\\ldots,u) > u^K$ on a 1-D grid.

    Parameters
    ----------
    copula
        Concrete copula instance with a ``cdf_diagonal(u)`` method.
    K
        Replicate dimension.
    u_grid
        1-D grid in (0, 1). Defaults to :func:`default_grid` with 99 points.
    tol
        Slack on the strict inequality, to absorb float round-off near the boundary.

    Returns
    -------
    Boolean array of the same shape as ``u_grid``.
    """
    if u_grid is None:
        u_grid = default_grid()
    diag = copula.cdf_diagonal(u_grid)
    return diag > u_grid**K + tol


def assert_plod(
    copula: Copula,
    K: int,
    *,
    n_grid: int = 99,
    tol: float = 1e-6,
) -> None:
    """Raise ``ValueError`` if the copula fails PLOD on the default grid.

    Used by :mod:`stick_idr.copulas.registry` at import time and by the simulation
    harness on every newly instantiated atom.
    """
    u_grid = default_grid(n_grid)
    pass_mask = holds(copula, K, u_grid, tol=tol)
    if not bool(jnp.all(pass_mask)):
        bad = u_grid[~pass_mask]
        raise ValueError(
            f"PLOD violated for {type(copula).__name__} at u={bad.tolist()}: "
            f"C(u,...,u) <= u^K (within tol={tol}). The independence copula c_0 "
            f"is the only family with diagonal section identically u^K — atoms "
            f"of the reproducible component must satisfy strict PLOD per "
            f"equation (3.4) of the report."
        )
