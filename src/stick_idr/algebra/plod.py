"""Positive lower-orthant dependence (PLOD) probe.

PLOD: C(u, ..., u) > u^K for all u in (0, 1). Eq. (3.4) of the report.
The PLOD constraint is the hinge of Theorem 3.1 (identifiability via Yakowitz-Spragins
plus diagonal-section separation).

See plans/02_algebra_and_copulas.md.
"""

from __future__ import annotations


def holds(copula, u_grid):  # type: ignore[no-untyped-def]  # noqa: ANN001
    """Return a boolean array over u_grid testing C(u,...,u) > u^K."""
    raise NotImplementedError("W1 — see plans/02_algebra_and_copulas.md")


def assert_plod(copula, n_grid: int = 99) -> None:  # type: ignore[no-untyped-def]  # noqa: ANN001
    """Raise if the copula fails PLOD on a default u-grid.

    Used by `copulas/registry.py` at import time to gate the base measure.
    """
    raise NotImplementedError("W1 — see plans/02_algebra_and_copulas.md")
