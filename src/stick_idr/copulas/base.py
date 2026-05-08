"""Abstract Copula base class.

See plans/02_algebra_and_copulas.md for the full module design and the contract every
concrete atomic type must satisfy.
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class Copula(ABC):
    """Abstract copula density on [0, 1]^K."""

    @abstractmethod
    def log_density(self, u):  # type: ignore[no-untyped-def]  # noqa: ANN001
        """Return log c(u) at each row of u; vmap-friendly."""
        raise NotImplementedError

    @abstractmethod
    def cdf_diagonal(self, u):  # type: ignore[no-untyped-def]  # noqa: ANN001
        """Return C(u, ..., u) on a 1D grid (used by `algebra.plod.holds`)."""
        raise NotImplementedError
