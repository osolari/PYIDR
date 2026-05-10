"""Atomic-type registry for the PY-IDR base measure.

``ATOMIC_TYPES`` is the source of truth for the support of $t_m$ in §3.3.2 of the
report. The registry binds each name to a representative-instance constructor and
runs the PLOD probe on registration to gate the family — a violation aborts package
import, so a non-PLOD family cannot silently slip into the base measure (which would
break Theorem 3.1's identifiability hinge).

The proposal kernel used in Algorithm 1 step 4 (atomic-type independence Metropolis)
draws uniformly from ``ATOMIC_TYPES``.

See plans/02_algebra_and_copulas.md.
"""

from __future__ import annotations

from collections.abc import Callable

import jax.numpy as jnp

from stick_idr.algebra.plod import assert_plod
from stick_idr.copulas.base import Copula
from stick_idr.copulas.clayton import ClaytonCopula
from stick_idr.copulas.gaussian import GaussianCopula
from stick_idr.copulas.gumbel import GumbelCopula
from stick_idr.copulas.student_t import StudentTCopula

ATOMIC_TYPES: tuple[str, ...] = ("gauss", "t5", "clayton", "gumbel")


def _representative_gaussian(K: int) -> Copula:
    """Build a representative positive-correlation Gaussian atom for the PLOD probe."""
    rho = 0.6
    R = (1.0 - rho) * jnp.eye(K) + rho * jnp.ones((K, K))
    L = jnp.linalg.cholesky(R)
    return GaussianCopula(L)


def _representative_student_t(K: int) -> Copula:
    """Build a representative $t_5$ atom with positive correlation."""
    rho = 0.6
    R = (1.0 - rho) * jnp.eye(K) + rho * jnp.ones((K, K))
    L = jnp.linalg.cholesky(R)
    return StudentTCopula(L)


def _representative_clayton(K: int) -> Copula:
    """Build a representative Clayton atom."""
    return ClaytonCopula(theta=2.0, K=K)


def _representative_gumbel(K: int) -> Copula:
    """Build a representative Gumbel atom."""
    return GumbelCopula(theta=2.0, K=K)


REPRESENTATIVES: dict[str, Callable[[int], Copula]] = {
    "gauss": _representative_gaussian,
    "t5": _representative_student_t,
    "clayton": _representative_clayton,
    "gumbel": _representative_gumbel,
}


def gate_with_plod(K: int = 2) -> None:
    """Run the PLOD probe on every atomic type at the given replicate dimension.

    Called at module import time with $K = 2$ — the cheapest non-trivial dimension at
    which every family is well-defined (Gumbel density is closed-form bivariate, the
    SciPy multivariate-normal CDF is fast, etc.). Per-instance PLOD checks at
    inference time use the actual $K$ of the run.
    """
    for name, builder in REPRESENTATIVES.items():
        cop = builder(K)
        try:
            assert_plod(cop, K)
        except ValueError as e:
            raise ValueError(
                f"Atomic type {name!r} failed the PLOD probe at K={K}: {e}. "
                "This indicates a bug in the corresponding copula module — refusing "
                "to register a non-PLOD family in the reproducible-component base "
                "measure (would break Theorem 3.1)."
            ) from e


# Run the gate eagerly so import-time misregistration is loud.
gate_with_plod(K=2)
