"""Atomic-type registry for the PY-IDR base measure.

``ATOMIC_TYPES`` is the source of truth for the support of $t_m$ in §3.2.2 of the
revised report. The atomic-type support has size 4 (Gauss, $t_5$, Clayton, Gumbel);
the structural-class support for Gaussian / $t_5$ atoms has size 3 (exchangeable,
one-factor, two-factor) — the latter is enforced in the sampler, not here.

The registry binds each name to a representative-instance constructor and runs the
PLOD probe (:func:`py_idr.algebra.plod.assert_plod`) on registration. PLOD plays a
**scoping** role per Theorem 3.1's revised statement: it separates admissible
reproducible atoms from the independence boundary $c_0$. Identifiability of the
mixing weight $\\pi$ on the finite sieve comes from the linear-independence
criterion of Yakowitz–Spragins (1968) applied to $\\{c_0, c_{\\theta_1, t_1}, \\ldots,
c_{\\theta_T, t_T}\\}$, **not** from PLOD alone. The registry's gate is therefore a
necessary precondition, not a sufficient identifiability check.

For factor-structured Gaussian / $t_5$ atoms, the additional runtime check is
:func:`py_idr.algebra.plod.plod_with_positive_pairwise`; the sampler (Algorithm 1
step 3) either constrains factor loadings to a common sign or rejects proposals
whose normalised correlation matrix produces any negative pairwise correlation.

The proposal kernel used in Algorithm 1 step 4 (atomic-type independence Metropolis)
draws uniformly from ``ATOMIC_TYPES``.

See plans/02_algebra_and_copulas.md.
"""

from __future__ import annotations

from collections.abc import Callable

import jax.numpy as jnp

from py_idr.algebra.plod import assert_plod
from py_idr.copulas.base import Copula
from py_idr.copulas.clayton import ClaytonCopula
from py_idr.copulas.gaussian import GaussianCopula
from py_idr.copulas.gumbel import GumbelCopula
from py_idr.copulas.student_t import StudentTCopula

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
