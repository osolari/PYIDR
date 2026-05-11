"""Abstract Copula base class.

Every concrete atomic type in the PY-IDR base measure subclasses :class:`Copula` and
implements :meth:`log_density` and :meth:`cdf_diagonal`. The PLOD probe in
:mod:`py_idr.algebra.plod` consumes only the diagonal section, so a custom
family can be registered without implementing the full multivariate CDF.

See plans/02_algebra_and_copulas.md.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from jaxtyping import Array, Float


class Copula(ABC):
    r"""Abstract copula density on $[0, 1]^K$.

    A *copula* $C$ is a CDF on the unit hypercube with uniform marginals;
    equivalently, $C(u_1, \ldots, u_K) = \PP(U_1 \le u_1, \ldots, U_K \le u_K)$
    where each $U_j \sim \mathrm{Uniform}(0, 1)$. The associated density
    $c = \partial^K C / \partial u_1 \cdots \partial u_K$ is what we
    evaluate inside the chain — every Algorithm 1 sweep step that touches
    the reproducible component routes through :meth:`log_density`.

    Subclasses
    ----------
    Concrete families (Gaussian, Clayton, Gumbel, $t_\nu$, independence)
    live in sibling modules; they register themselves with
    :mod:`py_idr.copulas.registry` so the type-update step can swap
    them in and out of clusters.

    Required methods
    ----------------
    Subclasses must implement :meth:`log_density` (used by the per-feature
    likelihood and by NUTS) and :meth:`cdf_diagonal` (used by the
    :func:`py_idr.algebra.plod.holds` predicate to decide whether the
    family is PLOD-compatible at the current parameters).

    Attributes
    ----------
    K
        Replicate count. Set by the concrete constructor.
    """

    K: int

    @abstractmethod
    def log_density(self, u: Float[Array, "n K"]) -> Float[Array, "n"]:
        r"""Return $\log c(u_i)$ at each row $u_i$ of ``u``.

        Implementations must be JAX-jit and ``vmap``-friendly along the leading
        (feature) axis. Inputs are assumed to lie in the open cube; callers should
        clamp $u \in (\varepsilon, 1 - \varepsilon)$ for numerical safety.
        """
        raise NotImplementedError

    @abstractmethod
    def cdf_diagonal(self, u: Float[Array, "n"]) -> Float[Array, "n"]:
        r"""Return $C(u, \ldots, u)$ on a 1-D grid in $(0, 1)$.

        Used by :func:`py_idr.algebra.plod.holds` to gate registration of the
        family into the base measure.
        """
        raise NotImplementedError
