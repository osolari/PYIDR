"""The independence copula $c_0(u) = 1$ used as the irreproducible component of IDR.

The diagonal section is $C_0(u, \\ldots, u) = u^K$ exactly, so $c_0$ is the unique
family with PLOD slack zero — the point on which the report's identifiability
argument hinges (Theorem 3.1, step 2).

See plans/02_algebra_and_copulas.md.
"""

from __future__ import annotations

import jax.numpy as jnp
from jaxtyping import Array, Float

from py_idr.copulas.base import Copula


class IndependenceCopula(Copula):
    r"""The independence copula on $[0, 1]^K$.

    The independence copula factors as $C^\bot(u) = \prod_j u_j$. Its
    density is identically 1; its diagonal section is $u^K$. PY-IDR uses
    it for the **irreproducible component**: features with $Z_i = 0$
    have copula-scale data drawn from $\mathrm{Uniform}(0, 1)^K$ in each
    coordinate, with no across-replicate dependence.

    The class is intentionally parameter-free — there is nothing to fit.
    The chain only ever evaluates the density (a constant 0 on the log
    scale) and never proposes parameter updates for it.

    Parameters
    ----------
    K
        Replicate count. Determines the shape of the diagonal section
        $C^\bot(u, \ldots, u) = u^K$.
    """

    def __init__(self, K: int) -> None:
        """Bind the replicate dimension."""
        self.K = K

    def log_density(self, u: Float[Array, "n K"]) -> Float[Array, "n"]:
        """Return zero — the independence copula density is identically one."""
        return jnp.zeros(u.shape[0])

    def cdf_diagonal(self, u: Float[Array, "n"]) -> Float[Array, "n"]:
        """Return $u^K$ — the diagonal section of the independence copula."""
        return u**self.K
