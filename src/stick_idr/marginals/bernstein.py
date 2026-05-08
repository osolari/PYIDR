"""Bernstein basis for the per-replicate Bernstein-Dirichlet marginals.

Eq. (3.7) of the report:
    F_j(x) = sum_r p_{j,r} B_r(F_j^{(0)}(x); M_j)
with B_r(t; M) = I_t(r, M-r+1) the regularized incomplete beta.

See plans/03_marginals.md.
"""

from __future__ import annotations


def cumulative_basis(t, M: int):  # type: ignore[no-untyped-def]  # noqa: ANN001, N803
    """Return B_r(t; M) = betainc(r, M-r+1, t) for r = 1..M."""
    raise NotImplementedError("W1 — see plans/03_marginals.md")


def density_basis(t, M: int):  # type: ignore[no-untyped-def]  # noqa: ANN001, N803
    """Return b_r(t; M) = M choose(M-1, r-1) t^(r-1) (1-t)^(M-r) for r = 1..M."""
    raise NotImplementedError("W1 — see plans/03_marginals.md")
