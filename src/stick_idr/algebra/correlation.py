"""Cholesky parameterization, LKJ prior, and structured-correlation classes.

See plans/02_algebra_and_copulas.md.
"""

from __future__ import annotations


def cholesky_to_corr(L):  # type: ignore[no-untyped-def]  # noqa: ANN001
    """Return R = L @ L.T (lower-triangular Cholesky factor → correlation matrix)."""
    raise NotImplementedError("W1 — see plans/02_algebra_and_copulas.md")


def corr_to_cholesky(R):  # type: ignore[no-untyped-def]  # noqa: ANN001, N803
    """Inverse of `cholesky_to_corr`."""
    raise NotImplementedError("W1 — see plans/02_algebra_and_copulas.md")


def lkj_log_density(L, eta: float = 2.0):  # type: ignore[no-untyped-def]  # noqa: ANN001
    """LKJ log-density (Lewandowski–Kurowicka–Joe 2009) on the Cholesky factor."""
    raise NotImplementedError("W1 — see plans/02_algebra_and_copulas.md")
