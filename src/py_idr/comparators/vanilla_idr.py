r"""Vanilla IDR: Li, Brown, Huang & Bickel (2011) two-component bivariate IDR.

The original IDR model (henceforth "Vanilla IDR"). Two components:

- Irreproducible: independent uniform on $(0, 1)^2$.
- Reproducible: bivariate Gaussian copula with correlation $\rho$
  applied to shared standard-normal marginals.

The mixture has two scalar parameters $(\pi, \rho)$, fit by EM on the
copula-scale data $u = F_0(X) \in (0, 1)^{n \times 2}$ where $F_0$ is
the empirical-rank pilot CDF (the same pilot PY-IDR uses).

For $K \ge 3$ replicates, the original paper applies the bivariate
model to each of the $\binom{K}{2}$ replicate pairs and aggregates the
per-pair local idrs by a "consensus" rule. We implement two
aggregations:

- ``"mean"`` (default) — arithmetic mean of pairwise local idrs.
  Equivalent to "average pairwise confidence" and tracks the report's
  intuition that a feature is reproducible iff *most* pairs agree.
- ``"max"`` — maximum of pairwise local idrs. Conservative (a feature
  is treated as irreproducible if *any* pair labels it so); closest to
  the report's "intersection" wording.

References
----------
Li, Q., Brown, J. B., Huang, H., & Bickel, P. J. (2011). "Measuring
reproducibility of high-throughput experiments." *Annals of Applied
Statistics*, 5(3), 1752-1779.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import scipy.stats as sps
from jaxtyping import Array, Float


@dataclass(frozen=True)
class VanillaIDRResult:
    """One Vanilla IDR fit on a single replicate pair.

    Attributes
    ----------
    pi_hat
        Estimated reproducible mass.
    rho_hat
        Estimated Gaussian-copula correlation under the reproducible
        component.
    idr
        Per-feature local idr (length $n$).
    n_iter
        Number of EM iterations actually run.
    converged
        Whether the EM loop converged (parameter-change tolerance met).
    """

    pi_hat: float
    rho_hat: float
    idr: np.ndarray
    n_iter: int
    converged: bool


def _empirical_rank_uniform(x: np.ndarray) -> np.ndarray:
    """Rank-based pilot CDF: $F_0(x_i) = \\mathrm{rank}(x_i) / (n + 1)$.

    Mid-rank tie convention (same as
    :func:`py_idr.marginals.transform.empirical_rank_transform`).
    """
    ranks = sps.rankdata(x, method="average")
    return ranks / (len(x) + 1)


def _bivariate_gaussian_copula_logpdf(
    u: np.ndarray,
    rho: float,
    eps: float = 1e-9,
) -> np.ndarray:
    r"""Closed-form bivariate Gaussian copula log density.

    $\log c_\rho(u_1, u_2) = -\tfrac{1}{2}\log(1 - \rho^2)
    - \tfrac{\rho^2(z_1^2 + z_2^2) - 2\rho z_1 z_2}{2(1 - \rho^2)}$,

    where $z_j = \Phi^{-1}(u_j)$. Clipped at the cube boundary for
    numerical safety.
    """
    u_clipped = np.clip(u, eps, 1.0 - eps)
    z = sps.norm.ppf(u_clipped)
    z1 = z[..., 0]
    z2 = z[..., 1]
    one_minus_rho2 = 1.0 - rho * rho
    quad = (rho * rho * (z1 * z1 + z2 * z2) - 2.0 * rho * z1 * z2) / (2.0 * one_minus_rho2)
    return -0.5 * np.log(one_minus_rho2) - quad


def fit_vanilla_idr(
    X_pair: Float[Array, "n 2"],
    *,
    pi_init: float = 0.5,
    rho_init: float = 0.5,
    max_iter: int = 100,
    tol: float = 1e-5,
) -> VanillaIDRResult:
    r"""Fit the original 2-component IDR via EM on one replicate pair.

    Parameters
    ----------
    X_pair
        $(n, 2)$ array of raw scores from two replicates. Marginals are
        converted to copula-scale via the empirical-rank pilot.
    pi_init, rho_init
        Initial values for the EM chain.
    max_iter
        Maximum EM iterations.
    tol
        Convergence tolerance on $\max(|\Delta\pi|, |\Delta\rho|)$.

    Returns
    -------
    A :class:`VanillaIDRResult` with the per-feature local idr and the
    fitted scalar parameters.

    Raises
    ------
    ValueError
        If ``X_pair`` is not $(n, 2)$ or initial parameters are out of
        range.
    """
    X_pair = np.asarray(X_pair)
    if X_pair.ndim != 2 or X_pair.shape[1] != 2:
        raise ValueError(f"X_pair must be shape (n, 2); got {X_pair.shape}")
    if not (0.0 < pi_init < 1.0):
        raise ValueError(f"pi_init must be in (0, 1); got {pi_init}")
    if not (-1.0 < rho_init < 1.0):
        raise ValueError(f"rho_init must be in (-1, 1); got {rho_init}")

    # Pilot: empirical rank per column.
    u = np.column_stack(
        [_empirical_rank_uniform(X_pair[:, 0]), _empirical_rank_uniform(X_pair[:, 1])]
    )
    z = sps.norm.ppf(np.clip(u, 1e-9, 1.0 - 1e-9))

    pi, rho = float(pi_init), float(rho_init)
    converged = False
    iter_used = max_iter

    for it in range(max_iter):
        # E step: posterior responsibility that each feature is reproducible.
        # Under the reproducible component the density on the copula scale is
        # c_rho(u); under irreproducible it is the independence copula = 1.
        log_c1 = _bivariate_gaussian_copula_logpdf(u, rho)
        # numerator: pi * c_1(u); denominator: pi * c_1 + (1 - pi) * 1.
        # Compute in log-space.
        log_num = np.log(pi) + log_c1
        log_den = np.logaddexp(np.log(pi) + log_c1, np.log1p(-pi))
        p = np.exp(log_num - log_den)  # responsibility = Pr(Z_i = 1 | u_i)

        # M step:
        # pi_new = mean(p_i)
        pi_new = float(np.mean(p))

        # rho_new: weighted MLE on z = Phi^{-1}(u). Under the reproducible
        # component, z ~ N_2(0, R(rho)). Weighted score for rho cancels nicely
        # so rho is the weighted cross-correlation, normalised by the
        # weighted second moments.
        w = p
        wsum = float(np.sum(w))
        if wsum < 1e-12:
            # Degenerate: no reproducible mass. Hold rho fixed.
            rho_new = rho
        else:
            m1 = np.sum(w * z[:, 0] * z[:, 1])
            v1 = np.sum(w * z[:, 0] * z[:, 0])
            v2 = np.sum(w * z[:, 1] * z[:, 1])
            # Standard symmetric estimator: rho = sum w z1 z2 / sqrt(sum w z1^2 * sum w z2^2)
            denom = float(np.sqrt(max(v1, 1e-30) * max(v2, 1e-30)))
            rho_new = float(np.clip(m1 / denom, -0.999, 0.999))

        if max(abs(pi_new - pi), abs(rho_new - rho)) < tol:
            pi, rho = pi_new, rho_new
            converged = True
            iter_used = it + 1
            break
        pi, rho = pi_new, rho_new

    # Final E step to produce the reported per-feature local idr.
    log_c1 = _bivariate_gaussian_copula_logpdf(u, rho)
    log_num = np.log(pi) + log_c1
    log_den = np.logaddexp(np.log(pi) + log_c1, np.log1p(-pi))
    p = np.exp(log_num - log_den)
    idr = 1.0 - p

    return VanillaIDRResult(pi_hat=pi, rho_hat=rho, idr=idr, n_iter=iter_used, converged=converged)


def fit_vanilla_idr_pairwise(
    X: np.ndarray,
    *,
    aggregation: Literal["mean", "max"] = "mean",
    pi_init: float = 0.5,
    rho_init: float = 0.5,
    max_iter: int = 100,
    tol: float = 1e-5,
) -> np.ndarray:
    r"""Fit Vanilla IDR on every replicate pair and aggregate the per-feature local idrs.

    For $K = 2$ this reduces to a single :func:`fit_vanilla_idr` call.
    For $K \ge 3$, the original paper recommends applying the bivariate
    model to each of the $\binom{K}{2}$ pairs and aggregating.

    Parameters
    ----------
    X
        $(n, K)$ array of raw scores.
    aggregation
        ``"mean"`` (default) — arithmetic mean of pairwise local idrs.
        ``"max"`` — most-conservative (max over pairs).
    pi_init, rho_init, max_iter, tol
        Forwarded to each per-pair :func:`fit_vanilla_idr` call.

    Returns
    -------
    Length-$n$ aggregated local idr vector in $[0, 1]$.

    Raises
    ------
    ValueError
        If ``X.shape[1] < 2`` or ``aggregation`` is not recognised.
    """
    X = np.asarray(X)
    if X.ndim != 2 or X.shape[1] < 2:
        raise ValueError(f"X must be (n, K) with K >= 2; got {X.shape}")
    if aggregation not in ("mean", "max"):
        raise ValueError(f"aggregation must be 'mean' or 'max'; got {aggregation!r}")

    n, K = X.shape
    if K == 2:
        return fit_vanilla_idr(
            X, pi_init=pi_init, rho_init=rho_init, max_iter=max_iter, tol=tol
        ).idr

    pair_idrs: list[np.ndarray] = []
    for j in range(K):
        for k in range(j + 1, K):
            res = fit_vanilla_idr(
                X[:, [j, k]],
                pi_init=pi_init,
                rho_init=rho_init,
                max_iter=max_iter,
                tol=tol,
            )
            pair_idrs.append(res.idr)

    stack = np.stack(pair_idrs, axis=0)  # (n_pairs, n)
    if aggregation == "mean":
        return stack.mean(axis=0)
    return stack.max(axis=0)
