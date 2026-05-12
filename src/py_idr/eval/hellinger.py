r"""Hellinger distance between two exchangeable Gaussian copulas.

PY-IDR's posterior-contraction diagnostic (F5) compares the
posterior-mean copula to the truth. For S1 — and only S1 — both are
exchangeable Gaussian copulas, so the comparison reduces to "Hellinger
distance between two $K$-variate Gaussian copulas with exchangeable
correlations $\rho_1$ and $\rho_2$".

The closed form
----------------

For two Gaussian copulas with correlation matrices $R_1, R_2$, the
Bhattacharyya coefficient $B = \int \sqrt{c_{R_1} c_{R_2}}\, du$ reduces
to the Bhattacharyya coefficient between the underlying $K$-variate
Gaussian distributions (the marginal $\phi$ factors cancel under the
copula change of variables $u = \Phi(z)$):

$$
B = \frac{|R_1|^{1/4} \, |R_2|^{1/4}}{|R_{\mathrm{mid}}|^{1/2}},
\qquad R_{\mathrm{mid}} = \tfrac{R_1 + R_2}{2}.
$$

For exchangeable $R = (1-\rho)I + \rho \mathbf{1}\mathbf{1}^\top$ the
determinant has a clean form:

$$
|R| = (1 - \rho)^{K-1} \, (1 + (K-1)\rho).
$$

So $H = \sqrt{1 - B}$ comes out in three multiplications + one square root —
no Monte Carlo needed. The closed form is exact, free of MC noise
floor, and trivially differentiable (useful for a future analytic
contraction-rate test).

We also expose a Monte-Carlo estimator on $[0, 1]^K$ as a sanity
companion. The two should agree to MC noise. See plans/09_figures_and_tables.md.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from py_idr.copulas.gaussian import GaussianCopula


def _exchangeable_det(rho: float, K: int) -> float:
    r"""Return $|R|$ for the exchangeable $K \times K$ correlation matrix.

    $R = (1 - \rho) I + \rho \mathbf{1}\mathbf{1}^\top$ has eigenvalues
    $1 - \rho$ (multiplicity $K - 1$) and $1 + (K - 1)\rho$
    (multiplicity 1).
    """
    return float((1.0 - rho) ** (K - 1) * (1.0 + (K - 1) * rho))


def _validate_rho(rho: float, K: int, name: str) -> None:
    """Check that ``rho`` keeps the exchangeable matrix positive-definite."""
    lo = -1.0 / (K - 1)
    if not (lo < rho < 1.0):
        raise ValueError(
            f"{name} must be in ({lo}, 1) so the exchangeable correlation "
            f"matrix is positive-definite; got {rho}"
        )


def hellinger_gaussian_copula(
    rho_1: float,
    rho_2: float,
    *,
    K: int,
) -> float:
    r"""Closed-form Hellinger distance between two exchangeable Gaussian copulas.

    Returns $H(c_{\rho_1}, c_{\rho_2}) \in [0, 1]$, the standard
    Hellinger distance (square root of squared Hellinger). Identical
    correlations yield 0 exactly; the function is symmetric in
    $(\rho_1, \rho_2)$.

    Parameters
    ----------
    rho_1, rho_2
        Exchangeable correlation parameters in $(-1/(K-1), 1)$. Both
        copulas share the same dimension $K$.
    K
        Replicate count $\ge 2$.

    Returns
    -------
    Hellinger distance, a float in $[0, 1]$.

    Raises
    ------
    ValueError
        If $K < 2$ or either correlation is outside the admissible range.
    """
    if K < 2:
        raise ValueError(f"K must be >= 2; got {K}")
    _validate_rho(rho_1, K, "rho_1")
    _validate_rho(rho_2, K, "rho_2")

    rho_mid = 0.5 * (rho_1 + rho_2)
    # rho_mid stays in (-1/(K-1), 1) by convexity, no extra check needed.
    det1 = _exchangeable_det(rho_1, K)
    det2 = _exchangeable_det(rho_2, K)
    det_mid = _exchangeable_det(rho_mid, K)
    bhatt = (det1 * det2) ** 0.25 / det_mid**0.5
    # Numerical safety: clamp the squared Hellinger into [0, 1].
    h_sq = max(0.0, min(1.0, 1.0 - bhatt))
    return float(np.sqrt(h_sq))


def hellinger_gaussian_copula_mc(
    rho_1: float,
    rho_2: float,
    *,
    K: int,
    n_mc: int = 10_000,
    rng_seed: int = 0,
) -> float:
    r"""Monte-Carlo estimator companion to :func:`hellinger_gaussian_copula`.

    Useful as a closed-form sanity check at extreme correlations or for
    non-exchangeable structure (where the closed form does not apply
    in this implementation). MC variance grows quickly near
    $\rho \to 1$; for production use prefer the closed form.

    Parameters mirror :func:`hellinger_gaussian_copula` plus ``n_mc``
    and ``rng_seed`` for the MC draw.
    """
    if K < 2:
        raise ValueError(f"K must be >= 2; got {K}")
    if n_mc < 1:
        raise ValueError(f"n_mc must be >= 1; got {n_mc}")
    _validate_rho(rho_1, K, "rho_1")
    _validate_rho(rho_2, K, "rho_2")

    rng = np.random.default_rng(rng_seed)
    U = rng.uniform(low=1e-6, high=1.0 - 1e-6, size=(n_mc, K))
    L1 = jnp.asarray(np.linalg.cholesky((1.0 - rho_1) * np.eye(K) + rho_1 * np.ones((K, K))))
    L2 = jnp.asarray(np.linalg.cholesky((1.0 - rho_2) * np.eye(K) + rho_2 * np.ones((K, K))))
    log1 = np.asarray(GaussianCopula(L1).log_density(jnp.asarray(U)))
    log2 = np.asarray(GaussianCopula(L2).log_density(jnp.asarray(U)))
    log_avg = 0.5 * (log1 + log2)
    m = float(log_avg.max())
    bhatt = float(np.exp(m) * np.mean(np.exp(log_avg - m)))
    h_sq = max(0.0, min(1.0, 1.0 - bhatt))
    return float(np.sqrt(h_sq))
