"""S1–S5 simulation scenarios from §4.1 of the revised report.

Implemented here: **S1–S4** — all four Gaussian-copula regimes from Table 3.
S5 / S5-sparse use the heavy-tail $t_5$ / Clayton / Gumbel mixture and live
in their own module (one new copula draw routine each); they land in a
subsequent commit.

Quick map (defaults; full Table 3 in §4.1 of the report):

- **S1** — baseline IDR. Gaussian copula, $\\rho = 0.85$, shared
  standard-Gaussian marginals, $\\pi^* = 0.30$.
- **S2** — weak signal. Identical to S1 with $\\rho = 0.40$ instead; the
  Sun--Cai rule should still control FDR but at substantially lower power.
- **S3** — sparse + heterogeneous Gaussian. $\\pi^* = 0.10$ and per-replicate
  scale $\\sigma_j$ cycled from ``sigma_grid``; rank pilot strips the scale,
  so the chain still sees a unit-uniform marginal.
- **S4** — marginal mismatch via skew-normal marginals. Per-replicate
  skewness cycled from ``skewness_grid``; the latent copula matches S1.

All generators return a :class:`SimulationResult` carrying raw scores ``X``,
the true class labels ``true_Z``, and the parameters used. The pilot-CDF
input to the chain is built downstream from ``X`` via
:func:`py_idr.marginals.transform.empirical_rank_transform`.

See plans/07_simulation_studies.md.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np
import scipy.stats as sps
from jaxtyping import Array, Float, Integer


@dataclass(frozen=True)
class SimulationResult:
    """One realisation of an S1–S5 simulation cell.

    Attributes
    ----------
    regime
        Regime label (``"S1"``, ``"S2"``, ...) for downstream filtering.
    X
        Raw replicate scores; shape $(n, K)$.
    true_Z
        Ground-truth class indicators in $\\{0, 1\\}$.
    true_pi
        Ground-truth reproducible mass.
    true_rho
        Ground-truth correlation (only meaningful when the reproducible
        component is Gaussian).
    seed
        Random seed used; recorded so the seed ledger can reproduce.
    K, n
        Replicate and feature counts; cached from ``X.shape``.
    """

    regime: str
    X: Float[Array, "n K"]
    true_Z: Integer[Array, "n"]
    true_pi: float
    true_rho: float
    seed: int

    @property
    def K(self) -> int:
        """Replicate count, derived from ``X``."""
        return int(self.X.shape[1])

    @property
    def n(self) -> int:
        """Feature count, derived from ``X``."""
        return int(self.X.shape[0])


def _gaussian_copula_mixture_X(
    rng: np.random.Generator,
    K: int,
    n: int,
    pi_star: float,
    rho: float,
    marginal_x_fn: Callable[[np.ndarray, int], np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """Shared core: produce ``(X, true_Z)`` from a Gaussian-copula mixture.

    Pipeline:

    1. Reproducible block — draw $Z \\sim \\mathcal{N}_K(0, R)$ with exchangeable $\\rho$.
    2. Apply $\\Phi$ to obtain the copula uniforms $U_{\\text{rep}}$.
    3. Irreproducible block — draw $U_{\\text{irrep}}$ as independent ``Uniform(0,1)``.
    4. Per-replicate, apply ``marginal_x_fn(u_column, j)`` to get $X_{:,j}$ on
       the observed scale. S1/S2 use $\\Phi^{-1}$; S3 scales per replicate;
       S4 uses a skew-normal inverse CDF.

    The block layout (reproducible-first, then irreproducible) is unshuffled —
    the public entrypoint shuffles via :func:`_finalise` so the chain does not
    see the structure.
    """
    n_rep = int(round(pi_star * n))
    n_irrep = n - n_rep
    R = (1.0 - rho) * np.eye(K) + rho * np.ones((K, K))
    z_rep = rng.multivariate_normal(np.zeros(K), R, size=n_rep)
    u_rep = sps.norm.cdf(z_rep)
    u_irrep = rng.uniform(size=(n_irrep, K))
    U = np.vstack([u_rep, u_irrep])
    X = np.column_stack([marginal_x_fn(U[:, j], j) for j in range(K)])
    true_Z = np.concatenate([np.ones(n_rep, dtype=np.int32), np.zeros(n_irrep, dtype=np.int32)])
    return X, true_Z


def simulate_S1(
    *,
    K: int,
    n: int,
    pi_star: float = 0.30,
    rho: float = 0.85,
    seed: int = 0,
) -> SimulationResult:
    """Generate one realisation of regime S1 (baseline IDR; shared marginals).

    The reproducible component is a $K$-variate Gaussian copula with exchangeable
    correlation $\\rho$, transformed to standard-Gaussian marginals; the
    irreproducible component is the independence copula with the same
    marginals. The two are mixed at fraction $\\pi^*$.

    Parameters
    ----------
    K
        Number of replicates.
    n
        Total feature count.
    pi_star
        Reproducible mass; default 0.30 matches Table 3 of the report.
    rho
        Gaussian-copula correlation; default 0.85 matches Table 3 S1.
    seed
        Random seed; the generator is fully reproducible from this.

    Returns
    -------
    A :class:`SimulationResult` with ``X`` of shape $(n, K)$ on the raw
    Gaussian scale, ``true_Z`` of shape $(n,)$, and the parameter record.
    """
    _validate_basic_args(K, n, pi_star, rho)
    rng = np.random.default_rng(seed)

    # S1 uses shared standard-Gaussian marginals — the inverse-CDF is Φ⁻¹.
    def marginal(u_col: np.ndarray, j: int) -> np.ndarray:
        return sps.norm.ppf(u_col)

    X, true_Z = _gaussian_copula_mixture_X(rng, K, n, pi_star, rho, marginal)
    return _finalise(rng, "S1", X, true_Z, pi_star, rho, seed)


def simulate_S2(
    *,
    K: int,
    n: int,
    pi_star: float = 0.30,
    rho: float = 0.40,
    seed: int = 0,
) -> SimulationResult:
    """Generate one realisation of regime S2 (weak signal; shared marginals).

    Identical structure to S1 but with $\\rho = 0.40$ — a deliberately weaker
    reproducible correlation. The Sun--Cai threshold rule has more difficulty
    distinguishing the reproducible component when $\\rho$ is small; S2 stresses
    the small-effect regime.

    Parameters mirror :func:`simulate_S1`; only the default $\\rho$ differs.
    """
    _validate_basic_args(K, n, pi_star, rho)
    rng = np.random.default_rng(seed)

    def marginal(u_col: np.ndarray, j: int) -> np.ndarray:
        return sps.norm.ppf(u_col)

    X, true_Z = _gaussian_copula_mixture_X(rng, K, n, pi_star, rho, marginal)
    return _finalise(rng, "S2", X, true_Z, pi_star, rho, seed)


def simulate_S3(
    *,
    K: int,
    n: int,
    pi_star: float = 0.10,
    rho: float = 0.85,
    seed: int = 0,
    sigma_grid: tuple[float, ...] = (1.0, 1.5, 2.0),
) -> SimulationResult:
    """Generate one realisation of regime S3 (sparse signal; heterogeneous Gaussian marginals).

    Reproducible fraction is sparse ($\\pi^* = 0.10$) and each replicate has a
    different marginal *scale*. The per-replicate standard deviation cycles
    through ``sigma_grid``; the rotated marginals stress the Bernstein-Dirichlet
    correction since the empirical-rank pilot CDF maps every replicate to the
    same uniform.

    Parameters
    ----------
    sigma_grid
        Tuple of per-replicate $\\sigma_j$ values; cycled mod $K$. Default
        ``(1.0, 1.5, 2.0)`` matches Table 3 of the report.
    """
    _validate_basic_args(K, n, pi_star, rho)
    if not sigma_grid or any(s <= 0.0 for s in sigma_grid):
        raise ValueError(
            f"sigma_grid must be a non-empty tuple of positive floats; got {sigma_grid}"
        )
    sigmas = np.array([sigma_grid[j % len(sigma_grid)] for j in range(K)])
    rng = np.random.default_rng(seed)

    def marginal(u_col: np.ndarray, j: int) -> np.ndarray:
        # Normal marginal with per-replicate scale sigma_j: X = sigma_j * Φ⁻¹(U).
        return sigmas[j] * sps.norm.ppf(u_col)

    X, true_Z = _gaussian_copula_mixture_X(rng, K, n, pi_star, rho, marginal)
    return _finalise(rng, "S3", X, true_Z, pi_star, rho, seed)


def simulate_S4(
    *,
    K: int,
    n: int,
    pi_star: float = 0.30,
    rho: float = 0.85,
    seed: int = 0,
    skewness_grid: tuple[float, ...] = (-1.0, 0.0, 1.0),
) -> SimulationResult:
    """Generate one realisation of regime S4 (marginal mismatch; heterogeneous skewed marginals).

    Each replicate's marginal is a skew-normal with a different shape
    parameter cycled from ``skewness_grid``. The latent copula structure is
    unchanged from S1 ($\\rho = 0.85$, $\\pi^* = 0.30$); only the observed
    marginals differ, which stresses the Bernstein-Dirichlet correction —
    rank-only methods will not see the asymmetry, but the marginal model has
    to learn it for the local-idr calibration to remain accurate.

    Parameters
    ----------
    skewness_grid
        Per-replicate skewness $\\xi_j$; cycled mod $K$. Default
        ``(-1.0, 0.0, 1.0)`` matches Table 3 of the report.
    """
    _validate_basic_args(K, n, pi_star, rho)
    if not skewness_grid:
        raise ValueError("skewness_grid must be non-empty.")
    skews = np.array([skewness_grid[j % len(skewness_grid)] for j in range(K)])
    rng = np.random.default_rng(seed)

    def marginal(u_col: np.ndarray, j: int) -> np.ndarray:
        # Skew-normal inverse CDF — scipy's parameterisation: a = shape (Azzalini).
        return sps.skewnorm.ppf(u_col, a=skews[j])

    X, true_Z = _gaussian_copula_mixture_X(rng, K, n, pi_star, rho, marginal)
    return _finalise(rng, "S4", X, true_Z, pi_star, rho, seed)


# ---- internal helpers --------------------------------------------------------------


def _validate_basic_args(K: int, n: int, pi_star: float, rho: float) -> None:
    """Shared validators across S1–S4."""
    if K < 2:
        raise ValueError(f"K must be >= 2; got {K}")
    if n < 10:
        raise ValueError(f"n must be >= 10 to be meaningful; got {n}")
    if not (0.0 < pi_star < 1.0):
        raise ValueError(f"pi_star must be in (0, 1); got {pi_star}")
    if not (0.0 < rho < 1.0):
        raise ValueError(f"rho must be in (0, 1); got {rho}")


def _finalise(
    rng: np.random.Generator,
    regime: str,
    X: np.ndarray,
    true_Z: np.ndarray,
    pi_star: float,
    rho: float,
    seed: int,
) -> SimulationResult:
    """Shuffle features so the chain doesn't see the (reproducible-first) block layout."""
    perm = rng.permutation(X.shape[0])
    return SimulationResult(
        regime=regime,
        X=jnp.asarray(X[perm]),
        true_Z=jnp.asarray(true_Z[perm]),
        true_pi=float(pi_star),
        true_rho=float(rho),
        seed=int(seed),
    )
