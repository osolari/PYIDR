"""S1–S5 simulation scenarios from §4.1 of the revised report.

All six regimes (S1, S2, S3, S4, S5, S5-sparse) live here. The
Gaussian-copula regimes share the ``_gaussian_copula_mixture_X``
helper; the heavy-tail regimes share ``_draw_s5_mixture_uniforms`` plus
per-family draw routines (``_draw_tnu_uniforms``,
``_draw_clayton_uniforms``, ``_draw_gumbel_uniforms_k2``).

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
- **S5** — heavy-tail mixture. Equal-weight mixture of $t_5$ / Clayton /
  Gumbel(K=2) copulas at matched Kendall's $\\tau$. Marginals are
  heterogeneous $t_\\nu$ via ``sigma_grid``. For $K \\ge 3$ the family
  pool collapses to $\\{t_5, \\mathrm{Clayton}\\}$ (multivariate Gumbel
  is planned, plan 11). $\\pi^* = 0.30$ default.
- **S5-sparse** — S5 mixture with $\\pi^* = 0.10$; planned ROC/PR
  stress test.

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


# ---- S5: heavy-tail mixture ---------------------------------------------------------


# Default family pool. Gumbel is K=2 only today (multivariate Hofert is planned
# in plan 11); for K >= 3 the pool falls back to {t5, clayton}.
_S5_FAMILY_POOL_K2: tuple[str, ...] = ("t5", "clayton", "gumbel")
_S5_FAMILY_POOL_KGE3: tuple[str, ...] = ("t5", "clayton")


def simulate_S5(
    *,
    K: int,
    n: int,
    pi_star: float = 0.30,
    tau: float = 0.6,
    seed: int = 0,
    sigma_grid: tuple[float, ...] = (1.0, 1.5, 2.0),
    nu: int = 5,
) -> SimulationResult:
    """Generate one realisation of regime S5 (heavy-tail mixture).

    The reproducible component is an **equal-weight mixture** of three
    Archimedean / elliptical copula families calibrated to the same
    Kendall's $\\tau$:

    - $t_\\nu$ elliptical copula with $\\rho = \\sin(\\pi \\tau / 2)$.
    - Clayton with $\\theta_{\\mathrm{Cl}} = 2\\tau / (1 - \\tau)$
      (lower-tail dependence).
    - Gumbel with $\\theta_{\\mathrm{Gu}} = 1 / (1 - \\tau)$
      (upper-tail dependence). **$K = 2$ only**; for $K \\ge 3$ the
      pool falls back to $\\{t_\\nu, \\mathrm{Clayton}\\}$ because the
      multivariate Gumbel sampler is not yet implemented (plan 11).

    Each reproducible feature picks one family uniformly from the active
    pool. The matched-$\\tau$ parameterisation ensures the bivariate
    Spearman correlation is comparable across families, per §4.1 of the
    report.

    Marginals are heterogeneous $t_\\nu$ (heavy-tailed) with
    per-replicate scale cycled from ``sigma_grid``.

    Parameters
    ----------
    K
        Number of replicates.
    n
        Total feature count.
    pi_star
        Reproducible mass; default 0.30 matches Table 3 S5.
    tau
        Reference Kendall's $\\tau$; the three families are calibrated to
        match it. Must be in $(0, 1)$.
    seed
        Random seed.
    sigma_grid
        Per-replicate $t_\\nu$ scale; cycled mod $K$.
    nu
        Degrees of freedom for the $t_\\nu$ elliptical copula *and* the
        per-replicate marginal. Default 5.

    Returns
    -------
    A :class:`SimulationResult` with ``regime="S5"`` and ``true_rho``
    set to :data:`math.nan` (no single $\\rho$ is meaningful under a
    family mixture; downstream consumers should branch on ``regime``).
    """
    _validate_basic_args(K, n, pi_star, rho=0.5)  # rho irrelevant — pass valid placeholder
    if not (0.0 < tau < 1.0):
        raise ValueError(f"tau must be in (0, 1); got {tau}")
    if nu < 1:
        raise ValueError(f"nu must be >= 1; got {nu}")
    if not sigma_grid or any(s <= 0.0 for s in sigma_grid):
        raise ValueError(
            f"sigma_grid must be a non-empty tuple of positive floats; got {sigma_grid}"
        )

    rng = np.random.default_rng(seed)
    n_rep = int(round(pi_star * n))
    n_irrep = n - n_rep

    # Pick the active family pool by K.
    family_pool = _S5_FAMILY_POOL_K2 if K == 2 else _S5_FAMILY_POOL_KGE3

    # Reproducible block: copula-mixture draw on (0, 1)^K.
    u_rep, _families = _draw_s5_mixture_uniforms(rng, K, n_rep, tau, family_pool, nu=nu)
    # Irreproducible block: independence copula.
    u_irrep = rng.uniform(size=(n_irrep, K))
    U = np.vstack([u_rep, u_irrep])

    # Apply per-replicate heterogeneous t_nu marginals.
    sigmas = np.array([sigma_grid[j % len(sigma_grid)] for j in range(K)])
    X = np.column_stack([sigmas[j] * sps.t.ppf(U[:, j], df=nu) for j in range(K)])
    true_Z = np.concatenate([np.ones(n_rep, dtype=np.int32), np.zeros(n_irrep, dtype=np.int32)])

    # rho is undefined for S5 (mixture); store nan.
    return _finalise(rng, "S5", X, true_Z, pi_star, float("nan"), seed)


def simulate_S5_sparse(
    *,
    K: int,
    n: int,
    pi_star: float = 0.10,
    tau: float = 0.6,
    seed: int = 0,
    sigma_grid: tuple[float, ...] = (1.0, 1.5, 2.0),
    nu: int = 5,
) -> SimulationResult:
    """Generate one realisation of S5-sparse — S5 with smaller reproducible mass.

    Identical to :func:`simulate_S5` except the default ``pi_star`` drops
    from 0.30 to 0.10. The report's S5-sparse cell is the planned
    ROC/PR stress test for the heavy-tail multi-cluster setting.
    """
    return _SimulationResult_relabel(
        simulate_S5(K=K, n=n, pi_star=pi_star, tau=tau, seed=seed, sigma_grid=sigma_grid, nu=nu),
        "S5-sparse",
    )


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


def _SimulationResult_relabel(sim: SimulationResult, regime: str) -> SimulationResult:
    """Return a new :class:`SimulationResult` with a different ``regime`` label.

    Used by S5-sparse, which is structurally S5 with a smaller
    ``pi_star`` but should report a distinct regime label downstream.
    """
    return SimulationResult(
        regime=regime,
        X=sim.X,
        true_Z=sim.true_Z,
        true_pi=sim.true_pi,
        true_rho=sim.true_rho,
        seed=sim.seed,
    )


# ---- S5 copula-mixture draw routines ------------------------------------------------


def _draw_s5_mixture_uniforms(
    rng: np.random.Generator,
    K: int,
    n_rep: int,
    tau: float,
    family_pool: tuple[str, ...],
    nu: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample $n_\\mathrm{rep}$ rows from an equal-weight copula mixture.

    Returns ``(U, families)`` where ``U`` is the $(n_\\mathrm{rep}, K)$
    array of copula-scale uniforms and ``families`` is the
    $(n_\\mathrm{rep},)$ object array recording each feature's family
    label.
    """
    if n_rep == 0:
        return np.empty((0, K)), np.empty(0, dtype=object)

    # Family-specific parameters matched at Kendall's tau.
    rho_t = float(np.sin(np.pi * tau / 2.0))
    theta_clayton = float(2 * tau / (1 - tau))
    theta_gumbel = float(1.0 / (1 - tau))

    families = rng.choice(np.array(family_pool), size=n_rep)
    U = np.zeros((n_rep, K))

    for fam in family_pool:
        mask = families == fam
        nf = int(mask.sum())
        if nf == 0:
            continue
        if fam == "t5":
            U[mask] = _draw_tnu_uniforms(rng, rho_t, K, nf, nu=nu)
        elif fam == "clayton":
            U[mask] = _draw_clayton_uniforms(rng, theta_clayton, K, nf)
        elif fam == "gumbel":
            if K != 2:
                raise ValueError(
                    f"Gumbel draw is implemented only for K=2; got K={K}. "
                    "Use _S5_FAMILY_POOL_KGE3 for K>=3."
                )
            U[mask] = _draw_gumbel_uniforms_k2(rng, theta_gumbel, nf)
        else:
            raise ValueError(f"unknown family label {fam!r}")

    # Clip to the open cube — t_nu PPF / CDF round-trip can return 0/1 at
    # the extremes, which downstream callers (e.g. sps.t.ppf for the
    # marginal) don't like.
    return np.clip(U, 1e-10, 1.0 - 1e-10), families


def _draw_tnu_uniforms(
    rng: np.random.Generator,
    rho: float,
    K: int,
    n: int,
    nu: int,
) -> np.ndarray:
    """Sample from the $t_\\nu$ copula with exchangeable correlation $\\rho$."""
    R = (1.0 - rho) * np.eye(K) + rho * np.ones((K, K))
    Z = rng.multivariate_normal(np.zeros(K), R, size=n)
    chi2 = rng.chisquare(df=nu, size=n)
    T = Z / np.sqrt(chi2[:, None] / nu)
    return sps.t.cdf(T, df=nu)


def _draw_clayton_uniforms(
    rng: np.random.Generator,
    theta: float,
    K: int,
    n: int,
) -> np.ndarray:
    """Marshall-Olkin frailty draw for the Clayton copula.

    Theory: a Clayton-$\\theta$ random vector is generated by drawing
    one Gamma frailty $M \\sim \\Gamma(1/\\theta, 1)$ shared across
    coordinates plus $K$ independent $\\mathrm{Exp}(1)$ jumps; the
    coordinates are $U_j = (1 + E_j / M)^{-1/\\theta}$.
    """
    M = rng.gamma(shape=1.0 / theta, scale=1.0, size=n)
    E = rng.exponential(scale=1.0, size=(n, K))
    return (1.0 + E / M[:, None]) ** (-1.0 / theta)


def _draw_gumbel_uniforms_k2(
    rng: np.random.Generator,
    theta: float,
    n: int,
) -> np.ndarray:
    """Bivariate Gumbel draw via positive-$\\alpha$-stable frailty.

    The Gumbel copula is generated by the positive $\\alpha$-stable
    frailty with $\\alpha = 1/\\theta \\in (0, 1)$. Concretely, the
    construction is:

    1. Draw $V \\sim S_\\alpha(1, 1, 0)$ — totally positively skewed
       stable with stability $\\alpha$ and scale $\\cos(\\pi\\alpha/2)^{1/\\alpha}$,
       which has Laplace transform $\\mathbb{E}[e^{-tV}] = e^{-t^\\alpha}$.
    2. Draw $E_1, E_2 \\sim \\mathrm{Exp}(1)$ independently of $V$ and
       each other.
    3. Set $U_j = \\exp(-(E_j / V)^\\alpha)$.

    Step 1 uses :func:`scipy.stats.levy_stable.rvs` with the scipy
    "S0" parameterisation; the explicit scale is what makes the
    LST match the Gumbel generator.
    """
    alpha = 1.0 / theta  # in (0, 1) since theta > 1
    # scipy's levy_stable: in the S0 parameterisation, β=1 and
    # scale = (cos(πα/2))**(1/α) yields V with LST exp(-t^α).
    scale = float(np.cos(np.pi * alpha / 2.0) ** (1.0 / alpha))
    V = sps.levy_stable.rvs(
        alpha=alpha,
        beta=1.0,
        loc=0.0,
        scale=scale,
        size=n,
        random_state=rng,
    )
    # V should be strictly positive; numerical issues occasionally produce
    # tiny non-positive values — clip them.
    V = np.maximum(V, 1e-300)
    E = rng.exponential(scale=1.0, size=(n, 2))
    return np.exp(-((E / V[:, None]) ** alpha))
