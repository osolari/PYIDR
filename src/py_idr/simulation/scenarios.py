"""S1–S5 simulation scenarios from §4.1 of the revised report.

This commit implements **S1 only** — the baseline IDR regime: Gaussian copula
with $\\rho = 0.85$ and shared standard-Gaussian marginals at $\\pi^* = 0.30$.
S2/S3/S4 differ only in parameter values + heterogeneous marginals;
S5 and S5-sparse use the heavy-tail $t_5$/Clayton/Gumbel mixture. Adding
each is a small parameter change + (for S5) a new copula draw routine; they
land in subsequent commits.

The generator returns a :class:`SimulationResult` carrying raw scores ``X``,
the true class labels ``true_Z``, and the parameters used. The pilot-CDF
input to the chain is built downstream from ``X`` via
:func:`py_idr.marginals.transform.empirical_rank_transform`.

See plans/07_simulation_studies.md.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np
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
    if K < 2:
        raise ValueError(f"K must be >= 2; got {K}")
    if n < 10:
        raise ValueError(f"n must be >= 10 to be meaningful; got {n}")
    if not (0.0 < pi_star < 1.0):
        raise ValueError(f"pi_star must be in (0, 1); got {pi_star}")
    if not (0.0 < rho < 1.0):
        raise ValueError(f"rho must be in (0, 1); got {rho}")

    rng = np.random.default_rng(seed)
    n_rep = int(round(pi_star * n))
    n_irrep = n - n_rep

    # Reproducible: K-variate Gaussian with exchangeable correlation rho.
    R = (1.0 - rho) * np.eye(K) + rho * np.ones((K, K))
    z_rep = rng.multivariate_normal(np.zeros(K), R, size=n_rep)

    # Irreproducible: K independent standard Gaussians.
    z_irrep = rng.standard_normal(size=(n_irrep, K))

    X = np.vstack([z_rep, z_irrep])
    true_Z = np.concatenate([np.ones(n_rep, dtype=np.int32), np.zeros(n_irrep, dtype=np.int32)])

    # Shuffle so the chain doesn't see the structure.
    perm = rng.permutation(n)
    X = X[perm]
    true_Z = true_Z[perm]

    return SimulationResult(
        regime="S1",
        X=jnp.asarray(X),
        true_Z=jnp.asarray(true_Z),
        true_pi=float(pi_star),
        true_rho=float(rho),
        seed=int(seed),
    )
