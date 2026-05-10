"""Per-feature posterior local idr (Eq. 3.9 of the revised report).

The local idr is the posterior probability of the irreproducible class:

.. math::
   \\widehat\\idr_i = \\Pr(Z_i = 0 \\mid \\mathrm{data})
   = \\int \\Pr(Z_i = 0 \\mid \\mathrm{data}, \\theta, w, F)\\,
       \\Pi(\\,d\\theta, dw, dF \\mid \\mathrm{data}\\,).

For an MCMC trace this collapses to a Monte Carlo average across post-warmup
samples; for a variational posterior it reads $q(Z_i = 0)$ directly.

The :func:`local_idr_error_scale` helper estimates $\\delta_n$ — the local-idr
error scale that appears as the rate in Theorem 3.3 of the revised report —
by comparing per-feature posterior estimates across split chains. The verdict
ledger records $\\delta_n$ alongside realised FDR for each fit.

See plans/06_decision_theory.md.
"""

from __future__ import annotations

import jax.numpy as jnp
from jaxtyping import Array, Float


def from_mcmc(z_samples: Float[Array, "draws n"]) -> Float[Array, "n"]:
    """Compute per-feature local idr from Bernoulli $Z_i$ samples across the chain.

    Parameters
    ----------
    z_samples
        2-D array indexed as ``(draw, feature)``; each entry is the sampled
        $Z_i \\in \\{0, 1\\}$ at that post-warmup draw. Multi-chain inputs should be
        flattened across ``(chain, draw)`` before calling.

    Returns
    -------
    A length-$n$ vector $\\widehat\\idr_i = (1/D) \\sum_d (1 - Z_{i,d})$ — the
    Monte Carlo estimator of $\\Pr(Z_i = 0 \\mid \\mathrm{data})$.
    """
    z = jnp.asarray(z_samples)
    if z.ndim != 2:
        raise ValueError(f"z_samples must be (draws, n); got shape {z.shape}")
    return 1.0 - jnp.mean(z.astype(jnp.float32), axis=0)


def from_vi(q_z_eq_zero: Float[Array, "n"]) -> Float[Array, "n"]:
    """Read per-feature local idr from a variational categorical posterior.

    Parameters
    ----------
    q_z_eq_zero
        Length-$n$ vector $q(Z_i = 0)$ from the structured mean-field family
        (plan 05). This is already the local idr by definition; the function
        is provided for API symmetry with :func:`from_mcmc`.

    Returns
    -------
    The input vector unchanged after a sanity check that values lie in $[0, 1]$.
    """
    q = jnp.asarray(q_z_eq_zero)
    if q.ndim != 1:
        raise ValueError(f"q_z_eq_zero must be 1-D; got shape {q.shape}")
    if bool(jnp.any((q < 0.0) | (q > 1.0))):
        raise ValueError("q_z_eq_zero must lie in [0, 1].")
    return q


def local_idr_error_scale(
    chain_idrs: Float[Array, "n_chains n"],
) -> float:
    """Estimate $\\delta_n$ — the local-idr error scale across split chains.

    Per Theorem 3.3 of the revised report, the Bayes-mFDR rate is $\\alpha + O(\\delta_n)$
    where $\\delta_n$ is a *uniform or threshold-local* error bound on the local idr.
    The split-chain estimator below is the operational analogue: split the post-warmup
    samples into chain groups, compute the per-feature idr on each group, return the
    largest absolute deviation across groups.

    Parameters
    ----------
    chain_idrs
        2-D array indexed as ``(chain, feature)``; each row is the local idr computed
        on a different chain (or the two halves of a single long chain).

    Returns
    -------
    $\\delta_n$ — the maximum across features of $\\max_g \\widehat\\idr^{(g)}_i -
    \\min_g \\widehat\\idr^{(g)}_i$.

    Notes
    -----
    This is operational, not a formal certificate of the Theorem 3.3 hypothesis;
    a small $\\delta_n$ is *necessary* for the rate but not sufficient. We use it
    as a regression signal: $\\delta_n$ should shrink as the chain length grows
    on a fixed problem.
    """
    arr = jnp.asarray(chain_idrs)
    if arr.ndim != 2:
        raise ValueError(f"chain_idrs must be (n_chains, n); got shape {arr.shape}")
    if arr.shape[0] < 2:
        raise ValueError("Need at least 2 chains/folds to estimate delta_n.")
    spread = jnp.max(arr, axis=0) - jnp.min(arr, axis=0)
    return float(jnp.max(spread))
