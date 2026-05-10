"""Sun--Cai (2007) plug-in step-up cutoff for local idr.

Implements the operational step-up rule of Eq. (3.11) of the revised report.
Sort the per-feature local idr ascending,
$\\widehat\\idr_{(1)} \\le \\widehat\\idr_{(2)} \\le \\cdots \\le \\widehat\\idr_{(n)}$,
and choose

.. math::
   \\widehat k_\\alpha = \\max\\!\\left\\{k \\geq 1 :
       \\frac{1}{k}\\sum_{j=1}^k \\widehat\\idr_{(j)} \\le \\alpha\\right\\}.

When the set is empty (i.e., even the smallest single idr exceeds $\\alpha$), no
features are rejected. The :class:`StepUpResult` dataclass below makes this
empty-rejection branch explicit so the caller never sees a silent NaN.

The running average is monotone non-decreasing in $k$ (each new sorted value is at
least the prior maximum), so the rule selects the longest prefix of the sorted
idr sequence whose mean is at most $\\alpha$.

See plans/06_decision_theory.md.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
from jaxtyping import Array, Float, Integer


@dataclass(frozen=True)
class StepUpResult:
    """Outcome of one Sun--Cai step-up application.

    Attributes
    ----------
    k_alpha
        Number of rejections; **0 if the rule's set is empty**.
    gamma_alpha
        The largest threshold satisfying Eq. (3.11) — concretely,
        $\\widehat\\idr_{(\\widehat k_\\alpha)}$. Symbolically $-\\infty$ when
        $\\widehat k_\\alpha = 0$; numerically we report ``nan``.
    selected
        Indices into the original (unsorted) ``idr`` vector of the rejected features;
        an empty array when $\\widehat k_\\alpha = 0$.
    """

    k_alpha: int
    gamma_alpha: float
    selected: Integer[Array, "k"]


def step_up_threshold(
    idr: Float[Array, "n"],
    alpha: float,
) -> StepUpResult:
    """Apply the Sun--Cai step-up rule at level $\\alpha$ to a vector of local idrs.

    Parameters
    ----------
    idr
        Vector of per-feature posterior local idrs $\\widehat\\idr_i \\in [0, 1]$.
    alpha
        Operating level; must satisfy $0 < \\alpha < 1$.

    Returns
    -------
    A :class:`StepUpResult`. The ``selected`` indices are returned in ascending order
    of the original feature index (not in sorted-idr order).

    Notes
    -----
    The running mean is monotone non-decreasing in $k$; we exploit this to find
    $\\widehat k_\\alpha$ in one sort + one cumulative pass ($O(n \\log n)$).
    Empty rejection set is the only branch that does not raise; all malformed
    inputs (NaN, length 0, ``alpha`` outside $(0, 1)$) raise ``ValueError``.
    """
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"alpha must be in (0, 1); got {alpha}")
    idr = jnp.asarray(idr)
    if idr.ndim != 1:
        raise ValueError(f"idr must be 1-dimensional; got shape {idr.shape}")
    n = int(idr.shape[0])
    if n == 0:
        raise ValueError("idr is empty")
    if bool(jnp.any(jnp.isnan(idr))):
        raise ValueError("idr contains NaN values")

    sorted_idx = jnp.argsort(idr)
    sorted_idr = idr[sorted_idx]
    running_mean = jnp.cumsum(sorted_idr) / jnp.arange(1, n + 1)
    # Largest k with running_mean[k-1] <= alpha. Because running_mean is monotone
    # non-decreasing, the qualifying set is a prefix of the index range.
    qualifies = running_mean <= alpha
    if not bool(qualifies[0]):
        return StepUpResult(
            k_alpha=0, gamma_alpha=float("nan"), selected=jnp.array([], dtype=jnp.int32)
        )
    # k_alpha is the count of qualifying entries.
    k_alpha = int(jnp.sum(qualifies))
    gamma_alpha = float(sorted_idr[k_alpha - 1])
    selected = jnp.sort(sorted_idx[:k_alpha]).astype(jnp.int32)
    return StepUpResult(k_alpha=k_alpha, gamma_alpha=gamma_alpha, selected=selected)


def discoveries(
    idr: Float[Array, "n"],
    gamma: float,
) -> Integer[Array, "k"]:
    """Return indices of features with $\\widehat\\idr_i < \\gamma$, in ascending index order.

    Convenience wrapper for callers that already have a chosen threshold.
    """
    idr = jnp.asarray(idr)
    return jnp.where(idr < gamma)[0].astype(jnp.int32)
