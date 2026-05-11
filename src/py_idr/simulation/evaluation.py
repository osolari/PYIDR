"""Realized FDR and power computation against simulation ground truth.

When the simulation's ``true_Z`` is known, we can evaluate the local-idr
ranking + Sun--Cai threshold against the actual class labels:

- **Realized FDR** at threshold $\\gamma$:
  $\\#\\{i \\in R(\\gamma) : Z_i^* = 0\\} \\,/\\, \\#R(\\gamma)$ (or 0 if no rejections).
- **Power** at threshold $\\gamma$:
  $\\#\\{i \\in R(\\gamma) : Z_i^* = 1\\} \\,/\\, \\#\\{i : Z_i^* = 1\\}$.

These per-replicate scalars are the long-format columns consumed by figure F1
(idr-vs-realized-FDR calibration) and table T4 (sim FDR / power).

See plans/06_decision_theory.md and plans/07_simulation_studies.md.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np
from jaxtyping import Array, Float, Integer

from py_idr.decision.sun_cai import step_up_threshold


@dataclass(frozen=True)
class RecoveryResult:
    """Per-replicate evaluation outputs at a single Sun--Cai operating level.

    Attributes
    ----------
    alpha
        Operating level passed to Sun--Cai.
    k_alpha
        Number of rejections selected. ``0`` when the rejection set is empty.
    realized_fdr
        Realised false-discovery rate $V/R$; ``0.0`` when $R = 0$.
    power
        True-positive rate $\\mathrm{TP} / \\#\\{Z^* = 1\\}$; ``0.0`` when the
        reproducible class is empty (degenerate simulation).
    """

    alpha: float
    k_alpha: int
    realized_fdr: float
    power: float


def evaluate_recovery(
    idr: Float[Array, "n"],
    true_Z: Integer[Array, "n"],
    alpha: float = 0.05,
) -> RecoveryResult:
    """Run the Sun--Cai step-up at level $\\alpha$; compute realised FDR + power.

    Parameters
    ----------
    idr
        Per-feature posterior local idr from the chain
        (:func:`py_idr.decision.local_idr.from_mcmc`).
    true_Z
        Ground-truth class indicators in $\\{0, 1\\}$.
    alpha
        Operating level for the Sun--Cai threshold rule (Eq. 3.11).

    Returns
    -------
    A :class:`RecoveryResult` with the per-replicate scalars.
    """
    idr_arr = np.asarray(idr)
    true_arr = np.asarray(true_Z, dtype=np.int32)
    if idr_arr.shape != true_arr.shape:
        raise ValueError(
            f"idr and true_Z must share shape; got {idr_arr.shape} vs {true_arr.shape}"
        )
    if idr_arr.size == 0:
        raise ValueError("idr must be non-empty.")

    result = step_up_threshold(jnp.asarray(idr_arr), alpha)
    selected = np.asarray(result.selected)

    n_true_positives = int(np.sum(true_arr == 1))
    if result.k_alpha == 0:
        return RecoveryResult(alpha=alpha, k_alpha=0, realized_fdr=0.0, power=0.0)

    false_discoveries = int(np.sum(true_arr[selected] == 0))
    true_discoveries = int(np.sum(true_arr[selected] == 1))
    realized_fdr = false_discoveries / result.k_alpha
    power = true_discoveries / n_true_positives if n_true_positives > 0 else 0.0

    return RecoveryResult(
        alpha=alpha,
        k_alpha=int(result.k_alpha),
        realized_fdr=float(realized_fdr),
        power=float(power),
    )
