"""Sun-Cai (2007) plug-in step-up cutoff for local idr.

Largest γ such that the running average of the sorted local idrs is ≤ α at the matching k.

See plans/06_decision_theory.md.
"""

from __future__ import annotations


def step_up_threshold(idr, alpha: float):  # type: ignore[no-untyped-def]  # noqa: ANN001
    """Return γ_α — the Sun-Cai threshold at operating level α."""
    raise NotImplementedError("W3 — see plans/06_decision_theory.md")


def discoveries(idr, gamma: float):  # type: ignore[no-untyped-def]  # noqa: ANN001
    """Return the indices of features with idr_i < γ."""
    raise NotImplementedError("W3 — see plans/06_decision_theory.md")
