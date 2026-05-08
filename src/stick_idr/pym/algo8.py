"""Neal (2000) Algorithm 8 cluster reassignment, extended to Pitman-Yor.

H auxiliary atoms drawn from P_0; the predictive uses the PY (Pitman 1996) form:
    P(z_i = m) ∝ (n_m - σ) c_θ_m(U_i)         for occupied clusters
    P(z_i = T+h) ∝ ((α + Tσ) / H) c_θ_{T+h}(U_i)   for auxiliary atoms.

See plans/04_inference_mcmc.md.
"""

from __future__ import annotations


def step(state, key, H: int = 5):  # type: ignore[no-untyped-def]  # noqa: ANN001, N803
    """One Algorithm 8 reassignment for a single feature; functional state update."""
    raise NotImplementedError("W2 — see plans/04_inference_mcmc.md")
