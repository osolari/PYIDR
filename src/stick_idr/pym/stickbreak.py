"""Pitman-Yor stick-breaking (GEM) sampler.

(w_1, w_2, ...) ~ GEM(α, σ); v_m ~ Beta(1 - σ, α + m σ); w_m = v_m * prod_{k<m}(1 - v_k).

See plans/04_inference_mcmc.md.
"""

from __future__ import annotations


def gem_sample(key, alpha: float, sigma: float, T: int):  # type: ignore[no-untyped-def]  # noqa: ANN001
    """Return the truncated stick-breaking weights w_{1:T} drawn from GEM(α, σ)."""
    raise NotImplementedError("W2 — see plans/04_inference_mcmc.md")
