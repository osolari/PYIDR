"""One full Algorithm 1 sweep of the slice-sampled Pólya-urn MCMC for PY-IDR.

Sequence (per the report's Algorithm 1):
    1. Class assignment      Z_i ~ Bernoulli(π̂_i)
    2. Cluster reassignment  Algorithm 8 with H aux atoms
    3. Atom-parameter update NUTS on Cholesky parameterization
    4. Atomic-type update    IM proposal over {Gauss, t5, Clayton, Gumbel}
    5. Marginal-weight update  conjugate Dirichlet given latent S
    6. Marginal-degree update  MH-within-Gibbs on (M_j, α_F)
    7. PY hyperparameters    Escobar-West for α; slice for σ
    8. Mixing proportion     π ~ Beta(1 + n_1, 1 + n_0)

See plans/04_inference_mcmc.md.
"""

from __future__ import annotations


def sweep(state, key):  # type: ignore[no-untyped-def]  # noqa: ANN001
    """Run one full Algorithm 1 sweep; return the new (state, key)."""
    raise NotImplementedError("W2 — see plans/04_inference_mcmc.md")
