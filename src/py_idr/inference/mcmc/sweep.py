"""One full Algorithm 1 sweep of the Pólya-urn auxiliary-atom MCMC for PY-IDR.

Sequence (per Algorithm 1 of the revised report):
    1. Class assignment      Z_i ~ Bernoulli(π̂_i)                 (Eq. 3.4)
    2. Cluster reassignment  PY Pólya-urn with H aux atoms drawn from P_0,
                              using Eqs. (3.5)–(3.6) over (n_{m,-i}, T_{-i})
    3. Atom-parameter update NUTS or Metropolis on a valid unconstrained
                              parameterization; covariance factors normalised to
                              correlation matrices via cov_to_corr; PLOD rejection
    4. Atomic-type update    IM proposal over {Gauss, t5, Clayton, Gumbel}
    5. Marginal update       conjugate Dirichlet given latent S
    6. Marginal-degree update  MH-within-Gibbs on (M_j, α_F)
    7. PY hyperparameters    auxiliary-variable / slice updates on (α, σ)
    8. Mixing proportion     π ~ Beta(1 + n_1, 1 + n_0)

See plans/04_inference_mcmc.md.
"""

from __future__ import annotations


def sweep(state, key):  # type: ignore[no-untyped-def]  # noqa: ANN001
    """Run one full Algorithm 1 sweep; return the new (state, key)."""
    raise NotImplementedError("W2 — see plans/04_inference_mcmc.md")
