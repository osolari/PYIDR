"""Variational inference back-end for PY-IDR (planned).

The VI back-end is on the roadmap (plan 09). Design sketch:

- **Surrogate.** Mean-field normalising flow over the latent partition
  $\\{Z_i\\}$ + per-cluster atom parameters. The Bernstein-Dirichlet
  marginals retain their conjugate update — VI only replaces the
  partition + atom layers.
- **Objective.** ELBO with a temperature-annealing schedule. The
  Pitman-Yor stick-breaking prior is handled via a continuous
  relaxation through the Beta variates.
- **API.** ``run_chain_vi(F0, M, ...)`` returns the same
  ``ChainResult`` / ``MultiChainResult`` shape as the MCMC drivers, so
  downstream decision and diagnostic code is unchanged.

Not yet implemented. The current chain code is MCMC only — see
:mod:`py_idr.inference.mcmc`.
"""
