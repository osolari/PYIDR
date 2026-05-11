"""Inference back-ends for PY-IDR.

Two back-ends are planned; only :mod:`py_idr.inference.mcmc` is
implemented today.

- :mod:`py_idr.inference.mcmc` — Markov chain Monte Carlo. Implements
  Algorithm 1 from §3 of the report: an 8-step Gibbs sweep with
  NumPyro NUTS for atom parameters and PY hyperparameters, plus an
  independence-Metropolis step for the copula family. Two chain
  drivers:

  - ``run_chain_simple`` / ``run_multi_chain_simple`` for the
    single-cluster ($T = 1$) fast path — the right model for S1-S4.
  - ``run_chain_multi`` for the full multi-cluster path — required
    for S5 and for any real data where the reproducible component is
    not a single Gaussian copula.

- :mod:`py_idr.inference.vi` — variational inference (planned). Will
  expose the same `run_chain_*` shape but with a mean-field normalising
  flow surrogate for the latent partition. Plan 09 covers the design.

See plans/04_inference_mcmc.md and plans/05_inference_vi.md.
"""
