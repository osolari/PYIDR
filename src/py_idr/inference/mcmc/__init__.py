"""MCMC back-end for PY-IDR.

The 8-step Gibbs sweep of Algorithm 1 is split across several
submodules so each step can be unit-tested in isolation:

- :mod:`py_idr.inference.mcmc.class_assign` — step 1: resample $Z_i$
  given the current copula structure.
- :mod:`py_idr.inference.mcmc.sweep_multi` — step 2 onwards in the
  multi-cluster (Pitman-Yor) path. Calls into ``polya_urn``,
  ``atom_dispatch``, and ``type_update`` for the cluster mechanics.
- :mod:`py_idr.inference.mcmc.sweep_simple` — fast path: single
  Gaussian-copula cluster, no Pólya-urn step, no type update.
- :mod:`py_idr.inference.mcmc.nuts_atoms` — NumPyro NUTS updates for
  the cluster atom parameters (Gaussian Cholesky, Clayton $\\theta$,
  Gumbel-K2 $\\theta$, Student-$t_5$ Cholesky).
- :mod:`py_idr.inference.mcmc.atom_dispatch` — dispatcher: routes each
  cluster atom to the right NUTS update by family.
- :mod:`py_idr.inference.mcmc.type_update` — step 4: independence-
  Metropolis proposal across copula families. Excludes Gumbel from
  the proposal pool for $K \\ge 3$ (multivariate Hofert is future).
- :mod:`py_idr.inference.mcmc.py_hyperparam_update` — step 7: NUTS on
  the Pitman-Yor $(\\alpha, \\sigma)$ EPPF.
- :mod:`py_idr.inference.mcmc.pi_update` — step 8: conjugate
  Beta-Bernoulli update for the reproducible mass $\\pi$.
- :mod:`py_idr.inference.mcmc.run_chain` — top-level chain driver;
  glues the sweep steps into multi-chain runs and packs the output
  into ``arviz.InferenceData`` for downstream diagnostics.

See plans/04_inference_mcmc.md and §3.3 of the report.
"""
