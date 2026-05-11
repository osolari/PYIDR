"""Algebraic helpers used by the Gaussian-copula and PLOD machinery.

Two small modules:

- :mod:`py_idr.algebra.correlation` — correlation-matrix
  reparameterisations: Cholesky $\\leftrightarrow$ correlation,
  unit-row-norm normalisation, exchangeable-correlation construction.
  These are the helpers that the multi-cluster NUTS sampler uses to
  keep the Gaussian-copula atom in a valid space.
- :mod:`py_idr.algebra.plod` — positive-lower-orthant-dependence
  checks. PLOD is the constraint "every pairwise off-diagonal of $R$
  is positive," which is the right structural assumption for the
  reproducible component. The module exposes ``holds``,
  ``assert_plod``, and ``plod_with_positive_pairwise`` as the
  predicates used by the type-update and atom-NUTS rejection logic.

See plans/02_algebra_and_copulas.md and §3.2.2 of the report.
"""
