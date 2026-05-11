"""Copula families used by PY-IDR's reproducible component.

PY-IDR represents the reproducible component as a Pitman-Yor mixture over
copula clusters. Each cluster carries one copula family and its
parameters; the chain re-samples the family via an
independence-Metropolis step (see
:mod:`py_idr.inference.mcmc.type_update`).

Available families
------------------

- :class:`py_idr.copulas.gaussian.GaussianCopula` — $K$-variate Gaussian
  copula in the Cholesky parameterisation $R = L L^\\top$.
- :class:`py_idr.copulas.clayton.ClaytonCopula` — Archimedean copula
  with lower-tail dependence.
- :class:`py_idr.copulas.gumbel.GumbelCopula` — Archimedean copula with
  upper-tail dependence. Currently restricted to $K = 2$; the
  multivariate Hofert recursion lands in a later milestone.
- :class:`py_idr.copulas.student_t.StudentTCopula` — elliptical copula
  with heavier joint-tail mass than the Gaussian copula.
- :class:`py_idr.copulas.independence.IndependenceCopula` — the
  irreproducible component; density is uniformly 1 on $[0, 1]^K$.

All families derive from :class:`py_idr.copulas.base.Copula` and expose
``log_density(u)``, ``cdf_diagonal(u)``, and a draw helper. The
:mod:`py_idr.copulas.registry` module ties the family labels to their
constructors for the type-update step.

See plans/02_algebra_and_copulas.md and §3.2 of the report.
"""
