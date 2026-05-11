"""Marginal model: pilot CDF + Bernstein-Dirichlet correction.

The chain consumes data on the copula scale $u_i \\in [0, 1]^K$. To get
there from raw observed scores $X$ we apply a per-replicate CDF estimate
$F_j$ to each column. PY-IDR has two layers:

1. **Pilot CDF** — a fixed initial estimate. The default is the
   empirical rank transform
   ($\\hat F_n(x) = \\#\\{i : X_i \\le x\\} / (n + 1)$ with mid-rank ties),
   exposed by :mod:`py_idr.marginals.transform`.
2. **Bernstein-Dirichlet correction** — a learned nonparametric tweak
   on top of the pilot. The CDF is parameterised as a Bernstein
   polynomial $F_j(u) = \\sum_r w_{j, r} B_{r, M}(u)$ with weights on
   the simplex; the conjugate update samples
   $w_j \\mid \\text{rest} \\sim \\mathrm{Dir}(\\alpha_F + n_j)$.

Submodule map:

- :mod:`py_idr.marginals.transform` — empirical rank transform and the
  general ``cdf_pilot`` helper.
- :mod:`py_idr.marginals.bernstein` — basis evaluation, ``cdf``,
  ``pdf``, ``cumulative_basis``, ``density_basis``.
- :mod:`py_idr.marginals.dirichlet_weights` — the conjugate update
  (``sample_latent_S`` + ``update_weights``) used by Algorithm 1 step 5.
- :mod:`py_idr.marginals.pilot` — glue between pilot CDF and the
  Bernstein-weighted mixture.
- :mod:`py_idr.marginals.degree_prior` — truncated geometric prior on
  the Bernstein degree $M$; step 6 of the sweep updates it.

See plans/03_marginals.md and §3.3.1 of the report.
"""
