# PY-IDR

A reference implementation of **PY-IDR** — a hierarchical nonparametric Bayesian
extension of the Irreproducible Discovery Rate (IDR) framework. The reproducible
component is modelled as a Pitman–Yor mixture of Gaussian copulas with structured
correlation, augmented with $t_5$, Clayton, and Gumbel atoms; per-replicate marginals
follow Bernstein–Dirichlet priors. Inference is by **Pólya-urn auxiliary-atom MCMC**
(Neal-style auxiliary atoms drawn from $P_0$ each sweep + NUTS or Metropolis on a
valid correlation parameterization), with a finite-truncation structured mean-field
variational alternative for genomic-scale data.

The accompanying paper is in [`docs/report/`](docs/report/); a compiled PDF is
produced by `pdflatex docs/report/main.tex`.

## Quickstart

```bash
bash scripts/create_env.sh
source .venv/bin/activate
pytest -m smoke
```

## Where to look

- [`IMPLEMENTATION_PLAN.md`](IMPLEMENTATION_PLAN.md) — top-level engineering plan.
- [`plans/`](plans/) — per-workstream detailed plans (skeleton, algebra+copulas,
  marginals, MCMC, VI, decision theory, simulations, real data, figures+tables, theory tests).
- [`.claude/skills/`](.claude/skills/) — recurring engineering workflows for Claude.
- [`docs/report/`](docs/report/) — the SAIM-style preregistration paper.

## Status

Skeleton stage. Algorithmic submodules raise `NotImplementedError` and reference the
sub-plan that owns them. The `py_idr.doctor()` probe is green on a fresh env.

## License

Apache-2.0.
