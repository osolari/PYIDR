# StickIDR — Implementation Plan

Engineering plan for turning the **PY-IDR** report in [docs/report/](docs/report/) into a
professional, reproducible research codebase. Reference: the model and inference scheme
are in [3.method.tex](docs/report/sections/3.method.tex), the empirical protocol is
[4.experiments.tex](docs/report/sections/4.experiments.tex), the supporting lemmas
are [6.appendix.tex](docs/report/sections/6.appendix.tex), and the illustrative figure
generator is [generate_figures.py](docs/report/figures/generate_figures.py).

> **PY-IDR** = a fully Bayesian, hierarchical nonparametric extension of the Irreproducible
> Discovery Rate (IDR) framework of Li, Brown, Huang & Bickel (2011). The reproducible-
> component copula is a Pitman–Yor mixture of Gaussian copulas with structured correlation,
> augmented with $t_5$, Clayton, and Gumbel atoms; per-replicate marginals are independent
> Bernstein–Dirichlet. Inference is by slice-sampled Pólya-urn MCMC (Neal Algorithm 8) with
> NUTS on the unconstrained Cholesky parameterization, plus a structured mean-field
> variational alternative with stochastic natural gradients.

The plan has three layers:

1. **Repo skeleton & tooling** — directory layout, JAX/NumPyro env, lint, types, precommit, CI/CD, release.
2. **Package architecture** — `stick_idr` library mapped one-to-one onto the report's components: copula atoms, PY mixture, Bernstein–Dirichlet marginals, Algorithm 8 + NUTS, structured-MF VI, and the Sun–Cai-type plug-in step-up rule.
3. **Deliverables** — docs (MkDocs), `.ipynb` tutorials, figure/table reproducer scripts, tests, logging, and the W1–W6 workstreams below.

Every artifact in the repo is traceable to either a theorem (Thm 3.1 identifiability,
Thm 3.2 contraction, Thm 3.3 Bayes-mFDR control, Thm 3.4 ergodicity, Thm 3.5 VI consistency),
a simulation regime (S1–S5), a real dataset (D1–D5), a figure (F1–F6), or a table (T1–T7).
If an artifact lacks such a link, it does not belong.

---

## 0. Principles

- **One claim → one reproducer.** Every figure / table / theorem of the report has a single entrypoint that produces it from a config and (where needed) a sampler trace.
- **Correctness first, speed second.** A pure-Python reference implementation of every kernel (copula densities, Algorithm 8 sweep, Sun–Cai plug-in, Bayes-mFDR) is the ground truth; JAX-jitted paths must agree to within `1e-6` in float64.
- **Identifiability is enforced, not assumed.** The PLOD constraint of equation (3.4) is a hard runtime check on every newly-instantiated atom; a violation is a hard test failure.
- **Preregistration discipline.** Hypothesis thresholds and Monte-Carlo replication counts live in a version-controlled YAML, consumed by the reporting script. Flipping them after measurement requires an explicit PR.
- **Sampler diagnostics are non-negotiable.** Split-$\widehat R \le 1.01$ and ESS per parameter $\ge 400$ are required before any posterior summary is reported.
- **No dead code.** If something is not used by a paper figure, a tutorial, or a test, it does not ship.

---

## 1. Repository Layout

```
StickIDR/
├── README.md                         # Top-level story, quickstart, links out
├── IMPLEMENTATION_PLAN.md            # This document
├── plans/                            # Detailed per-workstream sub-plans
│   ├── 01_repo_skeleton.md
│   ├── 02_algebra_and_copulas.md
│   ├── 03_marginals.md
│   ├── 04_inference_mcmc.md
│   ├── 05_inference_vi.md
│   ├── 06_decision_theory.md
│   ├── 07_simulation_studies.md
│   ├── 08_real_data_pipelines.md
│   ├── 09_figures_and_tables.md
│   └── 10_theory_tests.md
├── LICENSE                           # Apache-2.0 (default)
├── CITATION.cff
├── CODE_OF_CONDUCT.md
├── CONTRIBUTING.md
├── CHANGELOG.md
├── pyproject.toml
├── constraints.txt                   # pip-compile output for CI reproducibility
├── uv.lock                           # opt-in; ENV_MANAGER=uv path
├── .python-version
├── .gitattributes                    # LFS for large fixtures if needed
├── .gitignore
├── .editorconfig
├── .pre-commit-config.yaml
├── .github/
│   ├── workflows/                    # See §5
│   ├── ISSUE_TEMPLATE/
│   ├── PULL_REQUEST_TEMPLATE.md
│   ├── CODEOWNERS
│   └── dependabot.yml
├── .claude/
│   └── skills/                       # See §16
│
├── scripts/
│   ├── create_env.sh                 # JAX-aware env bootstrap (CPU/CUDA/Metal)
│   ├── download_data.sh              # ENCODE / DREAM5 / Tabula Muris / Pan-UKBB stubs
│   ├── verify_checksums.sh           # SHA-256 of canonical splits
│   ├── run_simulations.sh            # S1–S5 sweep launcher
│   ├── run_real_data.sh              # D1–D5 launcher
│   ├── make_all_figures.sh           # F1–F6 from artifacts/*.csv
│   ├── make_all_tables.sh            # T1–T7 from artifacts/*.csv
│   └── benchmark_runtime.sh          # F6 runtime / ESS-per-second harness
│
├── src/stick_idr/                    # The package — see §3
│   └── ...
│
├── configs/                          # Hydra/OmegaConf configs
│   ├── model/                        # py_idr.yaml, dp_idr.yaml, vanilla_idr.yaml, ...
│   ├── data/                         # encode_ctcf.yaml, encode_atac.yaml, dream5.yaml, ...
│   ├── inference/                    # mcmc_default.yaml, vi_default.yaml, init_em.yaml
│   ├── hparams/                      # priors and tuning knobs
│   ├── simulation/                   # s1.yaml..s5.yaml, K_grid.yaml, n_grid.yaml
│   └── preregistration/
│       └── hypotheses.yaml           # Thresholds for FDR calibration, contraction, etc.
│
├── tests/
│   ├── unit/
│   ├── property/                     # hypothesis-driven
│   ├── numerical/                    # rate / contraction sanity
│   ├── inference/                    # Algorithm 8 + NUTS correctness
│   ├── identifiability/              # PLOD + Yakowitz–Spragins regression
│   ├── decision/                     # Sun–Cai plug-in calibration
│   ├── integration/                  # end-to-end on tiny S1 cell
│   ├── regression/                   # golden-file outputs at fixed seeds
│   └── conftest.py
│
├── benchmarks/                       # airspeed-velocity (asv) suite
│   ├── asv.conf.json
│   └── benchmarks/
│
├── docs/
│   ├── report/                       # Existing LaTeX source (unchanged)
│   ├── mkdocs.yml
│   ├── index.md
│   ├── getting_started/
│   ├── math/                         # Sklar, PY mixtures, Bernstein–Dirichlet, mFDR
│   ├── guides/
│   ├── api/                          # mkdocstrings-generated
│   ├── tutorials/                    # Rendered .ipynb
│   ├── experiments/                  # Per-dataset reproducer pages
│   └── developer/
│
├── notebooks/                        # Source .ipynb (paired with .py via jupytext)
│   ├── 00_idr_recap.ipynb
│   ├── 01_gaussian_copula_density.ipynb
│   ├── 02_py_stickbreak.ipynb
│   ├── 03_bernstein_marginals.ipynb
│   ├── 04_algorithm8_step_by_step.ipynb
│   ├── 05_nuts_on_cholesky.ipynb
│   ├── 06_sun_cai_plugin.ipynb
│   ├── 07_simulation_S1_walkthrough.ipynb
│   ├── 08_encode_ctcf_demo.ipynb
│   └── 09_preregistration_vs_results.ipynb
│
├── experiments/                      # Reproducers; sbatch / argparse glue
│   ├── sim_S1/
│   ├── sim_S2/
│   ├── sim_S3/
│   ├── sim_S4/
│   ├── sim_S5/
│   ├── encode_ctcf/
│   ├── encode_atac/
│   ├── dream5/
│   ├── tabula_muris/
│   ├── pan_ukbb/
│   └── common/
│
├── docs/report/figures/              # Rendered PDFs consumed by the paper via \includegraphics
├── docs/report/tables/               # Rendered .tex table fragments consumed via \input
│
├── data/                             # .gitignore'd; populated by scripts/download_data.sh
├── logs/                             # .gitignore'd
├── checkpoints/                      # .gitignore'd; sampler traces published via releases
└── artifacts/                        # .gitignore'd; CSV / JSON consumed by figure scripts
```

Rationale for non-obvious choices:

- **`src/` layout** so imports do not pick up partially-installed code during tests.
- **JAX-only stack.** NumPyro is the inference vehicle for NUTS; the Pólya-urn / Algorithm 8 cluster reassignment is a custom JAX-jit-friendly Gibbs step. This avoids the discrete-cluster-update gap in pure Stan and keeps everything autodifferentiable end-to-end.
- **Notebooks paired with `.py`** via [jupytext](https://jupytext.readthedocs.io/) so precommit/CI diff the `.py` copies; the `.ipynb` are the render target.
- **`experiments/` separate from `src/`** because experiment glue has different review standards than library code.
- **`configs/preregistration/hypotheses.yaml`** is the single source of truth for FDR-calibration / contraction-rate / coverage thresholds; the ledger script generates §4 of the companion paper from it.

---

## 2. Environment & Build

### `scripts/create_env.sh`

Idempotent, OS-aware bootstrap. Detects accelerator (CPU / CUDA / Apple Metal), installs the matched JAX wheel, sets up pre-commit. Pseudocode:

```bash
#!/usr/bin/env bash
# scripts/create_env.sh — bootstrap dev env for StickIDR
set -euo pipefail
IFS=$'\n\t'

PY_VER="${PY_VER:-3.11}"
JAX_BACKEND="${JAX_BACKEND:-auto}"      # auto | cpu | cuda12 | metal
ENV_MANAGER="${ENV_MANAGER:-venv}"      # venv | uv
INSTALL_EXTRAS="${INSTALL_EXTRAS:-dev,docs}"

1. detect platform (darwin/linux), arch (x86_64/arm64), CUDA presence
2. python -m venv .venv  (or  uv venv --python ${PY_VER}  when ENV_MANAGER=uv)
3. pip install -e ".[${INSTALL_EXTRAS}]"
4. on CUDA hosts: pip install -U "jax[cuda12]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
   on Apple Silicon: pip install jax-metal
5. pre-commit install --install-hooks
6. python -c "import stick_idr; stick_idr.doctor()"   # sanity (§3.10)
7. print next-step hints: source .venv/bin/activate && pytest -m smoke
```

The script is the canonical install path; CI calls it with `INSTALL_EXTRAS=dev`.

### `pyproject.toml` skeleton

- Build backend: `hatchling`.
- **Runtime deps:** `jax>=0.4.30`, `jaxlib`, `numpyro>=0.15`, `numpy`, `scipy`, `pandas`, `pyarrow`, `hydra-core`, `omegaconf`, `tqdm`, `rich`, `structlog`, `arviz`, `pydantic>=2`.
- **Extras:**
  - `cuda`: `jax[cuda12]` selector.
  - `metal`: `jax-metal`.
  - `data`: `pyranges`, `h5py`, `hail` *(optional, Pan-UKBB)*, `anndata`, `scanpy` *(optional, Tabula Muris)*, `requests`.
  - `dev`: `pytest`, `pytest-xdist`, `pytest-benchmark`, `pytest-cov`, `hypothesis`, `coverage[toml]`, `ruff`, `black`, `mypy`, `pre-commit`, `jupytext`, `nbmake`, `pip-tools`.
  - `docs`: `mkdocs-material`, `mkdocstrings[python]`, `mkdocs-jupyter`, `mkdocs-gen-files`, `mkdocs-literate-nav`, `mkdocs-macros-plugin`.
  - `viz`: `matplotlib`, `seaborn`.
  - `logging`: `wandb`, `tensorboard`, `aim` (default chosen by Hydra config).
- **Tool configs co-located:** `[tool.ruff]`, `[tool.black]`, `[tool.mypy]` (strict on `src/`, permissive on `experiments/`), `[tool.pytest.ini_options]`, `[tool.coverage.run]`.

### Lockfile

Default dev path resolves from `pyproject.toml` at install time. For reproducible CI we maintain a `constraints.txt` via `pip-compile` (pip-tools) that pins the transitive graph used on the CI image. A `uv.lock` is opt-in for contributors who prefer uv. Renovate/Dependabot keeps both fresh.

---

## 3. Package Architecture (`src/stick_idr/`)

The package mirrors §3 of the report module-for-module so a reader going from section to module has no translation to do.

```
src/stick_idr/
├── __init__.py                       # Public API surface; version; doctor()
├── _version.py                       # __version__ — bumped by release CI
│
├── algebra/
│   ├── __init__.py
│   ├── correlation.py                # Cholesky <-> R; LKJ(eta=2) prior; structured classes
│   │                                 # (exchangeable / one-factor / two-factor)
│   ├── plod.py                       # PLOD check (eq. 3.4): C(u,...,u) > u^K probe
│   ├── ranks.py                      # rank → pseudo-uniform U; ties; censoring
│   └── numerics.py                   # log-sum-exp, stable Cholesky updates, jitter
│
├── copulas/
│   ├── __init__.py
│   ├── base.py                       # Copula base class: log_density(u), sample(rng,n)
│   ├── independence.py               # c_0 = 1 (the irreproducible component)
│   ├── gaussian.py                   # Gaussian copula via Cholesky factor
│   ├── student_t.py                  # t_5 copula; ν fixed at 5 per the report
│   ├── clayton.py                    # Archimedean Clayton; θ > 0 (PLOD-compatible)
│   ├── gumbel.py                     # Archimedean Gumbel; θ ≥ 1
│   ├── registry.py                   # ATOMIC_TYPES = {"gauss", "t5", "clayton", "gumbel"}
│   └── proposal.py                   # Independence-Metropolis proposal over types (Algo 8 step 4)
│
├── marginals/
│   ├── __init__.py
│   ├── pilot.py                      # Kernel-smoothed empirical CDF (Eq. 3.7 F_j^(0))
│   ├── bernstein.py                  # Bernstein basis B_r(t; M); cumulative + density
│   ├── dirichlet_weights.py          # Conjugate p_j | latent S update
│   ├── degree_prior.py               # M_j ∈ {5, 10, 20, 40, 80} with weights ∝ 1/M
│   └── transform.py                  # X -> U via fitted F_j (and inverse)
│
├── pym/                              # Pitman–Yor mixture machinery
│   ├── __init__.py
│   ├── stickbreak.py                 # GEM(alpha, sigma) sampler + truncated weights
│   ├── predictive.py                 # PY Blackwell–MacQueen predictive (Pitman 1996)
│   ├── algo8.py                      # Neal (2000) Algorithm 8 with H auxiliary atoms
│   ├── slice.py                      # Slice-sampled PY truncation (Walker 2007 / KGW 2011)
│   ├── alpha_update.py               # Escobar–West (1995) auxiliary-variable α update
│   └── sigma_update.py               # Slice update for σ ∈ [0, 1)
│
├── model/
│   ├── __init__.py
│   ├── prior.py                      # Joint prior π, α, σ, α_F; Hyperprior config struct
│   ├── likelihood.py                 # log p(U | Z, θ, w) at the per-feature level
│   ├── joint.py                      # Full joint log-density (NumPyro model fn)
│   ├── state.py                      # PyTree-friendly SamplerState dataclass
│   └── em_init.py                    # Bivariate EM-IDR initialization (Sec. 3.6)
│
├── inference/
│   ├── __init__.py
│   ├── mcmc/
│   │   ├── __init__.py
│   │   ├── sweep.py                  # One full Algorithm 1 sweep
│   │   ├── nuts_atoms.py             # NUTS on unconstrained Cholesky for atom updates
│   │   ├── class_assign.py           # Z_i ~ Bernoulli(π̂_i)
│   │   ├── reassign.py               # Cluster reassignment (Algo 8 calls)
│   │   ├── type_update.py            # t_m IM proposal (Algo 1 step 4)
│   │   ├── marginal_update.py        # p_j conjugate; (M_j, α_F) MH-within-Gibbs
│   │   ├── pi_update.py              # π ~ Beta(1+n_1, 1+n_0)
│   │   └── diagnostics.py            # split-R̂, ESS, energy, divergences, occupancy traces
│   ├── vi/
│   │   ├── __init__.py
│   │   ├── family.py                 # Structured mean-field q(Z, z, θ, t, w, π, F, α, σ)
│   │   ├── elbo.py                   # ELBO with truncation T
│   │   ├── natgrad.py                # Stochastic natural-gradient steps (HBWP 2013)
│   │   ├── reparam.py                # Reparam tricks for the continuous blocks
│   │   └── minibatch.py              # B ∈ {500, 2000} stochastic data passes
│   └── init.py                       # EM-IDR + k-means seeding for both MCMC and VI
│
├── decision/
│   ├── __init__.py
│   ├── local_idr.py                  # Posterior-averaged P(Z_i=0 | data) (Eq. 3.10)
│   ├── bayes_mfdr.py                 # E[V]/E[R] at threshold γ (Eq. 3.11)
│   ├── sun_cai.py                    # Sun–Cai plug-in step-up cutoff
│   └── calibration.py                # idr-vs-realized-FDR curves
│
├── simulation/
│   ├── __init__.py
│   ├── scenarios.py                  # S1–S5 generators (extends LiBrownHuangBickel 2011)
│   ├── designs.py                    # K ∈ {2,5,10,50} × n ∈ {2k,10k,50k} factorial
│   ├── replicates.py                 # 100-fold MC replication driver
│   └── truth.py                      # Holds true Z, π*, copula for evaluation metrics
│
├── data/
│   ├── __init__.py
│   ├── encode_ctcf.py                # ENCODE 4 CTCF ChIP-seq narrow peaks
│   ├── encode_atac.py                # ENCODE 4 ATAC-seq broad peaks
│   ├── dream5.py                     # DREAM5 algorithm rank lists, K=35
│   ├── tabula_muris.py               # SmartSeq2/10x pseudobulk
│   ├── pan_ukbb.py                   # 6-ancestry GWAS z-scores
│   └── splits.py                     # SHA-256 split registry; verify_checksums hook
│
├── eval/
│   ├── __init__.py
│   ├── hellinger.py                  # h(c_hat, c*) on a fixed grid
│   ├── coverage.py                   # 95% credible-band calibration of local idr
│   ├── ess.py                        # ESS / second; energy-based diagnostics via arviz
│   └── reporting.py                  # Consumes hypotheses.yaml -> verdict JSON
│
├── figures/                          # One script per paper figure (F1–F6)
│   ├── __init__.py
│   ├── f1_idr_calibration.py         # docs/report/figures/fig_idr_calibration.pdf
│   ├── f2_roc_pr.py                  # docs/report/figures/fig_roc_pr.pdf
│   ├── f3_contraction.py             # docs/report/figures/fig_contraction.pdf
│   ├── f4_py_vs_dp.py                # docs/report/figures/fig_py_vs_dp.pdf
│   ├── f5_realdata.py                # docs/report/figures/fig_realdata.pdf
│   ├── f6_runtime.py                 # docs/report/figures/fig_runtime.pdf
│   └── _style.py                     # Shared rcParams + palette (matches generate_figures.py)
│
├── tables/                           # One script per paper table (T1–T7)
│   ├── __init__.py
│   ├── t1_notation.py                # tab:notation (regenerated from a registry)
│   ├── t2_related_work.py            # tab:related-work
│   ├── t3_sim_design.py              # tab:sim-design
│   ├── t4_sim_fdr_power.py           # tab:sim-fdr + tab:sim-power
│   ├── t5_sim_K.py                   # tab:sim-K
│   ├── t6_datasets.py                # tab:datasets
│   ├── t7_real_results.py            # tab:real-disc + tab:real-clusters + tab:real-runtime
│   └── _common.py                    # LaTeX rendering helpers (booktabs, siunitx-safe)
│
├── preregistration/
│   ├── __init__.py
│   ├── schema.py                     # pydantic schema for hypotheses.yaml
│   ├── loader.py                     # parse + validate
│   └── verdict.py                    # measurements -> pass/fail/falsified JSON
│
├── logging_/
│   ├── __init__.py
│   ├── structlog_setup.py
│   ├── experiment.py                 # wandb / tensorboard / aim wiring
│   └── progress.py                   # rich.progress helpers
│
├── utils/
│   ├── seeding.py                    # JAX PRNGKey + numpy + python seeds
│   ├── hashing.py                    # checkpoint + dataset SHA-256 helpers
│   ├── profiling.py                  # jax.profiler + step timings
│   └── typing.py                     # PyTree dataclasses; jaxtyping shapes
│
├── cli.py                            # `stick-idr` entrypoint (fit, eval, sim, figure, table, doctor)
└── doctor.py                         # stick_idr.doctor(): JAX backend, NumPyro version,
                                      # PLOD probe round-trip, Bernstein basis sanity
```

### Public API (surface)

Exported at top level so users can write
`from stick_idr import PYMixture, GaussianCopula, ClaytonCopula, GumbelCopula, StudentTCopula, BernsteinMarginal, fit_mcmc, fit_vi, local_idr, bayes_mfdr, sun_cai_threshold, doctor`.

### Migration from the report

| Report element | Target module |
|---|---|
| Eq. (3.1) latent class & copula representation | `model/likelihood.py` |
| Eq. (3.3) PY mixture for $c_1$ | `pym/stickbreak.py` + `model/joint.py` |
| Eq. (3.4) PLOD constraint | `algebra/plod.py` |
| Eq. (3.7) Bernstein–Dirichlet marginals | `marginals/{bernstein,dirichlet_weights}.py` |
| Algorithm 1 (full sweep) | `inference/mcmc/sweep.py` |
| Step 1 — class assignment Eq. (3.8) | `inference/mcmc/class_assign.py` |
| Step 2 — Algorithm 8 cluster reassignment | `pym/algo8.py` + `inference/mcmc/reassign.py` |
| Step 3 — atom-parameter NUTS on Cholesky | `inference/mcmc/nuts_atoms.py` |
| Step 4 — atomic-type IM proposal | `copulas/proposal.py` + `inference/mcmc/type_update.py` |
| Step 5 — Bernstein-weight conjugate update | `inference/mcmc/marginal_update.py` |
| Step 7 — Escobar–West α update | `pym/alpha_update.py` |
| Step 7 — slice update for σ | `pym/sigma_update.py` |
| Eq. (3.10) local idr | `decision/local_idr.py` |
| Eq. (3.11) Bayes-mFDR | `decision/bayes_mfdr.py` |
| Sun–Cai step-up rule | `decision/sun_cai.py` |
| Variational family of §3.7.2 | `inference/vi/{family,elbo,natgrad}.py` |
| Lemma A.1 (Sklar regularity) | property test in `tests/property/test_sklar_regularity.py` |
| Lemma A.2 (mixture entropy decomposition) | `tests/property/test_mixent_decomposition.py` |
| Lemma A.3 (PY truncation tail bound) | `tests/numerical/test_py_truncation.py` |
| Lemma A.4 (sieve covering numbers) | `tests/numerical/test_sieve_covering.py` |

---

## 4. Pre-commit

`.pre-commit-config.yaml` stages (fast → slow):

1. **Hygiene** — `trailing-whitespace`, `end-of-file-fixer`, `check-yaml`, `check-toml`, `check-merge-conflict`, `mixed-line-ending`, `check-case-conflict`.
2. **Format** — `black`, `ruff format`, `jupytext --sync`, `nbstripout` for committed `.ipynb`.
3. **Lint** — `ruff check` with pydocstyle (numpy convention on public API).
4. **Types** — `mypy` strict on `src/`; permissive on `experiments/`.
5. **Security** — `detect-private-key`, `gitleaks`, `bandit` low-FP profile on `src/`.
6. **LaTeX** — `chktex` (warn-only) on `docs/report/**.tex`.
7. **Project-specific** custom hooks:
   - `scripts/hooks/check_plod.py` — round-trips a small grid of correlation matrices through `stick_idr.algebra.plod.holds()` and asserts the report's positive examples (Gaussian with all-positive ρ, Clayton, Gumbel) pass while the independence copula fails.
   - `scripts/hooks/check_hypotheses_schema.py` — validates `configs/preregistration/hypotheses.yaml` against the pydantic schema.
   - `scripts/hooks/check_figure_backlinks.py` — every `src/stick_idr/figures/*.py` and `tables/*.py` must reference its `fig:*` / `tab:*` LaTeX label in the docstring.

---

## 5. CI/CD (`.github/workflows/`)

Matrix: `{3.11, 3.12} × {ubuntu-latest, macos-latest}`. GPU jobs are gated by the `gpu` label on the PR (self-hosted runner with CUDA-capable JAX).

### Workflows

1. **`ci.yml`** (every push / PR)
   - `lint-type`: ruff, mypy, black --check.
   - `test-cpu`: `pytest -m "not gpu and not slow"` across the matrix, with coverage → Codecov.
   - `test-notebooks`: `pytest --nbmake notebooks/` with `FAST_TUTORIAL=1` (downsized scenarios).
   - `build-docs`: build MkDocs site strict-mode (warnings = errors).
   - `build-latex`: compile `docs/report/main.tex` in a `texlive-full` container; upload PDF artifact.
   - `verify-preregistration`: pydantic schema check + golden-file diff against `hypotheses.yaml`.

2. **`gpu-ci.yml`** (PRs labelled `gpu`, nightly on `main`)
   - `mcmc-correctness`: Algorithm 1 sweep on a tiny S1 cell; recovers $\pi^*$ within MC SE.
   - `nuts-correctness`: NUTS on a fixed correlation matrix has 0 divergences and split-$\widehat R \le 1.01$.
   - `runtime-smoke`: tiny throughput run; reports delta vs. `main` baseline.

3. **`benchmarks.yml`** (nightly on `main`, never on PR)
   - `asv run --bench` over the full suite; publishes to `gh-pages` (`/bench/`).

4. **`release.yml`** (tag push `v*`)
   - Build sdist + wheel.
   - Publish to PyPI (Trusted Publishing).
   - Build and push docs to `gh-pages`.
   - Create GitHub release from `CHANGELOG.md`.
   - Attach paper PDF + hypothesis-verdict JSON.

5. **`security.yml`** (weekly)
   - `pip-audit` / `osv-scanner` on the lockfile.
   - CodeQL analysis.
   - Dependabot PR auto-triage.

6. **`docs.yml`** (`main` push, paths-filtered to docs/notebooks/src docstrings)
   - Build and deploy MkDocs to `gh-pages`. Long-running tutorials are gated by `DOCS_FAST=1`.

7. **`experiment-ledger.yml`** (on every release)
   - Pulls trace hashes and metric CSVs from the release assets, updates `docs/experiments/ledger.md`: run date, git SHA, dataset version, hypothesis verdicts.

### Branch protection

- `main` requires: `ci.yml` green, ≥ 1 review, `CODEOWNERS` ack, linear history.
- Force push disabled on `main`.
- Signed commits encouraged; `release.yml` tagger must sign.

### Reusable actions

- `.github/actions/setup-env/` — composite action wrapping `scripts/create_env.sh` with GitHub cache (`actions/cache` on the `.venv/` and HF/ENCODE caches).

---

## 6. Tests

Organised by what they guarantee. All tests run under `pytest` with markers (`unit`, `property`, `numerical`, `inference`, `identifiability`, `decision`, `integration`, `gpu`, `slow`, `smoke`).

### 6.1 Unit (`tests/unit/`)
- `test_correlation.py` — Cholesky round-trip; LKJ(η=2) sampling positive-definite; structured classes (exchangeable / 1-factor / 2-factor) yield unit diagonal.
- `test_copula_logpdfs.py` — Gaussian / $t_5$ / Clayton / Gumbel log-density matches the canonical SciPy / closed-form expressions on a fixed grid.
- `test_bernstein_basis.py` — `B_r(t;M)` cumulative basis equals `betainc(r, M-r+1, t)`; partition of unity at every `t`.
- `test_dirichlet_conjugate.py` — Bernstein-weight update with synthetic latent S has the correct Dirichlet posterior parameters.
- `test_stickbreak.py` — `GEM(α, σ)` weights sum to 1 in expectation up to Monte-Carlo error; expected number of clusters scales as $\alpha n^\sigma$ for $\sigma > 0$ and $\alpha \log n$ for $\sigma = 0$ within tolerance.

### 6.2 Property (`tests/property/`, hypothesis-driven)
- `test_plod_invariance.py` — every $c_{\theta_m}$ from `copulas.{gaussian, student_t, clayton, gumbel}` with parameters in the prior support satisfies `C(u,...,u) > u^K` on a $u$-grid.
- `test_sklar_regularity.py` — Lemma A.1: under random β-Hölder marginals, $h^2(c, c^*) \le C \KL(f, f^*) + C \sum_j \KL(f_j, f_j^*)$ holds with explicit C, in finite-sample empirical form.
- `test_mixent_decomposition.py` — Lemma A.2: $\KL(f, f^*) = \KL(c, c^*) + \sum_j \KL(f_j, f_j^*)$ on synthetic densities.
- `test_predictive_consistency.py` — PY predictive (Pitman 1996) generates partition-exchangeable sequences (de Finetti).

### 6.3 Numerical (`tests/numerical/`)
- `test_py_truncation.py` — Lemma A.3: $\E[r_T]$ matches the analytic bound for $\sigma = 0$ and $\sigma \in (0,1)$ within constant factors.
- `test_sieve_covering.py` — Lemma A.4: enumerated covering counts on a small sieve match the bound.
- `test_contraction_simulation.py` — sample size sweep on regime S1 produces empirical Hellinger-error decay matching $\epsilon_n = n^{-\beta/(2\beta+K)}(\log n)^t$ within a 2× constant factor across $n \in \{500, 1000, 2000, 5000\}$.

### 6.4 Inference (`tests/inference/`)
- `test_algo8_step.py` — single Algorithm-8 reassignment on a fixed state moves features between clusters with the correct conditional probabilities (Monte-Carlo over 10⁵ trials).
- `test_nuts_atoms.py` — NUTS on Cholesky-parameterized Gaussian copula recovers the true correlation matrix on a synthetic K=4 dataset; 0 divergences, split-$\widehat R \le 1.01$, ESS ≥ 400.
- `test_full_sweep_smoke.py` — one full Algorithm 1 sweep on a tiny S1 cell completes in < 5 s and yields a finite log-likelihood.
- `test_alpha_sigma_updates.py` — Escobar–West and σ-slice updates have the correct stationary distributions on a frozen state (KS test against the analytic conditional).

### 6.5 Identifiability (`tests/identifiability/`)
- `test_plod_failure_breaks_id.py` — synthesizing two copula mixtures that agree marginally but differ in $\pi$ — both PLOD-violating — produces ambiguous EM/MCMC fits; switching to the PLOD-constrained prior resolves them.
- `test_yakowitz_spragins.py` — random samples of two distinct correlation matrices yield linearly independent log-density vectors on a fixed grid (rank-stability check).

### 6.6 Decision (`tests/decision/`)
- `test_sun_cai_calibration.py` — on a synthetic regime where the truth is known, the Sun–Cai-type plug-in step-up rule's realized FDR at $\alpha = 0.05$ is within 1 SE of the nominal level on $\ge 80$ of 100 replications.
- `test_bayes_mfdr_monotone.py` — Bayes-mFDR is non-decreasing in γ over a grid (operational sanity).
- `test_local_idr_marginalization.py` — averaging over a posterior trace recovers the local idr to within MC error of a brute-force enumeration on a 5-feature state.

### 6.7 Integration (`tests/integration/`)
- `test_simulation_S1_K4_n2000.py` — end-to-end MCMC fit on a tiny S1 cell; recovers $\pi^*$ within Monte-Carlo error and produces a valid Sun–Cai threshold.
- `test_encode_ctcf_dryrun.py` — `download_data.sh --dry-run` + a 100-feature subset fits without errors; gated by `slow`.
- `test_vi_matches_mcmc_means.py` — on a small problem, posterior means under VI agree with MCMC posterior means to within a fixed tolerance (qualitative not exact).

### 6.8 Regression (`tests/regression/`)
- Golden-file MCMC traces at 3 fixed seeds: PRNG state hash + posterior summary statistics must round-trip.
- Throughput regression: `benchmarks.yml` pushes a CSV; PRs fail if MCMC throughput drops > 5% on the smoke shape.

### 6.9 Smoke (`tests/`, `smoke` marker)
- `stick_idr.doctor()` succeeds.
- `scripts/make_all_figures.sh --dry-run` exits 0.

### Coverage targets
- `src/stick_idr/algebra/`, `copulas/`, `marginals/`, `decision/`: **100%** line+branch.
- `src/stick_idr/inference/`, `pym/`, `model/`: **≥ 90%**.
- Overall: **≥ 85%**.

---

## 7. Documentation (`docs/`)

Engine: **MkDocs + Material** with `mkdocstrings` for API, `mkdocs-jupyter` for tutorials. The existing LaTeX paper stays at `docs/report/` (unchanged) and is linked from the site.

### `docs/mkdocs.yml` — nav skeleton

```
nav:
  - Home: index.md
  - Getting Started:
    - Install: getting_started/install.md
    - Quickstart — 50 lines: getting_started/quickstart.md
    - Reproducing the paper: getting_started/reproduce.md
  - The Math:
    - The IDR framework, recalled: math/idr_recap.md
    - Sklar's theorem & copulas: math/sklar.md
    - The Pitman–Yor process: math/pitman_yor.md
    - Algorithm 8 with H aux atoms: math/algorithm8.md
    - Bernstein–Dirichlet marginals: math/bernstein_dirichlet.md
    - LKJ priors and structured correlation: math/lkj_structured.md
    - Sun–Cai compound decision: math/sun_cai.md
    - Posterior contraction (Thm 3.2): math/contraction.md
    - Variational consistency (Thm 3.5): math/vi_consistency.md
  - Guides:
    - Choosing K and atoms: guides/choosing_K.md
    - Tuning the auxiliary-atom count H: guides/auxiliary_atoms.md
    - When to prefer VI over MCMC: guides/vi_vs_mcmc.md
    - Diagnostics and traceplots: guides/diagnostics.md
    - Working with ranks vs. raw scores: guides/ranks_vs_scores.md
  - Tutorials (.ipynb):
    - 00 — IDR recap: tutorials/00_idr_recap.md
    - 01 — Gaussian-copula density: tutorials/01_gaussian_copula_density.md
    - 02 — PY stick-breaking: tutorials/02_py_stickbreak.md
    - 03 — Bernstein marginals: tutorials/03_bernstein_marginals.md
    - 04 — Algorithm 8 step by step: tutorials/04_algorithm8_step_by_step.md
    - 05 — NUTS on Cholesky: tutorials/05_nuts_on_cholesky.md
    - 06 — Sun–Cai plug-in: tutorials/06_sun_cai_plugin.md
    - 07 — Simulation S1 walkthrough: tutorials/07_simulation_S1_walkthrough.md
    - 08 — ENCODE CTCF demo: tutorials/08_encode_ctcf_demo.md
    - 09 — Preregistration vs results: tutorials/09_preregistration_vs_results.md
  - Experiments:
    - S1–S5 simulation suite: experiments/simulations.md
    - ENCODE CTCF: experiments/encode_ctcf.md
    - ENCODE ATAC: experiments/encode_atac.md
    - DREAM5: experiments/dream5.md
    - Tabula Muris: experiments/tabula_muris.md
    - Pan-UKBB: experiments/pan_ukbb.md
    - Verdict ledger: experiments/ledger.md
  - Developer:
    - Repo layout: developer/layout.md
    - Inference notes: developer/inference.md
    - Benchmarking protocol: developer/benchmarking.md
    - Style & conventions: developer/style.md
    - Release process: developer/release.md
  - API Reference: api/*
  - Paper (PDF): report/main.pdf
  - Changelog: changelog.md
```

### Content principles

- Every math page ends with a `Try it in a notebook` link.
- Every experiments page opens with its dataset table row, its hypothesis (if any), and the command that produces it:
  ```
  $ stick-idr fit --config configs/data/encode_ctcf.yaml --inference mcmc
  $ stick-idr eval --trace runs/<run_id>/trace.zarr
  $ stick-idr figure --fig f1
  ```
- API reference is fully generated from docstrings (numpy style, enforced by ruff/pydocstyle).
- Dark mode enabled; maths via MathJax; cross-refs via `[`PYMixture`][stick_idr.PYMixture]` syntax.

---

## 8. Notebook Tutorials (`notebooks/`)

Each `.ipynb` is paired with a `.py` (jupytext) so review is easy. Every notebook:
- Declares its dependencies at the top (`%pip install ...` behind `COLAB=1`).
- States, in the first markdown cell, the **paper section(s) it explains** and the **test(s) it's the narrative dual of**.
- Ends with a "check-yourself" block (a small assertion the reader runs).

| # | Notebook | Explains | Checks |
|---|---|---|---|
| 00 | `00_idr_recap.ipynb` | Original IDR (Li et al. 2011) on K=2 synthetic data | recovers $\pi^*$ within MC error |
| 01 | `01_gaussian_copula_density.ipynb` | Gaussian copula log-density via Cholesky | matches scipy.stats.multivariate_normal CDF on a grid |
| 02 | `02_py_stickbreak.ipynb` | $\GEM(\alpha,\sigma)$ + truncation | E[#clusters] tracks the analytic curves of §1.2 |
| 03 | `03_bernstein_marginals.ipynb` | Bernstein–Dirichlet on $[0,1]$ | partition of unity; conjugate update sanity |
| 04 | `04_algorithm8_step_by_step.ipynb` | Algorithm 8 with H aux atoms walked on tiny tensors | reassignment matches a brute-force enumeration |
| 05 | `05_nuts_on_cholesky.ipynb` | NUTS on the unconstrained Cholesky parameterization | 0 divergences; ESS ≥ 400 on a fixed problem |
| 06 | `06_sun_cai_plugin.ipynb` | Plug-in step-up rule on synthetic local-idr scores | realized FDR ≈ nominal at α=0.05 |
| 07 | `07_simulation_S1_walkthrough.ipynb` | End-to-end S1 cell at K=4, n=2000 | $\widehat R \le 1.01$; recovers $\pi^*$ |
| 08 | `08_encode_ctcf_demo.ipynb` | ENCODE 4 CTCF (downsampled to 5k peaks) | discovery count > Vanilla IDR pairwise on the same subset |
| 09 | `09_preregistration_vs_results.ipynb` | How `hypotheses.yaml` drives the verdict ledger | round-trip of measurements → JSON → markdown |

Notebook execution in CI:
- Fast path (`FAST_TUTORIAL=1`): sizes reduced, training to ~50 sweeps. Runs in < 3 min total.
- Slow path (nightly): full sizes, takes the docs build to ~20 min. Caches downloaded data.

---

## 9. Figure and Table Reproducer Scripts

Every `src/stick_idr/figures/*.py` and `src/stick_idr/tables/*.py` has the same interface:

```python
# src/stick_idr/figures/f1_idr_calibration.py
"""Reproducer for fig:calibration (Figure 2 in the report).

Inputs:
    --results artifacts/sim_calibration.csv   # CSV of (method, regime, K, nominal, realized_fdr, mc_se)
    --out docs/report/figures/fig_idr_calibration.pdf

Produces: 4-panel figure (S1, S2, S4, S5) of nominal IDR vs realized FDR
with 95% MC bands. Matches the illustrative figure produced by
docs/report/figures/generate_figures.py (which uses synthetic numbers).
"""
```

Convention:
- **No data collection in figure scripts.** A figure script reads a CSV from `artifacts/`.
- **CSVs are produced by evaluation scripts** (`src/stick_idr/eval/*.py`) or by the simulation driver.
- **The figure script signs its output**: writes a sidecar JSON with the git SHA, dataset version, and `fig:*` / `tab:*` label it implements.

`scripts/make_all_figures.sh`:

```bash
#!/usr/bin/env bash
set -euo pipefail
for fig in f1_idr_calibration f2_roc_pr f3_contraction f4_py_vs_dp f5_realdata f6_runtime; do
  python -m stick_idr.figures.${fig} \
    --results artifacts/${fig}.csv \
    --out docs/report/figures/$(python -c "import sys; print({'f1_idr_calibration':'fig_idr_calibration','f2_roc_pr':'fig_roc_pr','f3_contraction':'fig_contraction','f4_py_vs_dp':'fig_py_vs_dp','f5_realdata':'fig_realdata','f6_runtime':'fig_runtime'}['${fig}'])").pdf \
    --style-file src/stick_idr/figures/_style.py
done
```

`scripts/make_all_tables.sh` analogous, producing `docs/report/tables/*.tex` consumed by the paper via `\input`. Both directories are the paper's own tree so a LaTeX rebuild picks them up automatically.

---

## 10. Logging & Experiment Tracking

Structured logging via `structlog` — JSON lines on disk (`logs/<run_id>.jsonl`), pretty console via `rich`. Every log record carries `run_id`, `git_sha`, `config_hash`, `dataset_version`, `seed`.

Metrics: **wandb** by default (`inference.logger=wandb`); TensorBoard and Aim are alternative backends selected by Hydra config. A `DummyLogger` backend is used in CI.

What every fitting run logs, without exception (drives the ledger):

- Every 50 sweeps: log-likelihood, log-posterior, $\pi$ posterior, # active clusters $T$, split-$\widehat R$ on $\pi$, ESS-per-second on $\pi$, walltime.
- Every 200 sweeps: traceplots for $\pi$, $T$, $\sigma$, $\alpha$ via arviz; cluster-occupancy histogram.
- Every checkpoint: cluster correlation matrices (one per active cluster); atomic-type histogram.
- At run start: full resolved Hydra config, `pip freeze`, JAX device snapshot, dataset checksum.
- At run end: hypothesis verdicts (via `preregistration.verdict`), Bayes-mFDR curves, Sun–Cai threshold and discovery count.

---

## 11. Reproducibility Plumbing

- **Seeding** — `utils/seeding.py::seed_everything(seed)` returns a JAX `PRNGKey` and sets `random` and `numpy` seeds. JAX is functionally pure so trace reproducibility follows from the key alone; we still pin `JAX_DEFAULT_DTYPE_BITS=32` per-config and lift to 64-bit for theory-sensitive paths.
- **Hardware metadata** — `utils/profiling.py::system_snapshot()` dumps `jax.devices()`, `jaxlib.__version__`, host info, Slurm job id (if present).
- **Library versions** — `pip freeze` and `constraints.txt` hash committed into the run's artifact tarball.
- **Dataset splits** — `data/splits.py` registers canonical splits with SHA-256 checksums; `scripts/verify_checksums.sh` is pre-flight-required by `run_simulations.sh` and `run_real_data.sh`.
- **Inference budget** — driver tracks wall time and number of likelihood evaluations.
- **Trace archival** — every MCMC trace serialises to a Zarr store (atomic `*.tmp` rename) with a SHA-256 manifest. Final traces published to a GitHub release.

---

## 12. Workstreams W1–W6

| Workstream | Scope | Repo impact |
|---|---|---|
| **W1 — Skeleton + reference numerics** (2w) | Pure-Python copula densities, Bernstein basis, PLOD probe, fitness & shape tests | `src/stick_idr/{algebra,copulas,marginals,model}/*`, `tests/{unit,property}/*`, MkDocs site up, notebooks 00–03. |
| **W2 — Sampler v0** (3w) | Algorithm 1 sweep end-to-end on a tiny S1 cell | `src/stick_idr/{pym,inference/mcmc}/*`, `tests/inference/*`, notebooks 04–05, `experiments/sim_S1/`. |
| **W3 — Decision theory + simulation S1–S5** (3w) | Sun–Cai plug-in + Bayes-mFDR; full S1–S5 sweep at K∈{2,5,10,50}, n∈{2k,10k,50k}; verdict for FDR-calibration & power hypotheses | `src/stick_idr/decision/*`, `src/stick_idr/simulation/*`, figures F1–F4, tables T3–T5. |
| **W4 — Variational alternative** (2w, overlaps) | Structured-MF VI with stochastic natural gradients | `src/stick_idr/inference/vi/*`, notebook on VI; F6 runtime. |
| **W5 — Real-data pipelines** (4w) | ENCODE 4 CTCF / ATAC, DREAM5, Tabula Muris pseudobulk, Pan-UKBB | `src/stick_idr/data/*`, `experiments/{encode_ctcf,encode_atac,dream5,tabula_muris,pan_ukbb}/`, F5, T6, T7. |
| **W6 — Public release** (2w) | Open-source v0.1 → v1.0 | Tag `v1.0.0`, Zenodo DOI, blog post, `release.yml` publishes. |

Each workstream is a GitHub Project column; issues carry a `W1`–`W6` label and link to the corresponding `fig:*` / `tab:*` / theorem tag.

---

## 13. Release & Versioning

- **Semver.** Pre-1.0 during W1–W3; bump to 1.0 at W6.
- **Git tags** `v*` trigger `release.yml`.
- **Conda-forge** feedstock after 1.0.
- **Zenodo integration** for a DOI per release.
- **Citation file** (`CITATION.cff`) points to the preregistration paper (`docs/report/main.pdf`) and, once shipped, to the companion experimental paper.

---

## 14. Risk Register

| Risk | Trigger | Mitigation |
|---|---|---|
| NUTS divergences on Cholesky factor | `nuts-correctness` red | Reparametrize via `LowerCholeskyAffine`; tune target_accept up to 0.99; report and gate with diagnostic. |
| Algorithm 8 cluster reassignment is slow | `runtime-smoke` regresses | Lower H in default; cache atom log-density evaluations; investigate retrospective-sampler swap. |
| PLOD probe false positives near $\sigma = 1$ | Property test red | Tighten sampling region of $\rho$; add a defensive correlation-clipping in `algebra/correlation.py`. |
| Bernstein basis numerically unstable at $M=80$ | Bernstein property test red | Use the `betainc` fallback path; switch to log-domain accumulation. |
| Dataset checksum drift (ENCODE / Pan-UKBB mirrors) | `verify_checksums.sh` fails | Pin SHA-256 in `data/splits.py`; document fallback consortium URLs. |
| Prereg threshold flap | `verify-preregistration` red | Schema pinned; human-review required for every `hypotheses.yaml` edit; `CODEOWNERS` requires paper author. |
| Notebook CI flakes on download | `test-notebooks` red | Cache HF datasets; tiny-dataset fallbacks. |
| Bus-factor on inference math | Author unavailable | Sampler notes in `developer/inference.md` + notebooks 04–05 are mandatory reading for any sampler change. |

---

## 15. Delivery Checklist (definition of done for the plan)

- [ ] `scripts/create_env.sh` installs a working env on macOS (CPU/Metal) and Linux+CUDA in < 5 min.
- [ ] `pytest -m smoke` passes on a fresh clone.
- [ ] `mkdocs build --strict` passes; site deployed to `gh-pages`.
- [ ] Every figure in `docs/report/figures/` has a reproducer in `src/stick_idr/figures/` and a hyperlinked `docs/experiments/*.md` page.
- [ ] Every table in `docs/report/tables/` has a reproducer in `src/stick_idr/tables/`.
- [ ] Every theorem (3.1–3.5) and lemma (A.1–A.4) has a corresponding test or property check.
- [ ] Algorithm 1 sweep runs end-to-end on a tiny S1 cell, recovers $\pi^*$ within MC error, reports a Sun–Cai threshold, and writes a Zarr trace with a SHA-256 manifest.
- [ ] CI green on `main`; nightly benchmark publishes.
- [ ] `stick_idr.doctor()` returns all-green on a target dev machine.
- [ ] W1–W3 issues closed; companion paper rebuild produces a PDF whose tables and figures point at `artifacts/*.csv` rather than the synthetic `generate_figures.py`.

---

## 16. Claude Skill Files (`.claude/skills/`)

Skills capture recurring engineering workflows so a future agent can be invoked with the right vocabulary. The skill files are described in detail in [plans/00_skills_overview.md](plans/00_skills_overview.md); the inventory is:

| Skill | When to invoke |
|---|---|
| `add-copula-atom` | Adding a new copula family to the `copulas/registry.py` and updating Algorithm 8 step 4 |
| `algorithm8-sanity` | Running the Algorithm 8 reassignment sanity check on a saved sampler state |
| `run-simulation` | Launching one of the S1–S5 cells with the standard MC replication |
| `add-real-dataset` | Adding a new data loader under `src/stick_idr/data/` with checksum + smoke loader |
| `reproduce-figure` | Regenerating one of F1–F6 from `artifacts/*.csv` |
| `reproduce-table` | Regenerating one of T1–T7 |
| `verdict-update` | Producing the hypothesis verdict JSON from a finished run and updating the ledger |
| `mcmc-diagnostics-review` | Reviewing split-$\widehat R$, ESS, and divergence stats on a Zarr trace |

Each skill has a `SKILL.md` with: when-to-invoke heuristics, prerequisites, the command sequence, and the expected output paths.

---

## 17. First 30 Days — Concrete Order of Work

Suggested sequence (single engineer, 5 hours/day):

1. **Day 1–2** — Bootstrap: `pyproject.toml`, `create_env.sh`, `.pre-commit-config.yaml`, `ci.yml`, `src/stick_idr/` skeleton with stubs, doctor() smoke test. Write notebook 00 (IDR recap on K=2 synthetic data).
2. **Day 3–4** — `algebra/correlation.py` (Cholesky ↔ R, LKJ(η=2)), `copulas/{independence,gaussian}.py`, `algebra/plod.py`, unit + property tests. Notebook 01.
3. **Day 5** — `pym/stickbreak.py`, `marginals/{bernstein,dirichlet_weights}.py`, unit tests; notebooks 02 + 03.
4. **Day 6–7** — `copulas/{student_t,clayton,gumbel}.py` + property tests; PLOD test sweep. MkDocs site live with the math primer pages.
5. **Week 2** — `pym/algo8.py`, `inference/mcmc/{class_assign,reassign,nuts_atoms,sweep}.py`. Notebook 04 (Algo 8 walked) + notebook 05 (NUTS on Cholesky).
6. **Week 3** — `decision/{local_idr,bayes_mfdr,sun_cai}.py` + tests; `simulation/scenarios.py` for S1; first end-to-end fit on a tiny S1 cell. Notebook 06.
7. **Week 4** — Full S1–S5 simulation harness; figure F1 (idr-FDR calibration) reproducer wired to `artifacts/sim_calibration.csv`; benchmarks skeleton; `gpu-ci.yml` wired.

After 30 days the repo supports the full tutorial flow, an end-to-end fit on a small simulation cell, and one calibration figure produced by the package rather than `generate_figures.py`. The remaining workstreams are a matter of scaling.

---

## 18. Cross-Reference Index

For quick lookup, every element of the plan pointing back into the report:

- **Theorems** —
  Thm 3.1 identifiability: `tests/identifiability/*`.
  Thm 3.2 contraction: `tests/numerical/{test_contraction_simulation,test_py_truncation,test_sieve_covering}.py`.
  Thm 3.3 Bayes-mFDR: `decision/sun_cai.py`, `tests/decision/test_sun_cai_calibration.py`.
  Thm 3.4 ergodicity: `inference/mcmc/diagnostics.py` + nightly `gpu-ci.yml` long-chain mixing checks.
  Thm 3.5 VI consistency: `tests/integration/test_vi_matches_mcmc_means.py`.
- **Lemmas** — A.1: `tests/property/test_sklar_regularity.py`. A.2: `tests/property/test_mixent_decomposition.py`. A.3: `tests/numerical/test_py_truncation.py`. A.4: `tests/numerical/test_sieve_covering.py`.
- **Simulation regimes** — S1 → `experiments/sim_S1/`; … S5 → `experiments/sim_S5/`. Each has a Hydra config in `configs/simulation/`.
- **Datasets** — D1 ENCODE CTCF, D2 ENCODE ATAC, D3 DREAM5, D4 Tabula Muris, D5 Pan-UKBB; each gets a `data/<name>.py`, an `experiments/<name>/`, and a docs page.
- **Figures F1–F6** — F1 calibration → `figures/f1_idr_calibration.py` … F6 runtime → `figures/f6_runtime.py`.
- **Tables T1–T7** — T1 notation → `tables/t1_notation.py` … T7 real results → `tables/t7_real_results.py`.
- **Workstreams W1–W6** — GitHub Projects columns; each issue links a `fig:*`, `tab:*`, theorem, or simulation regime.

End of plan.
