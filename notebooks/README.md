# Hands-on tutorial notebooks

This directory contains a sequence of tutorial notebooks that walk
through the moving parts of PY-IDR — from the IDR likelihood through the
Pitman-Yor mixture to the Sun-Cai decision rule. Each notebook is
self-contained but builds on the previous one.

## File format

The notebooks are stored as **jupytext percent-format Python files**
(`*.py` with `# %%` cell separators). This format:

- Diffs cleanly in git (no JSON noise),
- Lints with ruff / black / mypy like normal Python,
- Renders as a real `.ipynb` notebook in JupyterLab, VS Code, and
  Cursor without any conversion step.

To open one as a notebook in JupyterLab:

```bash
jupyter lab notebooks/00_idr_recap.py
```

Jupytext's "open as notebook" handler is auto-registered after
`pip install -e ".[dev]"`. VS Code's Jupyter extension recognises the
percent format natively — just open the `.py` file and the cells appear.

To convert one to a standalone `.ipynb` (e.g. for sharing):

```bash
jupytext --to ipynb notebooks/00_idr_recap.py
```

## Running the suite

The dev dependencies include `nbmake`, which can execute the
percent-format notebooks via pytest. Convert + execute all:

```bash
# Convert .py to .ipynb (transient — clean up after).
jupytext --to ipynb notebooks/*.py

# Execute the notebooks as part of the test suite.
JAX_PLATFORMS=cpu pytest --nbmake notebooks/*.ipynb
```

(`JAX_PLATFORMS=cpu` is recommended on Mac because of the known
`jax-metal` `default_memory_space` bug; the chain code is CPU-friendly.)

## The sequence

| # | Notebook                              | What you learn                                                    |
|---|---------------------------------------|-------------------------------------------------------------------|
| 0 | `00_idr_recap.py`                     | The two-component IDR mixture, hand-implemented in numpy.         |
| 1 | `01_gaussian_copula.py`               | Drawing from a Gaussian copula; densities; PIT visualisation.     |
| 2 | `02_py_stickbreak.py`                 | Pitman-Yor stick-breaking; prior on the cluster count $T$.        |
| 3 | `03_bernstein_marginals.py`           | Bernstein-Dirichlet marginal fit on a heavy-tailed sample.        |
| 4 | `04_polya_urn_step_by_step.py`        | A single Pólya-urn cluster reassignment, traced.                  |
| 5 | `05_nuts_on_cholesky.py`              | NUTS on the LKJ-Cholesky + PLOD parameterisation.                 |
| 6 | `06_sun_cai_plugin.py`                | The Sun-Cai step-up rule on a synthetic local idr distribution.   |
| 7 | `07_simulation_S1_walkthrough.py`     | End-to-end S1 run with full diagnostics.                          |

If you only have time for one, do **07** — it pulls every piece
together. If you only have time for two, do **0** then **07**.

## Prerequisites

The notebooks assume:

- Python 3.11 or 3.12.
- PY-IDR installed in editable mode with the `dev` and `viz` extras.
- A Jupyter-aware editor (JupyterLab, VS Code, Cursor) or an
  IPython kernel running.

The plotting cells use `matplotlib` (in `[viz]`). If you do not have
that extra installed, the plots cells will fail; the analytic cells
still run.
