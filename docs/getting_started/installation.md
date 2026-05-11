# Installation

PY-IDR targets **Python 3.11 or 3.12**. The build system is
[hatchling](https://hatch.pypa.io/); the dependency stack is
[JAX](https://docs.jax.dev/) for the array layer, [NumPyro](https://num.pyro.ai/)
for NUTS, and [arviz](https://python.arviz.org/) for diagnostics.

## From source (development)

The repo includes a `pyproject.toml` with named extras. The typical
development setup is:

```bash
git clone git@github.com:osolari/PYIDR.git
cd PYIDR

# Create a venv (uv, conda, virtualenv — your choice).
python -m venv .venv
source .venv/bin/activate

# Editable install with dev + viz + data extras.
pip install -e ".[dev,viz,data]"

# Install pre-commit hooks (ruff, black, mypy, bandit).
pre-commit install
```

After this you should have:

```bash
$ py-idr doctor
# A self-check that prints the JAX backend, available accelerators,
# the package version, and the path to the seed-ledger directory.
```

## Optional extras

| Extra      | Adds                                          | When to install                                           |
|------------|-----------------------------------------------|-----------------------------------------------------------|
| `cuda`     | `jax[cuda12]`                                 | Linux + NVIDIA GPU. Cuts an A10 inference run to seconds. |
| `metal`    | `jax-metal`                                   | Apple Silicon. **Unstable** — see "Backends" below.       |
| `data`     | `pyranges`, `h5py`, `anndata`, `scanpy`, `requests` | Reading the real ENCODE / scRNA-seq inputs.        |
| `viz`      | `matplotlib`, `seaborn`                       | Reproducing figures F1–F4.                                |
| `logging`  | `wandb`, `tensorboard`, `aim`                 | Sweep tracking.                                           |
| `docs`     | mkdocs-material, mkdocstrings, etc.           | Building this site locally.                               |

To install several:

```bash
pip install -e ".[dev,viz,data,docs]"
```

## Backends

PY-IDR uses JAX's default platform selection. To force CPU (recommended on
Mac during development because `jax-metal` has known bugs around
`default_memory_space`):

```bash
export JAX_PLATFORMS=cpu
```

On the A10-dev workstation (4 × A10), the GPU backend is selected by
default; see the [Docker A10-dev guide](../guides/reproducibility.md) for
the production sweep configuration.

## Verifying the install

The fast test markers run in well under a minute on CPU:

```bash
JAX_PLATFORMS=cpu pytest -m "smoke or unit or property or decision" -q
```

The full suite (including slow integration and benchmark tests) takes
several minutes:

```bash
JAX_PLATFORMS=cpu pytest -q
```

If the smoke tests fail with a JAX backend error, set `JAX_PLATFORMS=cpu`
and try again — the most common cause is a partial `jax-metal` install on
Apple Silicon.
