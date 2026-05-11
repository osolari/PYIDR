# Testing

PY-IDR has ~250 tests organised by what part of the model they exercise.
This page is the operator's manual: how to run them, what each marker
costs, and how the slow / GPU markers interact.

## The marker taxonomy

| Marker        | Default? | Approximate cost   | What it tests                                         |
|---------------|----------|--------------------|-------------------------------------------------------|
| `smoke`       | yes      | < 1 s              | Imports work, doctor runs, schemas construct.         |
| `unit`        | yes      | < 1 s/test         | Pure functions; no chain.                             |
| `property`    | yes      | < 10 s/test        | Hypothesis-driven invariants.                         |
| `decision`    | yes      | < 1 s/test         | Sun-Cai, local idr, Bayes mFDR.                       |
| `integration` | yes      | 1-30 s/test        | End-to-end chain on tiny configs.                     |
| `slow`        | no       | 30 s-5 min/test    | Realistic-config integration; multi-replicate sweeps. |
| `benchmark`   | no       | varies             | pytest-benchmark micro-benchmarks.                    |

The default selection is configured in `pyproject.toml`:

```toml
[tool.pytest.ini_options]
addopts = "-m 'smoke or unit or property or decision or (integration and not slow)'"
```

So `pytest` (no arguments) runs the fast set. To run the full suite:

```bash
pytest -m ""    # no marker filter
```

## Common invocations

### Fast cycle (your every-commit gate)

```bash
JAX_PLATFORMS=cpu pytest -m "smoke or unit or property or decision" -q
```

~200 tests in ~30 seconds. Run this before every commit.

### After touching the chain code

```bash
JAX_PLATFORMS=cpu pytest -m "integration and not slow" -q
```

~15 tests in 2-3 minutes. The chain end-to-end on tiny configs.

### Full suite (CI)

```bash
JAX_PLATFORMS=cpu pytest -m "" -q
```

All tests. 10-15 minutes on CPU.

### Single module

```bash
pytest tests/simulation/test_scenarios.py -v
```

### Single test by name

```bash
pytest -k "test_run_chain_simple_z_trace" -v
```

### Running with coverage

```bash
JAX_PLATFORMS=cpu pytest --cov=py_idr --cov-report=term-missing -m "smoke or unit or property or decision"
```

The `[tool.coverage.run]` config in `pyproject.toml` covers
`src/py_idr/` and excludes test files. Branch coverage is enabled.

## Hypothesis strategies

The property-based tests live under `tests/property/` and `tests/copulas/`.
They use Hypothesis to generate random copula parameters and check
invariants like:

- Copula CDFs are monotone non-decreasing in each coordinate.
- The Bernstein-Dirichlet density integrates to 1.
- The PLOD projection produces a positive-definite matrix with
  non-negative off-diagonals.

If a property test fails, Hypothesis prints the *smallest* failing
example. Save it via `--hypothesis-explain` for the diagnosis.

## Benchmarks

The benchmark suite is opt-in:

```bash
JAX_PLATFORMS=cpu pytest -m "benchmark" --benchmark-only
```

The default thresholds are tuned for the M4 Pro dev workstation; CI does
not run benchmarks. To compare against a saved baseline:

```bash
pytest -m "benchmark" --benchmark-only --benchmark-compare=0001
```

## Snapshot tests

A few tests are snapshot-driven (the figure-reproducer tests under
`tests/figures/`). They compare a re-generated artifact against a
checked-in snapshot. To update a snapshot after an intentional change:

```bash
pytest tests/figures/ --snapshot-update
```

Inspect the diff carefully — the snapshots are checked into git on
purpose.

## CI specifics

CI runs on Linux + CPU. The matrix:

- Python 3.11 + Python 3.12.
- `pip install -e ".[dev,viz,data]"`.
- `pre-commit run --all-files` (lint gate).
- `pytest -m ""` (full suite).
- Coverage uploaded to a per-PR comment.

GPU CI runs nightly on the A10-dev workstation and exercises the
realistic-config sweeps. Plan 14 covers the Docker image used for those
runs.

## Writing a new test

The conventions are:

```python
# tests/<area>/test_<module>.py
"""Brief module docstring saying what's tested."""

from __future__ import annotations

import pytest


@pytest.mark.unit  # or whatever marker fits
def test_descriptive_name_with_what_it_asserts() -> None:
    """One-line docstring saying the invariant."""
    # arrange
    ...
    # act
    ...
    # assert
    assert ...
```

A few specific patterns:

- **Approximate equality.** Use `pytest.approx` for scalars,
  `np.allclose` for arrays. Pick the tolerance carefully — `1e-6` is the
  default but Monte Carlo tests need `1e-2` or coarser.
- **Random tests.** If your test draws randomness, seed it explicitly
  inside the test body. Do not rely on global RNG state.
- **JAX tests.** Set `JAX_PLATFORMS=cpu` at the top of `conftest.py`
  (already done in the repo). Do not call `.block_until_ready()` —
  pytest already waits for the call to return.
- **Skipping for backend.** `pytest.skipif(jax.devices()[0].platform !=
  "cpu", reason="...")` is fine for backend-specific tests.

## When a test is slow

Mark it `slow`. If it is also part of the chain integration suite, mark
both:

```python
@pytest.mark.integration
@pytest.mark.slow
def test_long_chain_converges() -> None:
    ...
```

The CI fast cycle will skip it; the nightly run will pick it up.

## When a test is flaky

Flaky tests in PY-IDR usually come from one of three places:

1. **Monte Carlo variance.** Use a fixed seed and check looser
   tolerances.
2. **JAX-Metal backend bugs.** Set `JAX_PLATFORMS=cpu` in
   `tests/conftest.py` (already done).
3. **Hypothesis edge cases.** Hypothesis sometimes finds bizarre inputs.
   Use `@given(... assume(...))` to filter, or tighten the strategy.

The codebase has zero tolerance for `@pytest.mark.flaky` — fix the test
or remove it.
