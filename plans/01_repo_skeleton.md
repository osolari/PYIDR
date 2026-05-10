# Plan 01 — Repo Skeleton

**Owner:** W1 (Skeleton + reference numerics)
**Predecessor:** none
**Successor:** [02_algebra_and_copulas](02_algebra_and_copulas.md)

## Goal

A fresh clone can run `bash scripts/create_env.sh && pytest -m smoke` and pass.

## Deliverables

1. `pyproject.toml` (hatchling, src layout, all extras).
2. `src/py_idr/` tree of stub modules per §3 of `IMPLEMENTATION_PLAN.md`.
3. `src/py_idr/__init__.py` exporting the public surface (stubs).
4. `src/py_idr/doctor.py` — JAX backend probe + minimal sanity check.
5. `src/py_idr/cli.py` — Typer CLI with `fit`, `eval`, `sim`, `figure`, `table`, `doctor`.
6. `scripts/create_env.sh` (idempotent, JAX-backend-aware).
7. `.pre-commit-config.yaml` with hygiene + format + lint + types + project-specific stages.
8. `.github/workflows/ci.yml` skeleton calling `setup-env` composite + `pytest -m "not gpu"`.
9. `.github/actions/setup-env/action.yml` composite action.
10. `tests/conftest.py` with shared fixtures (random key, tiny synthetic dataset, monkeypatched logger).
11. `tests/smoke/test_doctor.py` — calls `doctor()` and asserts all-green.

## Acceptance criteria

- `pip install -e ".[dev]"` succeeds on macOS-arm64 and ubuntu-x86_64.
- `pytest -m smoke` is green.
- `python -c "import py_idr; py_idr.doctor()"` exits 0.
- `pre-commit run --all-files` is green.

## Notes

- Default JAX dtype: 32-bit. The contraction-rate tests will lift to 64-bit per-test.
- All stub functions raise `NotImplementedError("see IMPLEMENTATION_PLAN.md §...")` referencing the planned implementing section.
