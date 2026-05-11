# Contributing

This page covers the local workflow, the lint/test gates, and the
conventions that the codebase enforces.

## Local setup

```bash
git clone git@github.com:osolari/PYIDR.git
cd PYIDR
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,viz,data,docs]"
pre-commit install
```

The pre-commit hooks include ruff (lint + format), black, mypy, bandit,
and a custom secrets-detection step. They run on every commit and on
`pre-commit run --all-files`.

## The PR cycle

1. Branch off `main`. We do not use feature flags or long-lived branches.
2. Make focused changes. One PR = one logical change.
3. Run `pre-commit run --files <changed files>` to catch lint issues.
4. Run the fast test markers:
   ```bash
   JAX_PLATFORMS=cpu pytest -m "smoke or unit or property or decision" -q
   ```
5. If you touched the chain or simulation paths, run the integration
   tests:
   ```bash
   JAX_PLATFORMS=cpu pytest -m "integration and not slow" -q
   ```
6. Open the PR. Title format: `Wx.y: <short imperative>` for code
   changes that map to a plan step, free-form for everything else.

## Commit messages

Use the format:

```
Wx.y: <one-line imperative summary, < 70 chars>

<body — what changed and why>
```

`Wx.y` is the work-item label from `plans/`. The body explains the
*why*, not the *what* — diffs show what changed. We never include the
`Co-Authored-By: Claude` trailer.

## Style: what to write and not write

The codebase has a few opinions encoded in `pyproject.toml` and in
`CLAUDE.md` (if present). The most important:

- **No unnecessary docstrings.** Module docstrings are required;
  per-function docstrings are required for the *public surface only*
  (anything callable from outside the module). Private helpers do not
  need them.
- **No inline comments explaining what the code does.** The code does
  that. Inline comments are for explaining *why* — a non-obvious
  invariant, a workaround for a bug, a reference to the report.
- **No unused vars renamed to `_var`.** Just delete them.
- **No fallback / backwards-compat shims.** If you remove a feature,
  remove its callers too.
- **No `if TYPE_CHECKING:` unless circular.** Plain imports are
  preferred.
- **Use `jaxtyping` shape annotations** on every public JAX-array
  argument. Example: `F0: Float[Array, "n K"]`. The `pyproject.toml`
  already has per-file ignores for ruff's `F821`/`F722` false positives
  on these.

## The test taxonomy

Every test must carry exactly one marker from: `unit`, `property`,
`decision`, `integration`, `slow`, `smoke`, `benchmark`.

Rules of thumb:

| You are testing                                       | Marker        |
|-------------------------------------------------------|---------------|
| A pure function with no chain                         | `unit`        |
| A hypothesis-driven invariant                         | `property`    |
| Anything in `decision/`                               | `decision`    |
| The chain end-to-end with a tiny config               | `integration` |
| The chain with a realistic config (multi-second)      | `integration` + `slow` |
| Import + constructor + smoke test of one module       | `smoke`       |
| A micro-benchmark via `pytest-benchmark`              | `benchmark`   |

The CI default is to run `smoke or unit or property or decision or
(integration and not slow)`. Slow integration tests run nightly.

## Adding dependencies

Edit `pyproject.toml` and add to the appropriate group (`dependencies`,
`[project.optional-dependencies].dev`, etc.). Then:

```bash
pip install -e ".[dev]"  # or whatever group you touched
```

If the dependency is JAX-related, double-check it is compatible with
`jax>=0.4.30`. The pinned versions in the dev group have been chosen
for arviz / NumPyro compatibility.

## Updating the report

The LaTeX report lives at `docs/report/`. Changes there are
*independent* of the code unless they touch §3 (Algorithm 1) or §4.1
(simulation regimes); in that case open a coordinated PR that updates
the code and the report together.

The MkDocs site (this site you are reading) lives at `docs/` —
everything *except* `docs/report/` and `docs/experiments/`. The site is
rebuilt on every push to `main`.

## Common gotchas

1. **Pre-commit reformats files and aborts the commit.** Re-stage with
   `git add -A` and re-commit. This is by design — pre-commit reformats
   *during* the commit hook, and if anything changed it asks you to
   review and re-stage. Use `--no-verify` only as a last resort, and
   only after you have decided you do not want the reformat.
2. **JAX-Metal flakiness on Mac.** Set `JAX_PLATFORMS=cpu` before
   running tests on Apple Silicon. The CI runs on Linux + CPU and is
   the authoritative environment.
3. **Stale arviz cache.** `result.idata.posterior` is lazy in some
   arviz versions; if a test mutates the underlying NumPy data it can
   confuse the cache. Always go through `np.asarray(...)` before any
   mutation.
4. **Reading a fixed-shape array out of `MultiClusterState.atoms`.**
   `atoms` is a Python tuple of dataclasses. Iterate, do not index
   with a JAX array.

## Where to ask

- For algorithmic questions, post in the PR description with a link to
  the relevant section of the report.
- For "why is this test failing on Mac but not CI?" — first try
  `JAX_PLATFORMS=cpu`. Most answers are there.
- For copula or PY-mixture math, the [math primers](../math/idr_recap.md)
  are the canonical reference. The report's appendix has full proofs.
