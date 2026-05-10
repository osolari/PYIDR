# Skills Overview

Skills under `.claude/skills/` capture recurring engineering workflows for this repo.
Each skill is a directory with a `SKILL.md` describing:

1. **When to invoke** — the user-intent fingerprint that should trigger the skill.
2. **Prerequisites** — env, data, configs that must exist.
3. **Procedure** — the exact command sequence.
4. **Expected outputs** — files, run IDs, and where they land.
5. **Failure modes** — common errors and how to recover.

## Inventory

| Skill | Purpose |
|---|---|
| `add-copula-atom` | Add a new copula family (registry, density, sampler, PLOD test, registry-update). |
| `polya-urn-sanity` | Run the Pólya-urn auxiliary-atom (Eqs. 3.5–3.6) reassignment correctness check on a saved sampler state. |
| `add-stress-test` | Scaffold a §4.6 stress test (weak $c_0$ / high tie rate / missing replicates / etc.) without disturbing the S1–S5 flow. |
| `run-simulation` | Launch one S1–S5 simulation cell with the standard 100-fold MC replication. |
| `add-real-dataset` | Add a new data loader, register checksums, emit a smoke loader. |
| `reproduce-figure` | Regenerate one of F1–F6 from `artifacts/*.csv`. |
| `reproduce-table` | Regenerate one of T1–T7. |
| `verdict-update` | Build a hypothesis verdict JSON from a finished run; update the ledger. |
| `mcmc-diagnostics-review` | Review split-$\widehat R$, ESS, and divergences on a Zarr trace. |

## Conventions

- Skills assume `.venv/` is activated and `py_idr.doctor()` is green.
- Skills write artifacts into `artifacts/<skill-name>/<run-id>/` and traces into `runs/<run-id>/`.
- Skills never modify `docs/report/sections/*.tex`.
- Skills emit a sidecar `manifest.json` with git SHA, config hash, and dataset version.
