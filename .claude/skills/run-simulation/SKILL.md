---
name: run-simulation
description: Launch one of the S1–S5 simulation cells with the standard 100-fold MC replication and produce the long-format CSV consumed by figures F1–F4. Use when the user asks to "run simulation", "launch S<N>", "regenerate the calibration data", or "fit the heavy-tail benchmark".
---

# Skill: Run a Simulation Cell

Drives one cell of the S1–S5 × $K$ × $n$ factorial design.

## When to invoke

Triggered by any of:

- "run S1" / "run S5" / "launch the heavy-tail simulation"
- "regenerate the FDR calibration data"
- "produce sim CSV for K=10"
- "MC replicate the simulation at n=10000"

Do NOT use this skill for the real-data pipelines — use `add-real-dataset` (or a
dedicated `run-encode-ctcf`-style follow-on) instead.

## Prerequisites

- `.venv` activated; `stick_idr.doctor()` is green.
- `configs/simulation/S<N>.yaml` exists OR is created on the fly from the user's args.
- Output directory `artifacts/sim/` exists (created if not).

## Inputs (ask the user if missing)

- Regime: `S1 | S2 | S3 | S4 | S5`
- $K$: number of replicates, default 4
- $n$: feature count, default 10000
- `reps`: MC replications, default 100
- `inference`: `mcmc | vi`, default `mcmc` (use `vi` for $n \ge 50000$)
- `seed`: base seed, default 0

## Procedure

1. **Smoke run first.** Always run a single replicate with `reps=1` and the user's chosen
   `(K, n)` before launching the full sweep. Confirm the output schema is correct.
2. **Launch the sweep:**
   ```bash
   python -m stick_idr.simulation.replicates \
     --regime <S> --K <K> --n <n> --reps <reps> \
     --inference <mcmc|vi> --seed <seed> \
     --out artifacts/sim/<S>_K<K>_n<n>.csv
   ```
3. **Monitor.** The driver prints progress every 10 replicates with running $\widehat\pi$,
   realized FDR at $\alpha=0.05$, and ESS-per-second.
4. **Validate.** After the run completes:
   ```bash
   python -m stick_idr.eval.validate_sim_csv artifacts/sim/<S>_K<K>_n<n>.csv
   ```
   This checks the schema, NaN-freeness, and that realized FDR is finite for every row.
5. **Append to the master.** If running as part of a sweep, concatenate to
   `artifacts/sim/calibration_long.csv`:
   ```bash
   python -m stick_idr.simulation.concat \
     --in artifacts/sim/<S>_K<K>_n<n>.csv \
     --out artifacts/sim/calibration_long.csv \
     --append
   ```
6. **Refresh figures.** If the user asked for an end-to-end refresh:
   ```bash
   bash scripts/make_all_figures.sh
   bash scripts/make_all_tables.sh
   ```

## Expected outputs

- `artifacts/sim/<S>_K<K>_n<n>.csv` — one row per replicate, with columns
  `(regime, K, n, replicate, method, alpha, realized_fdr, power, hellinger, ess_per_s, walltime_s)`.
- `artifacts/sim/<S>_K<K>_n<n>.manifest.json` — git SHA, config hash, RNG seed, walltime.
- Optional: refreshed figures in `docs/report/figures/`.

## Failure modes

- **NUTS divergences.** If the run logs > 5% divergences, raise `target_accept_prob=0.95`
  in the config and rerun the affected replicate. Persistent divergences suggest a
  bad initialization — verify EM-IDR seeding is on.
- **Non-PLOD atom sampled.** This should be impossible by construction; if it happens,
  the bug is in `algebra/plod.py` or the `copulas/proposal.py` sampler. Stop and
  invoke the `add-copula-atom` failure-mode review.
- **CSV schema mismatch.** Means a code change broke the long-format. Run
  `pytest tests/property/test_csv_schema.py` and fix the divergence before continuing.
