---
name: verdict-update
description: Build a hypothesis verdict JSON from a finished run and update the experiment ledger. Use when the user asks to "produce a verdict", "update the ledger", or "evaluate hypotheses against measurements".
---

# Skill: Update the Verdict Ledger

The verdict ledger is the audit trail linking preregistered hypothesis thresholds
(`configs/preregistration/hypotheses.yaml`) to measured outcomes. This skill produces
a single verdict JSON for a finished run and appends it to the ledger.

## When to invoke

Triggered by any of:

- "produce a verdict for <run-id>"
- "update the ledger"
- "evaluate hypotheses against the latest run"
- "run preregistration check"

## Prerequisites

- A finished run with `runs/<run_id>/measurements.json` containing every preregistered
  metric.
- `configs/preregistration/hypotheses.yaml` exists and validates against the pydantic
  schema.

## Procedure

1. **Load and validate the preregistration:**
   ```bash
   python -m py_idr.preregistration.loader configs/preregistration/hypotheses.yaml
   ```
2. **Compute the verdict:**
   ```bash
   python -m py_idr.preregistration.verdict \
     --measurements runs/<run_id>/measurements.json \
     --preregistration configs/preregistration/hypotheses.yaml \
     --out runs/<run_id>/verdict.json
   ```
   The output JSON has, for each hypothesis: `id`, `claim`, `threshold`, `measurement`,
   `pass | fail | falsified`, `evidence_path`.
3. **Update the ledger:**
   ```bash
   python -m py_idr.preregistration.ledger \
     --append runs/<run_id>/verdict.json \
     --ledger docs/experiments/ledger.md
   ```
   The ledger is markdown with one row per run.
4. **Commit (only if user requested):** stage `docs/experiments/ledger.md` and the
   verdict JSON; do not commit anything else.

## Expected outputs

- `runs/<run_id>/verdict.json`
- updated `docs/experiments/ledger.md` (one new row)

## Failure modes

- **Hypothesis schema drift.** If `hypotheses.yaml` was edited without updating the
  schema, the loader fails. Refer to `scripts/hooks/check_hypotheses_schema.py` for
  the canonical schema.
- **Falsified hypothesis.** If a measurement falls outside the falsification region,
  the verdict is `falsified`. This is *not* a script error — it is the expected
  behaviour of preregistration. Communicate the verdict prominently to the user.
- **Missing measurement.** If a preregistered hypothesis has no corresponding key in
  `measurements.json`, the verdict is `incomplete` and the ledger row is gated.
  Ask the user whether to fill the measurement before updating the ledger.
