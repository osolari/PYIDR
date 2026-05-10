---
name: add-stress-test
description: Add a new stress-test scenario to the simulation harness without re-running the basic S1–S5 flow. Use when the user wants to test PY-IDR robustness against a specific data pathology (high tie rate, missing replicates, near-singular correlations, low reproducible mass, etc.) per §4.6 of the revised report.
---

# Skill: Add a Stress Test

The revised manuscript §4.6 lists explicit stress tests for PY-IDR: weakly dependent
irreproducible component, high tie rate, missing replicates, near-singular correlation
structures, large $K$, marginal misspecification, low reproducible mass, heavy boundary
tail dependence. This skill scaffolds one such test as a self-contained Hydra config +
CSV output, without disturbing the S1–S5 simulation harness.

## When to invoke

Triggered by any of:

- "add a stress test for <pathology>"
- "test robustness to high tie rates"
- "add a near-singular correlation stress test"
- "what happens at very low π*?"

Do NOT use this skill to add a new *primary* simulation regime — those go into
`simulation/scenarios.py` directly. Use this for pathologies that perturb an existing
regime.

## Catalogue (revised §4.6)

The plan keeps a fixed list of stress configs per the manuscript. The skill **must** add
to this list rather than inventing new categories without manuscript backing:

| Stressor | Knob |
|---|---|
| Weakly dependent $c_0$ | `c0_dependence: gaussian_weak` |
| High tie rate | `tie_rate: 0.05 → 0.40` |
| Missing replicates | `missing_rate: 0 → 0.30` |
| Near-singular correlation | `correlation_eigvalue_floor: 1e-3` |
| High dimensionality | `K: 50 → 200` |
| Marginal misspecification | `marginal_family: cauchy / lognormal` |
| Low reproducible mass | `pi_star: 0.30 → 0.02` |
| Boundary tail dependence | `gumbel_theta: → 1+` (near independence) |

## Prerequisites

- `experiments/sim_<base_regime>/` exists (the regime being perturbed).
- The user has named the stressor (one of the catalogue entries above).

## Procedure

1. **Pick the base regime.** Most stress tests perturb S1 or S5; ask the user.
2. **Create the config.** `experiments/sim_<base>/stress/<stressor>.yaml`:
   ```yaml
   defaults:
     - ../config
     - _self_
   simulation:
     <stressor_knob>: <value>
   tags:
     - stress
     - <stressor>
   ```
3. **Wire the data generator.** If the stressor needs a new code path, edit
   `simulation/scenarios.py` to accept the new knob; otherwise the existing simulator
   handles it via Hydra interpolation.
4. **Add an evaluation hook.** `experiments/sim_<base>/stress/<stressor>_eval.py`:
   what summary metric defines pass/fail (e.g., realised FDR delta vs the unperturbed
   regime; ESS ratio; cluster-recovery success rate).
5. **Add to the master sweep.** Append a stanza to `scripts/run_stress.sh`:
   ```bash
   python -m py_idr.simulation.replicates \
       --config experiments/sim_<base>/stress/<stressor>.yaml \
       --reps 100 \
       --out artifacts/stress/<stressor>/
   ```
6. **Document.** Add a row to `docs/experiments/stress_tests.md` with: stressor name,
   knob, expected qualitative behaviour (per §4.6), pass/fail criterion.
7. **Smoke test.** Run with `--reps 1` to confirm the config loads and the simulator
   accepts the new knob.

## Expected outputs

- `experiments/sim_<base>/stress/<stressor>.yaml`
- `experiments/sim_<base>/stress/<stressor>_eval.py`
- Updated `scripts/run_stress.sh` and `docs/experiments/stress_tests.md`.
- A smoke artifact under `artifacts/stress/<stressor>/smoke.csv` with one MC replicate.

## Failure modes

- **Stressor not in the catalogue.** If the user requests something not in §4.6's list,
  push back: "the manuscript does not list this stressor; it needs an author edit
  before we can add it as a §4.6 test." Suggest tracking it in
  `plans/16_proposed_stress_tests.md` instead.
- **Knob not exposed by the simulator.** Most knobs (tie rate, missing rate, $\pi^*$,
  $K$) are already exposed; high-eigvalue-floor and boundary-tail-dependence may need
  new code in `simulation/scenarios.py`.
- **Pass/fail criterion not defined.** Refuse to land a stress test without a
  pass/fail criterion. The verdict ledger must be able to score it.
