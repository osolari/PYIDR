# Plan 07 — Simulation Studies S1–S5

**Owner:** W3
**Depends on:** 02_algebra_and_copulas, 03_marginals, 06_decision_theory (and 04_inference_mcmc for fitting)
**Implements:** §4.1 (simulation design); Table 3 (sim design); Figures F1, F2, F3; Tables T4 (sim FDR), T5 (sim power), T6 (sim K-effect)

## Goal

A reproducible simulation harness that can run the full S1–S5 × $K \in \{2,5,10,50\}$ × $n \in \{2k, 10k, 50k\}$ factorial design with 100 MC replications per cell (60 cells × 100 reps = 6000 fits).

## Simulation design (from Table 3)

| Regime | Signal copula | $\pi^*$ | Marginals | Notes |
|--------|---------------|---------|-----------|-------|
| S1 | Gaussian, $\rho=0.85$ | 0.30 | shared Gaussian | baseline |
| S2 | Gaussian, $\rho=0.40$ | 0.30 | shared Gaussian | weak signal |
| S3 | Gaussian, $\rho=0.85$ | 0.10 | heterogeneous Gaussian | sparse signal |
| S4 | Gaussian, $\rho=0.85$ | 0.30 | heterogeneous skewed | marginal mismatch |
| S5 | mixture: $t_5$/Clayton/Gumbel | 0.30 | heterogeneous heavy-tail | heavy-tail extension |

## Deliverables

### `src/py_idr/simulation/`

- `scenarios.py`
  - `simulate_S1(rng, K, n, pi_star=0.30, rho=0.85)` etc., returning `SimulationResult(X, true_Z, true_pi, true_copula_params)`.
  - For S5, the mixture is sampled equally across $t_5$, Clayton, Gumbel atoms, calibrated so that the bivariate Spearman rank correlation under each family equals 0.65.
- `designs.py` — the $5 \times 4 \times 3$ factorial enumerator; emits `(regime, K, n, replicate)` tuples.
- `replicates.py` — `run_replicate(regime, K, n, seed)` — fits PY-IDR (MCMC), records local idr, computes Sun–Cai threshold, and records a row of the `artifacts/sim_*.csv` long-format CSV with columns `(regime, K, n, replicate, method, alpha, realized_fdr, power, hellinger, ess_per_s, walltime_s)`.
- `truth.py` — holds the true $Z$, $\pi$, and copula parameters for evaluation; cleanly separates simulation logic from evaluation logic.

### `experiments/sim_S1/` ... `sim_S5/`

Each contains:
- `config.yaml` — Hydra config selecting model and inference defaults.
- `run.py` — argparse wrapper that invokes `simulation.replicates.run_replicate`.
- `sbatch.sh` — slurm header for cluster runs.

### `scripts/run_simulations.sh`

```bash
#!/usr/bin/env bash
set -euo pipefail
for regime in S1 S2 S3 S4 S5; do
  for K in 2 5 10 50; do
    for n in 2000 10000 50000; do
      python -m py_idr.simulation.replicates --regime $regime --K $K --n $n --reps 100 \
        --out artifacts/sim/${regime}_K${K}_n${n}.csv
    done
  done
done
```

## Tests

- `tests/integration/test_sim_S1_smoke.py` — one replicate of S1, K=2, n=200, completes in < 30 s and produces a valid Sun–Cai threshold.
- `tests/numerical/test_sim_calibration_synthetic.py` — at K=2, $n = 10^4$, the fitted PY-IDR's realized FDR at $\alpha = 0.05$ is within 0.01 of nominal averaged over 50 replicates.

## Acceptance criteria

- `scripts/run_simulations.sh` is rerunnable: identical seeds produce identical CSVs.
- Each cell produces a long-format CSV that can be concatenated into the master `artifacts/sim_calibration.csv` consumed by figure F1.
- Sensitivity analyses of §4.7 (varying LKJ shape $\eta$, auxiliary atoms $H$, Bernstein degree grid, $\pi$-prior) are configurable via Hydra without code edits.

## Notes

- 6000 fits × MCMC of 1000 sweeps × tiny $n$ is feasible on a single node in a few days; for the larger $n = 50{,}000$ cells, use VI per the report's recommendation.
- We do *not* need to reproduce the synthetic illustrative figures of `docs/report/figures/generate_figures.py` — those are placeholders. We need to produce real figures from real fits.
