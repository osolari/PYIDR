# Plan 07 — Simulation Studies S1–S5 (+ S5-sparse)

**Owner:** W3
**Depends on:** 02_algebra_and_copulas, 03_marginals, 06_decision_theory (and 04_inference_mcmc for fitting)
**Implements:** §4.1–§4.6 of the revised report; Table 3 (sim design); Figures F1, F2, F3, F4; Tables T4 (sim FDR), T5 (sim power), T6 (sim K-effect).

## Goal

A reproducible simulation harness that can run the full S1–S5 × $K \in \{2,5,10,50\}$ ×
$n \in \{2k, 10k, 50k\}$ factorial design with **100 MC replications per cell**, plus the
**S5-sparse** ROC/PR stress-test variant. Every replication writes a self-describing
truth + posterior bundle so figures, tables, and verdicts can be regenerated without
re-running fits.

## Simulation design (from the revised Table 3)

| Regime | Signal copula | $\pi^*$ | Marginals | Notes |
|--------|---------------|---------|-----------|-------|
| S1 | Gaussian, $\rho=0.85$ | 0.30 | shared Gaussian | baseline |
| S2 | Gaussian, $\rho=0.40$ | 0.30 | shared Gaussian | weak signal |
| S3 | Gaussian, $\rho=0.85$ | 0.10 | heterogeneous Gaussian | sparse signal |
| S4 | Gaussian, $\rho=0.85$ | 0.30 | heterogeneous skewed | marginal mismatch |
| S5 | mixture: $t_5$/Clayton/Gumbel | 0.30 | heterogeneous heavy-tail | heavy-tail extension |
| **S5-sparse** | mixture: $t_5$/Clayton/Gumbel | 0.10 | heterogeneous heavy-tail | planned ROC/PR stress test |

**Placeholder shells in the manuscript** report at `K = 5`, not `K = 4` (the revised Table 4
correction). The simulation harness sweeps the full $K$-grid; the manuscript table only
displays $K = 5$.

## Per-replication artifact bundle (revised §4.1)

For every replication, `simulation.replicates.run_replicate` writes a `bundle.zarr`
under `artifacts/sim/<regime>_K<K>_n<n>/rep_<replicate>/`:

| Key | Contents |
|----|----|
| `truth.Z` | true class labels |
| `truth.z` | true cluster assignments |
| `truth.atoms` | true copula family + parameters per cluster |
| `truth.marginals` | true $F_j$ and the inverse transform $X = F_j^{-1}(U)$ |
| `truth.ranks` | rank-tie summary for the simulated scores |
| `posterior.idr` | per-feature $\widehat\idr_i$ |
| `posterior.k_alpha` | Sun–Cai $\widehat k_\alpha$ at $\alpha \in \{0.01, 0.05, 0.10, 0.20\}$ |
| `posterior.cluster_count` | trace of $T$ across post-warmup samples |
| `posterior.delta_n` | local-idr error scale ($\delta_n$, per plan 06) |
| `diagnostics.rhat` | split-$\widehat R$ for $\pi$, $\sigma$, $\alpha$, $\alpha_F$ |
| `diagnostics.ess` | ESS for the same scalars |
| `diagnostics.divergences` | NUTS divergent-transition count |
| `manifest.json` | seed, git SHA, config hash, walltime, hardware metadata, software versions |

This bundle is the single source of truth for the figures, tables, and verdict ledger;
it is consumed by `figures/*.py` and `tables/*.py`.

## Deliverables

### `src/py_idr/simulation/`

- `scenarios.py`
  - `simulate_S1(rng, K, n, pi_star=0.30, rho=0.85)` etc.
  - `simulate_S5(rng, K, n, pi_star=0.30, mixture_weights=("t5", "clayton", "gumbel"))`
  - `simulate_S5_sparse(rng, K, n)` — same draw mechanism, $\pi^* = 0.10$.
  - Each returns a `SimulationResult(X, U_truth, true_Z, true_z, true_pi, true_atoms, true_marginals, ranks_summary)`.
- `designs.py` — the $5 \times 4 \times 3$ factorial enumerator + the S5-sparse cell at $K = 10$, $n = 10{,}000$ (the §4.4 ROC/PR setting); emits `(regime, K, n, replicate)` tuples.
- `replicates.py` — `run_replicate(regime, K, n, seed)` — fits PY-IDR (MCMC by default; VI for $n \ge 50{,}000$), writes the `bundle.zarr` above, plus a flat row in the long-format CSV with columns `(regime, K, n, replicate, method, alpha, realized_fdr, power, hellinger, delta_n, ess_per_s, walltime_s, rhat_pi, divergences)`.
- `truth.py` — holds the simulated truth and exposes evaluation helpers (Hellinger error, label-invariant cluster recovery, marginal calibration).
- `comparators.py` — vanilla IDR, eCV, nestedIDR, MaRR, ChIP-R, GMCM-AD, BNP-Archimedean wrappers (per the §4.2 list, with author-recommended defaults; subprocess shells for R packages).

### `experiments/sim_S1/` … `sim_S5/`, `sim_S5_sparse/`

Each contains:
- `config.yaml` — Hydra config selecting model and inference defaults.
- `run.py` — argparse wrapper.
- `sbatch.sh` — slurm header.
- `ablations/` — one config per ablation knob (PY vs DP, Gaussian-only vs heavy-tail, learned vs empirical-rank marginals, exchangeable vs factor structures, MCMC vs VI, joint vs pairwise aggregation, auxiliary-atom count $H \in \{1, 3, 5, 10\}$).
- `stress/` — one config per stress test (weak dependence in $c_0$, high tie rate, missing replicates, near-singular correlation, large $K$, marginal misspecification, low reproducible mass, heavy boundary tail dependence).

### `scripts/run_simulations.sh`

```bash
#!/usr/bin/env bash
set -euo pipefail
for regime in S1 S2 S3 S4 S5; do
  for K in 2 5 10 50; do
    for n in 2000 10000 50000; do
      python -m py_idr.simulation.replicates --regime $regime --K $K --n $n --reps 100 \
        --out artifacts/sim/${regime}_K${K}_n${n}/
    done
  done
done

# S5-sparse: K=10, n=10000 only (the §4.4 ROC/PR setting)
python -m py_idr.simulation.replicates --regime S5_sparse --K 10 --n 10000 --reps 100 \
  --out artifacts/sim/S5_sparse_K10_n10000/
```

`scripts/run_ablations.sh` and `scripts/run_stress.sh` follow the same pattern and use
the per-experiment `ablations/*.yaml` and `stress/*.yaml` configs.

## Tests

- `tests/integration/test_sim_S1_smoke.py` — one replicate of S1, K=2, n=200, completes in < 30 s, writes a `bundle.zarr` with the schema above, and produces a valid Sun–Cai threshold.
- `tests/integration/test_sim_S5_sparse_smoke.py` — one replicate of S5-sparse at K=4, n=200, completes and writes the ROC/PR fields.
- `tests/numerical/test_sim_calibration_synthetic.py` — at K=2, $n = 10^4$, the fitted PY-IDR's realized FDR at $\alpha = 0.05$ is within 0.01 of nominal averaged over 50 replicates.

## Acceptance criteria

- `scripts/run_simulations.sh` is rerunnable: identical seeds produce identical CSVs and Zarr bundles.
- Every cell writes a per-replicate `bundle.zarr` with the schema above; `manifest.json` records git SHA + config hash + hardware.
- Ablations and stress tests are configurable via Hydra without code edits.
- Figure / table modules consume only `artifacts/sim/**/bundle.zarr` and the long-format CSV; they never re-run fits.

## Notes

- 6000 fits × MCMC of 1000 sweeps × tiny $n$ is feasible on a single A10 node in a few days; for the larger $n = 50{,}000$ cells, use VI per the report's recommendation. Plan 14 (Docker on a10-dev) sets up the 4-GPU sharded harness.
- We do *not* reproduce the synthetic illustrative figures of `docs/report/figures/generate_figures.py` — those are placeholders. We produce real figures from real fits via the per-replicate bundle.
- The §4.6 "ablations and sensitivity" list and §4.6 "stress tests" list are itemised in the experiments folder so each is a separate Hydra config rather than a code branch. This makes future stress tests cheap to add.
