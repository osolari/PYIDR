# Plan 12 — Diagnostics Module

**Owner:** W2 + W3
**Depends on:** 04_inference_mcmc, 05_inference_vi, 06_decision_theory
**Implements:** §3.3.1 (sampler diagnostics list), Remark A.6 (theory-to-code diagnostics)
of the revised report.

## Status snapshot (W8.1)

| Deliverable | Module | Status |
|-------------|--------|--------|
| `DiagnosticsReport` pydantic schema | `eval/diagnostics_report.DiagnosticsReport` | DONE |
| split-$\hat R$ + ESS extraction | via `arviz.summary` in `build_diagnostics_report` | DONE |
| Divergent-transition fraction | `sweep_simple` / `multi_cluster_sweep` return `(state, {"num_divergences": int})`; `run_chain_*` aggregate the count across post-warmup sweeps and surface it on `ChainResult*.num_divergent_transitions`. | DONE (W8.11) |
| Cluster-count trace | `ChainResultMulti.T_trace` (multi-cluster path) | DONE |
| Pass/warn/fail verdict + thresholds | constants `RHAT_PASS_THRESHOLD=1.01`, `ESS_PASS_THRESHOLD=400`, `DIVERGENCE_FRACTION_THRESHOLD=0.005`, `MIN_CHAINS_PASS=4` | DONE |
| Label-switching postprocessor | Stephens (2000) relabeller | TODO |
| Hellinger to truth (theoretical contraction probe) | `eval/hellinger.{hellinger_gaussian_copula, hellinger_gaussian_copula_mc}` | DONE (closed form for exchangeable Gaussian copulas; F5 figure consumes it) |

## Goal

A first-class diagnostics layer that maps each theorem assumption to a measured signal
the implementation actually emits. Per the revised manuscript, every simulation and
real-data run is required to produce a machine-readable diagnostics report
(handoff IMPL-09).

## Required diagnostics (revised §3.3.1)

The revised report itemises the diagnostics that the sampler must emit. Every fit
writes the following into `runs/<run_id>/diagnostics/`:

| Diagnostic | Acceptance threshold |
|---|---|
| Deterministic seed ledger | Always produced; archived alongside the trace. |
| Rank-tie summary | Per-replicate tie count + tie-handling policy used (`mid_rank` / `random` / `truncate`). |
| Overdispersed chain starts | $\ge 4$ chains by default, with starts drawn from the prior or a wide kernel. |
| Split-$\widehat R$ for reported scalars ($\pi$, $\sigma$, $\alpha$, $\alpha_F$) | $\le 1.01$ |
| ESS for reported scalars | $\ge 400$ |
| Cluster-count $T$ trace | Visually plateaued after burn-in; no monotone drift. |
| Posterior label-invariant cluster summary | Wasserstein on the cluster-correlation distribution, robust to label switching. |
| NUTS divergent-transition counts | $< 0.5\%$ of post-warmup samples; $> 0.5\%$ raises a warning. |
| Sensitivity to auxiliary-atom count $H$ | At $H \in \{1, 3, 5, 10\}$, the cluster-count posterior should not vary materially. |

## Theory-to-code diagnostics (Remark A.6)

| Theory item | Diagnostic |
|---|---|
| PLOD / positive-correlation (Thm 3.1, Assumption 3.2) | Per-cluster `algebra.plod.holds()` + `plod_with_positive_pairwise()` checks; flag any cluster failing either. |
| Sklar step regularity (Lemma A.1) | Marginal PIT calibration plot — empirical $F_j(X_{i,j})$ should be approximately uniform. |
| PY rare-cluster behaviour | Cluster-count trace, cluster-occupancy histogram, expected #clusters vs sample size compared to $\alpha n^\sigma$. |
| Bounded-smooth contraction scope (Remark 3.6) | Boundary-singular cluster flag — if any cluster's atomic type is in $\{t_5, \text{Clayton}, \text{Gumbel}\}$ with parameters near the singularity, mark the contraction diagnostic as "out-of-scope" for that cluster. |
| Bayes-mFDR threshold transfer (Thm 3.3, Assumption 3.5) | idr-vs-realised-FDR calibration curve; $\delta_n$ from the split-chain estimator. |
| Harris ergodicity (Thm 3.4) | Trace plots, ESS, R-hat, autocorrelation, cluster-switching diagnostic. Geometric rate is **not** measured; the diagnostic notes the finite parameterisation in use. |

## Deliverables

### `src/py_idr/eval/diagnostics_report.py`

Single entry point: `build_diagnostics_report(run_dir: Path) -> DiagnosticsReport`.

The `DiagnosticsReport` is a pydantic model serialisable to `diagnostics.json`:

```python
class DiagnosticsReport(BaseModel):
    run_id: str
    git_sha: str
    config_hash: str
    rhat: dict[str, float]                     # {"pi": 1.001, ...}
    ess: dict[str, float]
    divergent_transitions: int
    cluster_count_trace: list[int]
    cluster_count_posterior_summary: dict[str, float]   # median, 5%, 95%
    label_invariant_summary: dict[str, float]   # Wasserstein on cluster correlations
    plod_per_cluster: list[bool]
    plod_positive_pairwise_per_cluster: list[bool]
    pit_calibration: list[float]                # KS stat per replicate
    delta_n: float
    h_sensitivity: dict[int, float]             # H -> posterior mean #clusters
    seed_ledger_path: str
    rank_tie_summary: dict[str, int]
    verdict: Literal["pass", "warn", "fail"]
    warnings: list[str]
```

### `src/py_idr/eval/plots.py`

- `traceplot(idata, params)` — split-chain traceplot via arviz.
- `cluster_occupancy_plot(idata)` — heatmap of cluster index vs iteration with a deterministic relabeling.
- `pit_calibration_plot(U_observed)` — empirical PIT curves per replicate.
- `idr_vs_fdr_plot(idr, true_Z, alpha_grid)` — calibration curve with 95% Monte Carlo bands.

### Tests

- `tests/diagnostics/test_report_schema.py` — `DiagnosticsReport.model_validate({...})` round-trips.
- `tests/diagnostics/test_rhat_threshold.py` — synthetic two-chain trace where the first chain is shifted; report flags `rhat > 1.01` as a warning.
- `tests/diagnostics/test_h_sensitivity.py` — synthetic chains at $H \in \{1, 3, 5, 10\}$ with identical posterior produce identical cluster-count summaries; deliberate degraded $H = 1$ chain is flagged.
- `tests/diagnostics/test_label_invariant.py` — cluster-correlation Wasserstein is invariant under cluster relabelling.

## Acceptance criteria

- Every fit writes a `diagnostics.json` validating against `DiagnosticsReport`.
- A run with `verdict == "fail"` cannot pass to the figure / table / verdict ledger
  pipeline (the make_all_figures script blocks).
- The §3.3.1 list is fully covered; no diagnostic in the revised manuscript is missing.

## Notes

- Diagnostics live in `eval/` rather than `inference/mcmc/diagnostics.py` because they
  are used by both MCMC and VI runs and by the verdict ledger.
- The label-invariant summary is the deterministic relabelling helper from Stephens (2000)
  applied per-iteration, then the cluster-correlation distribution is summarised via
  Wasserstein. We do not attempt full label-switching repair on the trace itself — the
  diagnostic is enough for pass/warn/fail decisions.
