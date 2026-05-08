# Plan 06 — Decision Theory: Local idr, Bayes-mFDR, Sun–Cai

**Owner:** W3
**Depends on:** 04_inference_mcmc (consumes traces), 05_inference_vi (optional)
**Implements:** §3.8 ("Local idr, aggregated IDR, and Bayes-mFDR"); Eq. (3.10), (3.11); Sun–Cai step-up rule

## Goal

Given a posterior trace (MCMC) or a variational posterior (VI), produce:

1. The per-feature local idr $\widehat\idr_i = \Pr(Z_i = 0 \mid \text{data})$.
2. The Bayes-mFDR curve as a function of threshold $\gamma$.
3. The Sun–Cai-type plug-in step-up cutoff $\gamma_\alpha$ at the operating level $\alpha$.

## Deliverables

### `src/stick_idr/decision/`

- `local_idr.py`
  - `from_mcmc(trace) -> idr` averages $\Pr(Z_i = 0 \mid \cdot)$ across the post-warmup samples.
  - `from_vi(posterior) -> idr` reads $q(Z_i = 0)$ directly.
- `bayes_mfdr.py`
  - `bayes_mfdr_curve(idr, gammas) -> jnp.ndarray` returns $\E[V(\gamma)]/\E[R(\gamma)]$ at each $\gamma$.
  - `posterior_decomposition(idr_samples, gamma)` returns the Monte-Carlo estimate of the numerator and denominator of Eq. (3.11).
- `sun_cai.py`
  - `step_up_threshold(idr, alpha) -> gamma_alpha` returns the largest $\gamma$ such that the running average $k^{-1}\sum_{j\le k} \widehat\idr_{(j)} \le \alpha$ at $k = R(\gamma)$.
  - `discoveries(idr, gamma) -> indices` returns the rejection set.
- `calibration.py`
  - `idr_vs_realized_fdr(idr, true_Z, alpha_grid)` for use in figure F1 and the calibration tests.

## Tests

- `tests/decision/test_sun_cai_calibration.py` — on a synthetic regime where the truth is known, the realized FDR at $\alpha = 0.05$ is within 1 SE of the nominal level on $\ge 80$ of 100 replications.
- `tests/decision/test_bayes_mfdr_monotone.py` — `bayes_mfdr_curve` is non-decreasing on a $\gamma$-grid (this is operational; not analytic).
- `tests/decision/test_local_idr_marginalization.py` — on a 5-feature toy problem, posterior averaging recovers a brute-force enumeration of $\Pr(Z_i = 0 \mid \text{data})$ to within 1 SE.
- `tests/decision/test_step_up_breakdown.py` — when all $\widehat\idr_i \to 0$ (very confident reproducibility), `step_up_threshold` returns the largest $\gamma$ in the grid; when all $\widehat\idr_i \to 1$, it returns 0.

## Acceptance criteria

- Sun–Cai threshold is computed in $O(n \log n)$ (sort + cumulative mean).
- Vectorized over $\alpha$-grids: `step_up_thresholds(idr, alpha_grid)` returns a vector in one pass.
- All decision functions accept either a posterior trace (xarray DataArray of $Z_i$ samples) or a precomputed $\widehat\idr_i$ vector.

## Notes

- Sun–Cai (2007) shows the step-up rule is the empirical-Bayes plug-in counterpart of the optimal compound-decision rule. The rate-to-mFDR transfer (Theorem 3.3) requires Theorem 3.2 (posterior contraction) plus continuity of the limiting CDF of $\widehat\idr_i$ at the threshold; we test the operational guarantee only (calibration on simulation).
- The `calibration.py` module is the data backbone of figure F1 (idr-vs-realized-FDR calibration) and tables T4–T5 (FDR / power across S1–S5).
