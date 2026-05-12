# Plan 06 — Decision Theory: Local idr, Bayes-mFDR, Sun–Cai

**Owner:** W3
**Depends on:** 04_inference_mcmc (consumes traces), 05_inference_vi (optional)
**Implements:** §3.3.3 of the revised report; Eqs. (3.9)–(3.11) (revised numbering); Sun–Cai step-up rule.

## Status snapshot (W8.1)

| Deliverable | Module | Status |
|-------------|--------|--------|
| Posterior local idr (Eq. 3.9) | `decision/local_idr.from_mcmc` (and `from_vi`) | DONE |
| Bayes-mFDR curve (Eq. 3.10 with `max(1, E[R])` denominator) | `decision/bayes_mfdr.bayes_mfdr_curve` | DONE |
| Sun-Cai step-up (Eq. 3.11) | `decision/sun_cai.step_up_threshold` | DONE |
| Realised-FDR + power evaluator | `simulation/evaluation.evaluate_recovery` | DONE |
| ROC / PR curves + AUC / AP scalars | `decision/roc_pr` (W7.2) | DONE |

**Cross-checks (W8.1):**

- Sun-Cai hand-computed 8-point example reproduces $k_\alpha = 4$, $\gamma_\alpha = 0.10$, selected = [0, 1, 2, 3] at $\alpha = 0.05$.
- Permutation invariance verified.
- Empty-rejection edge case ($\alpha < \min(\mathrm{idr})$) returns $k_\alpha = 0$, $\gamma = \mathrm{NaN}$.

## Goal

Given a posterior trace (MCMC) or a variational posterior (VI), produce:

1. The per-feature local idr $\widehat\idr_i = \Pr(Z_i = 0 \mid \text{data})$ (Eq. 3.9).
2. The Bayes-mFDR curve at threshold $\gamma$ with the **revised denominator**
   $\max\{1, \E[R(\gamma) \mid \mathrm{data}]\}$ (Eq. 3.10), so the ratio is well-defined
   even when no feature crosses the threshold.
3. The Sun–Cai-type plug-in **step-up** rule (Eq. 3.11):
   $\widehat k_\alpha = \max\{k \ge 1 : \tfrac{1}{k}\sum_{j=1}^k \widehat\idr_{(j)} \le \alpha\}$
   with **no rejections if the set is empty**.

## Revised formulas (Phase Three pass)

The denominator change is small but consequential — Theorem 3.3 in the revised report
requires the denominator to be bounded away from zero at the target threshold
(Assumption 3.5), and the operational form `max(1, …)` makes the empty-rejection branch
explicit:

$$\mathrm{Bayes\text{-}mFDR}(\gamma) =
   \frac{\E[V(\gamma) \mid \mathrm{data}]}{\max\{1, \E[R(\gamma) \mid \mathrm{data}]\}},
   \qquad V(\gamma) = \sum_i (1-Z_i)\bm{1}\{\widehat\idr_i < \gamma\},
   \qquad R(\gamma) = \sum_i \bm{1}\{\widehat\idr_i < \gamma\}.$$

Theorem 3.3 gives an $\alpha + O(\delta_n)$ bound where $\delta_n$ is a *local-idr*
error scale (uniform or threshold-local), not the global Hellinger contraction rate
$\epsilon_n$. When only $L^1$ local-idr consistency is available, Theorem 3.3 yields
plug-in convergence to the oracle mFDR but no sharp uniform rate. The decision module
exposes $\delta_n$ as a first-class diagnostic so the verdict ledger can record it.

## Deliverables

### `src/py_idr/decision/`

- `local_idr.py`
  - `from_mcmc(trace) -> idr` averages $\Pr(Z_i = 0 \mid \cdot)$ across the post-warmup samples.
  - `from_vi(posterior) -> idr` reads $q(Z_i = 0)$ directly.
  - `local_idr_error_scale(traces) -> delta_n` — Monte Carlo estimate of the largest deviation between feature-wise posterior idr estimates across chains or split-train/split-test folds. Used by the Theorem 3.3 diagnostic; populates the `δ_n` column of the verdict ledger.
- `bayes_mfdr.py`
  - `bayes_mfdr_curve(idr, gammas) -> jnp.ndarray` returns
    $\E[V(\gamma)] / \max(1, \E[R(\gamma)])$ at each $\gamma$.
  - `posterior_decomposition(idr_samples, gamma)` returns the Monte-Carlo numerator and denominator of Eq. (3.10) separately so the caller can audit the ratio.
- `sun_cai.py`
  - `step_up_threshold(idr, alpha) -> StepUpResult` returning a small dataclass:
    - `k_alpha: int` (number of rejections; **0 if the set is empty**)
    - `gamma_alpha: float` (the largest threshold satisfying Eq. 3.11; `+inf` symbolically and `nan` numerically if $k_\alpha = 0$)
    - `selected: jnp.ndarray` (indices, possibly empty)
  - `discoveries(idr, gamma) -> indices` — convenience wrapper over the threshold output.
- `calibration.py`
  - `idr_vs_realized_fdr(idr, true_Z, alpha_grid)` for figure F1 and the calibration tests; emits a long-format DataFrame consumable by `figures/f1_idr_calibration.py`.

## Tests

- `tests/decision/test_sun_cai_calibration.py` — on a synthetic regime where the truth is known, the realized FDR at $\alpha = 0.05$ is within 1 SE of the nominal level on $\ge 80$ of 100 replications.
- `tests/decision/test_sun_cai_empty_rejection.py` (new) — when every $\widehat\idr_i > \alpha$, the result has `k_alpha == 0` and an empty `selected` array (not `NaN` or an exception).
- `tests/decision/test_bayes_mfdr_max_one.py` (new) — for very small $\gamma$ (no rejections), `bayes_mfdr_curve(...)` returns 0 (since the numerator is 0) rather than 0/0; for $\gamma$ values that yield $R = 1$, the denominator stays at 1.
- `tests/decision/test_bayes_mfdr_monotone.py` — `bayes_mfdr_curve` is non-decreasing on a $\gamma$-grid (operational; not analytic).
- `tests/decision/test_local_idr_marginalization.py` — on a 5-feature toy problem, posterior averaging recovers a brute-force enumeration of $\Pr(Z_i = 0 \mid \text{data})$ to within 1 SE.
- `tests/decision/test_local_idr_error_scale.py` (new) — on a synthetic problem with two split chains drawn from the same posterior, $\delta_n \to 0$ as the chains lengthen.

## Acceptance criteria

- Sun–Cai threshold is computed in $O(n \log n)$ (sort + cumulative mean).
- Vectorized over $\alpha$-grids: `step_up_thresholds(idr, alpha_grid)` returns a vector of `StepUpResult` in one pass.
- All decision functions accept either a posterior trace (xarray DataArray of $Z_i$ samples) or a precomputed $\widehat\idr_i$ vector.
- Empty-rejection set is the *only* return path that does not raise; every other failure (NaN inputs, length mismatch) raises explicitly so the calling pipeline can fail loudly.

## Notes

- Sun–Cai (2007) shows the step-up rule is the empirical-Bayes plug-in counterpart of the optimal compound-decision rule. Theorem 3.3 of the revised report transfers the oracle threshold to the plug-in rule under threshold regularity (Assumption 3.5) and a uniform/threshold-local local-idr error bound. We test the *operational* calibration (the simulation realizes nominal FDR) and we expose $\delta_n$ as a measured quantity; we do not test the rate transfer itself.
- The `calibration.py` module is the data backbone of figure F1 (idr-vs-realized-FDR calibration) and tables T4–T5 (FDR / power across S1–S5).
- The `max(1, …)` denominator is *strictly* equivalent to the original definition once $R \ge 1$, so it does not change reported values for any non-empty rejection set. It only fixes the $0/0$ edge case.
