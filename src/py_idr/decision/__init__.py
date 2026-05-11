"""FDR-controlling decision rules built on top of the posterior local idr.

The chain returns a per-feature Z-trace; this package turns that trace
into actionable decisions. The flow is:

1. :func:`py_idr.decision.local_idr.from_mcmc` — convert the Z-trace
   into a posterior local idr $\\widehat{\\mathrm{idr}}_i =
   1 - \\bar Z_i$, a vector of shape $(n,)$ in $[0, 1]$.
2. :func:`py_idr.decision.sun_cai.step_up_threshold` — apply the Sun-Cai
   (2007) step-up rule at level $\\alpha$, returning the rank cutoff,
   the gamma threshold, and the indices of called features.
3. :func:`py_idr.decision.bayes_mfdr.bayes_mfdr_curve` — optional
   alternative: given a sweep of thresholds $\\tau \\in [0, 1]$, return
   the posterior expected number of discoveries and the corresponding
   posterior mFDR at each $\\tau$.

The two decision interfaces (Sun-Cai and Bayes-mFDR) are
*complementary* — Sun-Cai answers "what threshold gives FDR $= \\alpha$?"
while Bayes-mFDR answers "if my threshold is $\\tau$, what is my FDR?"

See plans/06_decision_theory.md and §3.4 of the report.
"""
