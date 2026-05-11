"""Evaluation, diagnostics, and reporting.

The single module today is
:mod:`py_idr.eval.diagnostics_report`, which packages arviz's
split-$\\hat R$, ESS, and divergence diagnostics into a single
:class:`DiagnosticsReport` Pydantic schema with a pass / warn / fail
verdict.

Thresholds (configurable via module constants):

- ``RHAT_PASS_THRESHOLD = 1.01`` — split-$\\hat R$ ceiling.
- ``ESS_PASS_THRESHOLD = 400`` — minimum effective sample size, both
  bulk and tail.
- ``DIVERGENCE_FRACTION_THRESHOLD = 0.005`` — at most half a percent
  divergent transitions.
- ``MIN_CHAINS_PASS = 4`` — fewer chains and the report can only emit
  ``warn``, never ``pass`` (because $\\hat R$ on 2 chains is unreliable).

See plans/12_diagnostics.md and the **diagnostics guide** in the
docs site.
"""
