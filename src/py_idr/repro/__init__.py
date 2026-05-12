"""Reproducibility utilities — verdict ledger, manifest helpers.

The verdict ledger (plan 13 / §4.10) captures, for each completed run,
whether the system met its pre-registered acceptance criteria. The
report mandates that **every** companion-paper analysis ship with a
JSON ledger so a reader can audit which hypotheses passed and which
failed.

This package provides:

- :class:`py_idr.repro.verdict.VerdictRecord` — one hypothesis test.
- :class:`py_idr.repro.verdict.VerdictLedger` — a bundle of records
  pinned to a single sweep CSV + git SHA + timestamp.
- :func:`py_idr.repro.verdict.build_verdict_ledger` — runs every
  canonical hypothesis evaluator against a sweep DataFrame.

See plans/13_reproducibility.md.
"""

from py_idr.repro.verdict import (
    VerdictLedger,
    VerdictOutcome,
    VerdictRecord,
    build_verdict_ledger,
    evaluate_h1_fdr_control,
    evaluate_h2_pyidr_beats_vanilla_on_s5,
    evaluate_h3_pyidr_beats_maxrank,
)

__all__ = [
    "VerdictLedger",
    "VerdictOutcome",
    "VerdictRecord",
    "build_verdict_ledger",
    "evaluate_h1_fdr_control",
    "evaluate_h2_pyidr_beats_vanilla_on_s5",
    "evaluate_h3_pyidr_beats_maxrank",
]
