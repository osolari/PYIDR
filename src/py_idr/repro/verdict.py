r"""Verdict ledger — pre-registered hypothesis tests against a sweep CSV.

Per §4.10 of the revised report, every completed analysis ships with a
JSON record that says, for each pre-registered hypothesis,
{pass, warn, fail} plus the evidence that produced the verdict. The
ledger is the single source of truth a reader uses to audit which
acceptance criteria the run met.

This module provides:

- :class:`VerdictRecord` — one hypothesis test (id, statement, outcome,
  evidence dict).
- :class:`VerdictLedger` — pydantic bundle: records + run metadata
  (run_id, git_sha, source CSV, timestamp).
- :func:`evaluate_h1_fdr_control`, :func:`evaluate_h2_*`,
  :func:`evaluate_h3_*` — three canonical evaluators for the
  pre-registered hypotheses below.
- :func:`build_verdict_ledger` — runs every evaluator against a sweep
  DataFrame and returns the assembled ledger.

Pre-registered hypotheses
-------------------------

**H1 — FDR control.** The PY-IDR + Sun-Cai pipeline controls the
realised FDR at every nominal $\alpha$, on every regime, within a
tolerance ``fdr_slack`` (default 0.02 absolute — accommodates finite-
sample MC noise on tiny test runs).

**H2 — PY-IDR beats Vanilla IDR on S5.** On the S5 heavy-tail mixture,
PY-IDR's multi-cluster atom should have strictly higher mean power
than the misspecified single-Gaussian Vanilla IDR. The verdict
``passes`` iff ``power(py-idr, S5) > power(vanilla-idr, S5) + delta``
for some small ``delta`` (default 0.0; ties fail).

**H3 — PY-IDR beats MaxRank on S1.** On the baseline S1 regime,
PY-IDR's parametric mixture should out-discriminate the
non-parametric rank baseline. Verdict ``passes`` iff
``power(py-idr, S1) > power(maxrank, S1) + delta``.

Each evaluator is *forgiving* about missing methods: if PY-IDR rows
are absent the verdict comes back ``insufficient_data`` rather than
``fail``. That way a partial sweep (one method only) still produces
a sensible ledger.

See plans/13_reproducibility.md.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from py_idr.utils.run import detect_git_sha, new_run_id, now_utc

# Pretty Python enums for the verdict outcomes; pydantic stringifies
# them as JSON strings.
VerdictOutcome = Literal["pass", "warn", "fail", "insufficient_data"]


class VerdictRecord(BaseModel):
    """One hypothesis test result.

    Attributes
    ----------
    hypothesis_id
        Short tag (``"H1"``, ``"H2"``, ...).
    statement
        Human-readable hypothesis text. Frozen at ledger-build time so
        a downstream reader sees exactly what was claimed.
    outcome
        ``"pass"``, ``"warn"``, ``"fail"``, or ``"insufficient_data"``.
    evidence
        Free-form dict of the numbers behind the verdict (e.g.,
        per-regime mean realised FDR, per-method mean power). The
        downstream reader can audit the verdict from the same evidence.
    rationale
        One-line explanation of the verdict.
    """

    model_config = ConfigDict(extra="forbid")

    hypothesis_id: str = Field(..., min_length=1)
    statement: str = Field(..., min_length=1)
    outcome: VerdictOutcome
    evidence: dict[str, Any] = Field(default_factory=dict)
    rationale: str = Field(default="")


class VerdictLedger(BaseModel):
    """Bundle of verdict records pinned to a single sweep CSV.

    Attributes
    ----------
    run_id
        Verdict-build run id (independent of the sweep's own run_id).
        Built via :func:`py_idr.utils.run.new_run_id`.
    sweep_csv
        Path to the source sweep CSV. Recorded so a reader can find
        the underlying data.
    git_sha
        Git SHA at verdict-build time. May differ from the sweep's
        git_sha if the verdict definitions evolved between sweep and
        verdict — both are recorded so the reader can detect drift.
    built_at
        ISO-8601 UTC timestamp of verdict construction.
    records
        Per-hypothesis :class:`VerdictRecord` list.
    """

    model_config = ConfigDict(extra="forbid")

    run_id: str
    sweep_csv: str
    git_sha: str | None
    built_at: datetime
    records: list[VerdictRecord]

    def summary(self) -> dict[str, int]:
        """Return a ``{outcome: count}`` summary for quick CLI display."""
        counts: dict[str, int] = {
            "pass": 0,
            "warn": 0,
            "fail": 0,
            "insufficient_data": 0,
        }
        for r in self.records:
            counts[r.outcome] += 1
        return counts


# ---- canonical hypothesis evaluators -------------------------------------------------


_METHOD_PY_IDR_T1 = "PY-IDR (MCMC, T=1)"
_METHOD_PY_IDR_TGT1 = "PY-IDR (MCMC, T>1)"
_METHOD_VANILLA = "Vanilla IDR (pairwise)"
_METHOD_MAXRANK = "MaxRank"


def evaluate_h1_fdr_control(
    df: pd.DataFrame,
    *,
    fdr_slack: float = 0.02,
    methods: tuple[str, ...] = (_METHOD_PY_IDR_T1, _METHOD_PY_IDR_TGT1),
) -> VerdictRecord:
    r"""H1 — realised FDR is controlled at every (method, regime, alpha).

    For every row whose method is in ``methods``, check
    ``realized_fdr <= alpha + fdr_slack`` (a hard pass) or
    ``realized_fdr <= alpha + 2 * fdr_slack`` (a warn). Any other row
    fails.

    Aggregation across replicates: we check the **mean** realised FDR
    per (method, regime, alpha) cell. Mean is more stable than the
    per-replicate max but still catches systematic miscalibration.
    """
    required = {"method", "regime", "alpha", "realized_fdr"}
    missing = required - set(df.columns)
    if missing:
        return VerdictRecord(
            hypothesis_id="H1",
            statement="Realised FDR is controlled at every (method, regime, alpha).",
            outcome="insufficient_data",
            evidence={"missing_columns": sorted(missing)},
            rationale=f"DataFrame missing required columns: {sorted(missing)}",
        )

    sub = df[df["method"].isin(methods)]
    if sub.empty:
        return VerdictRecord(
            hypothesis_id="H1",
            statement="Realised FDR is controlled at every (method, regime, alpha).",
            outcome="insufficient_data",
            evidence={"methods_searched": list(methods)},
            rationale="No rows with a PY-IDR method label.",
        )

    grouped = sub.groupby(["method", "regime", "alpha"])["realized_fdr"].mean().reset_index()
    grouped["excess"] = grouped["realized_fdr"] - grouped["alpha"]

    max_excess = float(grouped["excess"].max())
    if max_excess <= fdr_slack:
        outcome: VerdictOutcome = "pass"
    elif max_excess <= 2.0 * fdr_slack:
        outcome = "warn"
    else:
        outcome = "fail"

    # Capture the worst cell for the rationale.
    worst = grouped.loc[grouped["excess"].idxmax()]
    return VerdictRecord(
        hypothesis_id="H1",
        statement=(
            f"Realised FDR is controlled at every (method, regime, alpha) "
            f"within fdr_slack = {fdr_slack:.3f}."
        ),
        outcome=outcome,
        evidence={
            "fdr_slack": fdr_slack,
            "max_excess": max_excess,
            "worst_cell": {
                "method": str(worst["method"]),
                "regime": str(worst["regime"]),
                "alpha": float(worst["alpha"]),
                "realized_fdr": float(worst["realized_fdr"]),
            },
            "n_cells": int(len(grouped)),
        },
        rationale=(
            f"Worst cell: realised_fdr = {worst['realized_fdr']:.4f} at "
            f"alpha = {worst['alpha']:.3f} (method = {worst['method']}, "
            f"regime = {worst['regime']}); excess = {max_excess:+.4f}."
        ),
    )


def _compare_power(
    df: pd.DataFrame,
    *,
    regime: str,
    method_a: str,
    method_b: str,
    delta: float,
    alpha_filter: float | None,
) -> tuple[VerdictOutcome, dict[str, Any], str]:
    """Shared "method_a beats method_b on regime by delta in mean power" check."""
    required = {"method", "regime", "alpha", "power"}
    missing = required - set(df.columns)
    if missing:
        return (
            "insufficient_data",
            {"missing_columns": sorted(missing)},
            f"DataFrame missing required columns: {sorted(missing)}",
        )

    sub = df[df["regime"] == regime]
    if alpha_filter is not None:
        sub = sub[sub["alpha"] == alpha_filter]
    if sub.empty:
        return (
            "insufficient_data",
            {"regime": regime, "alpha_filter": alpha_filter},
            f"No rows for regime = {regime!r} (and alpha filter).",
        )

    has_a = (sub["method"] == method_a).any()
    has_b = (sub["method"] == method_b).any()
    if not (has_a and has_b):
        return (
            "insufficient_data",
            {
                "method_a_present": bool(has_a),
                "method_b_present": bool(has_b),
                "method_a": method_a,
                "method_b": method_b,
            },
            f"Missing one of the methods in the sweep ({method_a!r}, {method_b!r}).",
        )

    mean_a = float(sub[sub["method"] == method_a]["power"].mean())
    mean_b = float(sub[sub["method"] == method_b]["power"].mean())
    diff = mean_a - mean_b
    outcome: VerdictOutcome
    if diff > delta:
        outcome = "pass"
    elif diff >= 0.0:
        outcome = "warn"
    else:
        outcome = "fail"
    evidence = {
        "regime": regime,
        "alpha_filter": alpha_filter,
        "method_a": method_a,
        "method_b": method_b,
        "mean_power_a": mean_a,
        "mean_power_b": mean_b,
        "diff": diff,
        "delta": delta,
    }
    rationale = (
        f"Mean power on {regime}: {method_a} = {mean_a:.4f}, "
        f"{method_b} = {mean_b:.4f}; diff = {diff:+.4f} (delta = {delta})."
    )
    return outcome, evidence, rationale


def evaluate_h2_pyidr_beats_vanilla_on_s5(
    df: pd.DataFrame,
    *,
    delta: float = 0.0,
    alpha_filter: float | None = 0.05,
) -> VerdictRecord:
    r"""H2 — on S5, PY-IDR (multi-cluster) out-powers Vanilla IDR.

    Compares ``PY-IDR (MCMC, T>1)`` against
    ``Vanilla IDR (pairwise)`` on regime S5 at ``alpha_filter`` (default
    0.05). Returns ``pass`` iff PY-IDR's mean power exceeds Vanilla
    IDR's by more than ``delta``.
    """
    outcome, evidence, rationale = _compare_power(
        df,
        regime="S5",
        method_a=_METHOD_PY_IDR_TGT1,
        method_b=_METHOD_VANILLA,
        delta=delta,
        alpha_filter=alpha_filter,
    )
    return VerdictRecord(
        hypothesis_id="H2",
        statement=(
            f"On S5, PY-IDR (MCMC, T>1) has higher mean power than "
            f"Vanilla IDR (pairwise) by more than delta = {delta}."
        ),
        outcome=outcome,
        evidence=evidence,
        rationale=rationale,
    )


def evaluate_h3_pyidr_beats_maxrank(
    df: pd.DataFrame,
    *,
    regime: str = "S1",
    delta: float = 0.0,
    alpha_filter: float | None = 0.05,
) -> VerdictRecord:
    r"""H3 — on the baseline regime, PY-IDR out-discriminates MaxRank.

    Compares ``PY-IDR (MCMC, T=1)`` against ``MaxRank`` on the given
    regime (default S1) at ``alpha_filter``.
    """
    outcome, evidence, rationale = _compare_power(
        df,
        regime=regime,
        method_a=_METHOD_PY_IDR_T1,
        method_b=_METHOD_MAXRANK,
        delta=delta,
        alpha_filter=alpha_filter,
    )
    return VerdictRecord(
        hypothesis_id="H3",
        statement=(
            f"On {regime}, PY-IDR (MCMC, T=1) has higher mean power than "
            f"MaxRank by more than delta = {delta}."
        ),
        outcome=outcome,
        evidence=evidence,
        rationale=rationale,
    )


def build_verdict_ledger(
    df: pd.DataFrame,
    *,
    sweep_csv: str | Path,
    run_id: str | None = None,
    git_sha: str | None = None,
    fdr_slack: float = 0.02,
) -> VerdictLedger:
    """Assemble a :class:`VerdictLedger` by running all canonical evaluators.

    Parameters
    ----------
    df
        Long-format sweep DataFrame.
    sweep_csv
        Path to the underlying CSV (recorded in the ledger).
    run_id
        Verdict-build run id; defaults to a fresh
        :func:`py_idr.utils.run.new_run_id` call.
    git_sha
        Git SHA at build time. Defaults to
        :func:`py_idr.utils.run.detect_git_sha` (returns ``None`` if
        not inside a git checkout).
    fdr_slack
        Forwarded to :func:`evaluate_h1_fdr_control`.
    """
    records = [
        evaluate_h1_fdr_control(df, fdr_slack=fdr_slack),
        evaluate_h2_pyidr_beats_vanilla_on_s5(df),
        evaluate_h3_pyidr_beats_maxrank(df),
    ]
    return VerdictLedger(
        run_id=run_id or new_run_id(),
        sweep_csv=str(sweep_csv),
        git_sha=git_sha if git_sha is not None else detect_git_sha(),
        built_at=now_utc(),
        records=records,
    )
