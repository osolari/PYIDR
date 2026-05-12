"""Tests for the verdict-ledger machinery."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from py_idr.repro.verdict import (
    VerdictLedger,
    VerdictRecord,
    build_verdict_ledger,
    evaluate_h1_fdr_control,
    evaluate_h2_pyidr_beats_vanilla_on_s5,
    evaluate_h3_pyidr_beats_maxrank,
)


def _make_row(
    *,
    method: str,
    regime: str = "S1",
    alpha: float = 0.05,
    realized_fdr: float = 0.04,
    power: float = 0.50,
    seed: int = 0,
) -> dict[str, object]:
    """Build one sweep-CSV row with defaults for the fields the evaluators read."""
    return {
        "regime": regime,
        "K": 2,
        "n": 100,
        "replicate_seed": seed,
        "method": method,
        "alpha": alpha,
        "pi_true": 0.30,
        "pi_posterior_mean": 0.32,
        "rho_true": 0.85,
        "rho_posterior_mean": 0.83,
        "T_posterior_mean": float("nan"),
        "k_alpha": 30,
        "realized_fdr": realized_fdr,
        "power": power,
        "num_chains": 2,
        "num_samples_per_chain": 50,
        "walltime_s": 1.0,
    }


# ---- H1 FDR control -----------------------------------------------------------------


@pytest.mark.unit
def test_h1_passes_when_realized_fdr_within_slack() -> None:
    """Mean realised FDR within fdr_slack of alpha → pass."""
    rows = [
        _make_row(method="PY-IDR (MCMC, T=1)", alpha=0.05, realized_fdr=0.04 + 0.01 * i)
        for i in range(5)
    ]
    df = pd.DataFrame(rows)
    rec = evaluate_h1_fdr_control(df, fdr_slack=0.02)
    # Mean is 0.06, alpha=0.05, excess=0.01 <= slack=0.02 → pass.
    assert rec.outcome == "pass"
    assert "max_excess" in rec.evidence


@pytest.mark.unit
def test_h1_warns_at_2x_slack() -> None:
    """Excess between slack and 2x slack → warn."""
    rows = [_make_row(method="PY-IDR (MCMC, T=1)", realized_fdr=0.085, seed=i) for i in range(3)]
    df = pd.DataFrame(rows)
    rec = evaluate_h1_fdr_control(df, fdr_slack=0.02)
    # Mean = 0.085, alpha=0.05, excess=0.035, in (0.02, 0.04] → warn.
    assert rec.outcome == "warn"


@pytest.mark.unit
def test_h1_fails_when_far_above_alpha() -> None:
    """Excess > 2x slack → fail."""
    rows = [_make_row(method="PY-IDR (MCMC, T=1)", realized_fdr=0.20, seed=i) for i in range(3)]
    df = pd.DataFrame(rows)
    rec = evaluate_h1_fdr_control(df, fdr_slack=0.02)
    assert rec.outcome == "fail"


@pytest.mark.unit
def test_h1_insufficient_data_with_no_pyidr_rows() -> None:
    """Sweep with only competitor rows → insufficient_data, not fail."""
    df = pd.DataFrame([_make_row(method="MaxRank")])
    rec = evaluate_h1_fdr_control(df)
    assert rec.outcome == "insufficient_data"


@pytest.mark.unit
def test_h1_insufficient_data_with_missing_columns() -> None:
    """Missing columns → insufficient_data with the missing list in evidence."""
    df = pd.DataFrame({"foo": [1]})
    rec = evaluate_h1_fdr_control(df)
    assert rec.outcome == "insufficient_data"
    assert "missing_columns" in rec.evidence


# ---- H2 PY-IDR beats Vanilla IDR on S5 ---------------------------------------------


@pytest.mark.unit
def test_h2_passes_when_pyidr_dominates_on_s5() -> None:
    """PY-IDR (T>1) mean power > Vanilla IDR mean power by > delta → pass."""
    rows = []
    for i in range(5):
        rows.append(_make_row(method="PY-IDR (MCMC, T>1)", regime="S5", power=0.75, seed=i))
        rows.append(_make_row(method="Vanilla IDR (pairwise)", regime="S5", power=0.50, seed=i))
    rec = evaluate_h2_pyidr_beats_vanilla_on_s5(pd.DataFrame(rows), delta=0.0)
    assert rec.outcome == "pass"
    assert rec.evidence["mean_power_a"] > rec.evidence["mean_power_b"]


@pytest.mark.unit
def test_h2_fails_when_vanilla_wins() -> None:
    """Vanilla IDR ahead of PY-IDR → fail."""
    rows = []
    for i in range(5):
        rows.append(_make_row(method="PY-IDR (MCMC, T>1)", regime="S5", power=0.40, seed=i))
        rows.append(_make_row(method="Vanilla IDR (pairwise)", regime="S5", power=0.60, seed=i))
    rec = evaluate_h2_pyidr_beats_vanilla_on_s5(pd.DataFrame(rows))
    assert rec.outcome == "fail"


@pytest.mark.unit
def test_h2_warns_on_tie() -> None:
    """Methods exactly tied (diff >= 0 but < delta=0) → warn."""
    rows = []
    for i in range(5):
        rows.append(_make_row(method="PY-IDR (MCMC, T>1)", regime="S5", power=0.60, seed=i))
        rows.append(_make_row(method="Vanilla IDR (pairwise)", regime="S5", power=0.60, seed=i))
    rec = evaluate_h2_pyidr_beats_vanilla_on_s5(pd.DataFrame(rows), delta=0.0)
    # diff == 0.0 which is not > delta(0) → warn.
    assert rec.outcome == "warn"


@pytest.mark.unit
def test_h2_insufficient_data_when_one_method_missing() -> None:
    """One of the two methods absent on S5 → insufficient_data."""
    rows = [_make_row(method="PY-IDR (MCMC, T>1)", regime="S5", power=0.7)]
    rec = evaluate_h2_pyidr_beats_vanilla_on_s5(pd.DataFrame(rows))
    assert rec.outcome == "insufficient_data"


# ---- H3 PY-IDR beats MaxRank ---------------------------------------------------------


@pytest.mark.unit
def test_h3_passes_when_pyidr_dominates_on_s1() -> None:
    """PY-IDR (T=1) mean power > MaxRank mean power → pass on S1."""
    rows = []
    for i in range(5):
        rows.append(_make_row(method="PY-IDR (MCMC, T=1)", regime="S1", power=0.80, seed=i))
        rows.append(_make_row(method="MaxRank", regime="S1", power=0.50, seed=i))
    rec = evaluate_h3_pyidr_beats_maxrank(pd.DataFrame(rows))
    assert rec.outcome == "pass"


@pytest.mark.unit
def test_h3_alpha_filter_respected() -> None:
    """alpha_filter narrows to one alpha row before comparing."""
    rows = []
    # At alpha=0.05, PY-IDR > MaxRank.
    rows.append(_make_row(method="PY-IDR (MCMC, T=1)", regime="S1", alpha=0.05, power=0.80))
    rows.append(_make_row(method="MaxRank", regime="S1", alpha=0.05, power=0.40))
    # At alpha=0.20, the orderings are reversed — but we filter to 0.05.
    rows.append(_make_row(method="PY-IDR (MCMC, T=1)", regime="S1", alpha=0.20, power=0.10))
    rows.append(_make_row(method="MaxRank", regime="S1", alpha=0.20, power=0.95))
    rec = evaluate_h3_pyidr_beats_maxrank(pd.DataFrame(rows), alpha_filter=0.05)
    assert rec.outcome == "pass"


# ---- build_verdict_ledger / VerdictLedger -------------------------------------------


@pytest.mark.unit
def test_build_verdict_ledger_runs_all_three_evaluators(tmp_path: Path) -> None:
    """The full builder returns a ledger with three records and the expected ids."""
    rows = []
    for i in range(3):
        rows.append(_make_row(method="PY-IDR (MCMC, T=1)", regime="S1", power=0.8, seed=i))
        rows.append(_make_row(method="MaxRank", regime="S1", power=0.4, seed=i))
        rows.append(_make_row(method="PY-IDR (MCMC, T>1)", regime="S5", power=0.7, seed=i))
        rows.append(_make_row(method="Vanilla IDR (pairwise)", regime="S5", power=0.5, seed=i))
    df = pd.DataFrame(rows)
    ledger = build_verdict_ledger(df, sweep_csv=tmp_path / "sweep.csv")
    assert isinstance(ledger, VerdictLedger)
    assert [r.hypothesis_id for r in ledger.records] == ["H1", "H2", "H3"]


@pytest.mark.unit
def test_verdict_ledger_round_trips_json(tmp_path: Path) -> None:
    """Ledger serialises to JSON and parses back to itself."""
    df = pd.DataFrame([_make_row(method="PY-IDR (MCMC, T=1)")])
    ledger = build_verdict_ledger(df, sweep_csv=tmp_path / "sweep.csv")
    blob = ledger.model_dump_json()
    ledger_back = VerdictLedger.model_validate_json(blob)
    assert ledger_back == ledger


@pytest.mark.unit
def test_verdict_ledger_summary_counts() -> None:
    """`summary()` returns one count per outcome bucket."""
    rec_pass = VerdictRecord(hypothesis_id="HA", statement="x", outcome="pass")
    rec_fail = VerdictRecord(hypothesis_id="HB", statement="x", outcome="fail")
    rec_warn = VerdictRecord(hypothesis_id="HC", statement="x", outcome="warn")
    from py_idr.utils.run import new_run_id, now_utc

    ledger = VerdictLedger(
        run_id=new_run_id(),
        sweep_csv="x",
        git_sha=None,
        built_at=now_utc(),
        records=[rec_pass, rec_fail, rec_warn],
    )
    assert ledger.summary() == {"pass": 1, "warn": 1, "fail": 1, "insufficient_data": 0}


# ---- CLI ----------------------------------------------------------------------------


@pytest.mark.unit
def test_py_idr_verdict_cli_writes_json(tmp_path: Path) -> None:
    """`py-idr verdict` reads a CSV and writes a valid VerdictLedger JSON."""
    from typer.testing import CliRunner

    from py_idr.cli import app

    rows = []
    for i in range(3):
        rows.append(_make_row(method="PY-IDR (MCMC, T=1)", regime="S1", power=0.8, seed=i))
        rows.append(_make_row(method="MaxRank", regime="S1", power=0.4, seed=i))
    csv = tmp_path / "sweep.csv"
    pd.DataFrame(rows).to_csv(csv, index=False)

    out_json = tmp_path / "verdict.json"
    result = CliRunner().invoke(
        app,
        ["verdict", "--results", str(csv), "--out", str(out_json)],
    )
    assert result.exit_code == 0, result.output
    assert out_json.exists()
    parsed = VerdictLedger.model_validate_json(out_json.read_text())
    assert len(parsed.records) == 3
