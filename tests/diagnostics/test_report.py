"""DiagnosticsReport tests (plan 12)."""

from __future__ import annotations

import arviz as az
import numpy as np
import pytest

from py_idr.eval.diagnostics_report import (
    DIVERGENCE_FRACTION_THRESHOLD,
    ESS_PASS_THRESHOLD,
    MIN_CHAINS_PASS,
    RHAT_FAIL_THRESHOLD,
    RHAT_PASS_THRESHOLD,
    DiagnosticsReport,
    build_diagnostics_report,
)
from py_idr.utils.arviz_compat import idata_from_posterior as _idata_from_posterior


def _well_mixed_idata(
    num_chains: int = 4, num_draws: int = 1000, seed: int = 0
) -> az.InferenceData:
    """Synthesise a well-mixed multi-chain trace (each chain ~N(0.6, 0.05²))."""
    rng = np.random.default_rng(seed)
    return _idata_from_posterior(
        {
            "pi": rng.normal(0.6, 0.05, size=(num_chains, num_draws)),
            "rho": rng.normal(0.7, 0.05, size=(num_chains, num_draws)),
        }
    )


def _broken_idata(seed: int = 0) -> az.InferenceData:
    """Synthesise a non-mixing trace where each chain sits in a different mode."""
    rng = np.random.default_rng(seed)
    posterior = {
        "pi": np.stack(
            [
                rng.uniform(0.10, 0.20, size=500),
                rng.uniform(0.40, 0.50, size=500),
                rng.uniform(0.70, 0.80, size=500),
                rng.uniform(0.85, 0.95, size=500),
            ]
        )
    }
    return _idata_from_posterior(posterior)


# ---- verdict logic -----------------------------------------------------------------


@pytest.mark.unit
def test_well_mixed_chain_passes() -> None:
    """A well-mixed 4-chain trace yields verdict=pass with R̂≈1 and high ESS."""
    idata = _well_mixed_idata()
    report = build_diagnostics_report(idata)
    assert report.verdict == "pass"
    for var, r in report.rhat.items():
        assert r < RHAT_PASS_THRESHOLD, f"{var} R̂ = {r} should be ≤ {RHAT_PASS_THRESHOLD}"
    for var, ess in report.ess_bulk.items():
        assert ess > ESS_PASS_THRESHOLD, f"{var} ess_bulk = {ess} should be > {ESS_PASS_THRESHOLD}"


@pytest.mark.unit
def test_broken_chain_fails() -> None:
    """Per-chain disjoint posteriors yield verdict=fail with high R̂."""
    idata = _broken_idata()
    report = build_diagnostics_report(idata)
    assert report.verdict == "fail"
    assert report.rhat["pi"] > RHAT_FAIL_THRESHOLD


@pytest.mark.unit
def test_single_chain_fails() -> None:
    """A single chain can't produce split-R̂ — verdict=fail by §3.3.1."""
    idata = _idata_from_posterior({"pi": np.random.uniform(0.5, 0.6, size=(1, 500))})
    report = build_diagnostics_report(idata)
    assert report.verdict == "fail"
    assert any("1 chain" in w or "chain(s)" in w for w in report.warnings)
    assert report.rhat == {}


@pytest.mark.unit
def test_three_chains_warns_not_passes() -> None:
    """3 chains is fewer than the §3.3.1 recommendation of 4 — verdict drops to warn."""
    idata = _well_mixed_idata(num_chains=3, num_draws=2000)
    report = build_diagnostics_report(idata)
    assert report.verdict == "warn"
    assert any("4" in w for w in report.warnings)


@pytest.mark.unit
def test_low_ess_warns() -> None:
    """A short chain has ESS < 400, triggering a warn verdict."""
    idata = _well_mixed_idata(num_chains=4, num_draws=50)
    report = build_diagnostics_report(idata)
    assert report.verdict == "warn"
    assert any("ess_bulk" in w for w in report.warnings)


@pytest.mark.unit
def test_divergent_transitions_above_threshold_fail() -> None:
    """A divergence fraction above the threshold flips the verdict to fail."""
    idata = _well_mixed_idata()
    total = MIN_CHAINS_PASS * 1000
    # Above the threshold.
    n_div = int(total * DIVERGENCE_FRACTION_THRESHOLD * 2)
    report = build_diagnostics_report(idata, num_divergent_transitions=n_div)
    assert report.verdict == "fail"
    assert any("divergent" in w for w in report.warnings)


@pytest.mark.unit
def test_few_divergent_transitions_warns_not_fails() -> None:
    """A small number of divergent transitions below the fraction threshold = warn."""
    idata = _well_mixed_idata()
    report = build_diagnostics_report(idata, num_divergent_transitions=3)
    assert report.verdict == "warn"
    assert any("divergent" in w for w in report.warnings)


# ---- DiagnosticsReport pydantic round-trip -----------------------------------------


@pytest.mark.unit
def test_diagnostics_report_round_trip() -> None:
    """DiagnosticsReport serialises and reloads via JSON without losing fields."""
    report = build_diagnostics_report(_well_mixed_idata())
    blob = report.model_dump_json()
    reloaded = DiagnosticsReport.model_validate_json(blob)
    assert reloaded == report


@pytest.mark.unit
def test_diagnostics_report_rejects_unknown_field() -> None:
    """extra='forbid' guards against typos in the report schema."""
    payload = {
        "rhat": {"pi": 1.0},
        "ess_bulk": {"pi": 500.0},
        "ess_tail": {"pi": 500.0},
        "num_chains": 4,
        "num_draws_per_chain": 100,
        "num_divergent_transitions": 0,
        "divergence_fraction": 0.0,
        "verdict": "pass",
        "warnings": [],
        "foo": "bar",  # not in schema
    }
    with pytest.raises((ValueError, TypeError)):
        DiagnosticsReport.model_validate(payload)


@pytest.mark.unit
def test_scalar_vars_filter_restricts_summary() -> None:
    """Restricting scalar_vars to a subset only reports those variables."""
    idata = _well_mixed_idata()
    report = build_diagnostics_report(idata, scalar_vars=["pi"])
    assert "pi" in report.rhat
    assert "rho" not in report.rhat


# ---- multi-chain run shape integration ---------------------------------------------


@pytest.mark.integration
def test_run_multi_chain_simple_returns_meaningful_rhat() -> None:
    """The multi-chain orchestrator produces an idata that build_diagnostics_report consumes."""
    import jax
    import jax.numpy as jnp

    from py_idr.inference.mcmc.run_chain import run_multi_chain_simple
    from py_idr.marginals.transform import empirical_rank_transform

    rng = np.random.default_rng(0)
    U = rng.uniform(size=(30, 2))
    F0 = jnp.stack([empirical_rank_transform(jnp.asarray(U[:, j])) for j in range(2)], axis=1)
    result = run_multi_chain_simple(
        F0=F0,
        M=5,
        key=jax.random.PRNGKey(0),
        num_chains=2,
        n_warmup=5,
        n_samples=10,
        nuts_warmup=10,
    )
    report = build_diagnostics_report(result.idata, scalar_vars=["pi", "rho"])
    # 2 chains → verdict can't be 'pass' (need >= 4) but the report should populate
    # the rhat dict (split-Rhat needs ≥ 2 chains and we have that).
    assert report.num_chains == 2
    assert report.num_draws_per_chain == 10
    assert "pi" in report.rhat
    assert report.verdict in ("warn", "fail")
