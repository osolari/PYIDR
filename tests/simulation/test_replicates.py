"""End-to-end replicate driver tests."""

from __future__ import annotations

import pytest

from py_idr.simulation.replicates import ReplicateRow, run_replicate_S1


@pytest.mark.integration
@pytest.mark.slow
def test_run_replicate_S1_produces_valid_row() -> None:
    """The driver returns a ReplicateRow with finite scalar fields.

    Uses a tiny, short run — recovery is loose but the pipeline must
    produce a structurally valid row. Detailed recovery validation belongs
    to a multi-replicate simulation sweep (plan 07 Phase II).
    """
    row = run_replicate_S1(
        K=2,
        n=50,
        pi_star=0.30,
        rho=0.85,
        seed=0,
        alpha=0.05,
        num_chains=2,
        n_warmup=15,
        n_samples=20,
        nuts_warmup=10,
        M=5,
    )
    assert isinstance(row, ReplicateRow)
    assert row.regime == "S1"
    assert row.K == 2
    assert row.n == 50
    assert row.pi_true == pytest.approx(0.30)
    assert row.rho_true == pytest.approx(0.85)
    # Posterior means are in valid ranges.
    assert 0.0 < row.pi_posterior_mean < 1.0
    assert 0.0 < row.rho_posterior_mean < 1.0
    # Recovery scalars are finite and in [0, 1].
    assert 0.0 <= row.realized_fdr <= 1.0
    assert 0.0 <= row.power <= 1.0
    assert row.k_alpha >= 0
    # Chain metadata stored.
    assert row.num_chains == 2
    assert row.num_samples_per_chain == 20


@pytest.mark.integration
def test_replicate_row_to_dict_serialisable() -> None:
    """to_dict() returns a plain dict with the right field set for CSV writing."""
    from py_idr.simulation.scenarios import simulate_S1

    # Build a hand-constructed ReplicateRow without running the chain (fast).
    sim = simulate_S1(K=2, n=20, seed=0)
    row = ReplicateRow(
        regime=sim.regime,
        K=sim.K,
        n=sim.n,
        replicate_seed=sim.seed,
        method="PY-IDR (MCMC, T=1)",
        alpha=0.05,
        pi_true=sim.true_pi,
        pi_posterior_mean=0.4,
        rho_true=sim.true_rho,
        rho_posterior_mean=0.7,
        k_alpha=5,
        realized_fdr=0.1,
        power=0.8,
        num_chains=2,
        num_samples_per_chain=100,
    )
    d = row.to_dict()
    # The CSV writer needs these specific columns.
    expected_keys = {
        "regime",
        "K",
        "n",
        "replicate_seed",
        "method",
        "alpha",
        "pi_true",
        "pi_posterior_mean",
        "rho_true",
        "rho_posterior_mean",
        "k_alpha",
        "realized_fdr",
        "power",
        "num_chains",
        "num_samples_per_chain",
    }
    assert set(d.keys()) == expected_keys
