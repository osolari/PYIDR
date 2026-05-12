"""Tests for the Vanilla IDR (Li-Brown-Huang-Bickel 2011) comparator."""

from __future__ import annotations

import numpy as np
import pytest
import scipy.stats as sps

from py_idr.comparators.vanilla_idr import (
    VanillaIDRResult,
    _bivariate_gaussian_copula_logpdf,
    _empirical_rank_uniform,
    fit_vanilla_idr,
    fit_vanilla_idr_pairwise,
)

# ---- helper functions ----------------------------------------------------------------


@pytest.mark.unit
def test_empirical_rank_uniform_basic_properties() -> None:
    """Rank transform: output in (0, 1) and monotone in the input."""
    x = np.array([3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0])
    u = _empirical_rank_uniform(x)
    # All in (0, 1).
    assert np.all((u > 0.0) & (u < 1.0))
    # Order preserving (with mid-rank ties).
    order = np.argsort(x)
    assert np.all(np.diff(u[order]) >= 0.0)


@pytest.mark.unit
def test_bivariate_gaussian_copula_logpdf_matches_closed_form() -> None:
    """Compare against the bivariate Gaussian density / marginal density ratio."""
    rho = 0.7
    u = np.array([[0.5, 0.5], [0.2, 0.8], [0.9, 0.95]])
    ours = _bivariate_gaussian_copula_logpdf(u, rho)
    R = np.array([[1.0, rho], [rho, 1.0]])
    z = sps.norm.ppf(u)
    log_mvn = sps.multivariate_normal(mean=[0, 0], cov=R).logpdf(z)
    log_phi_marg = np.sum(sps.norm.logpdf(z), axis=1)
    ref = log_mvn - log_phi_marg
    assert np.allclose(ours, ref, atol=1e-7)


@pytest.mark.unit
def test_bivariate_copula_logpdf_at_rho_zero_is_zero() -> None:
    """At rho = 0 the copula reduces to the independence copula (density = 1)."""
    u = np.array([[0.3, 0.7], [0.5, 0.5], [0.8, 0.2]])
    out = _bivariate_gaussian_copula_logpdf(u, 0.0)
    assert np.allclose(out, 0.0, atol=1e-9)


# ---- fit_vanilla_idr: validation -----------------------------------------------------


@pytest.mark.unit
def test_fit_vanilla_idr_rejects_bad_shape() -> None:
    """X_pair must be (n, 2)."""
    with pytest.raises(ValueError, match="X_pair"):
        fit_vanilla_idr(np.zeros((10, 3)))


@pytest.mark.unit
def test_fit_vanilla_idr_rejects_bad_init() -> None:
    """pi_init in (0, 1); rho_init in (-1, 1)."""
    X = np.random.default_rng(0).normal(size=(10, 2))
    with pytest.raises(ValueError, match="pi_init"):
        fit_vanilla_idr(X, pi_init=0.0)
    with pytest.raises(ValueError, match="rho_init"):
        fit_vanilla_idr(X, rho_init=-1.0)


# ---- fit_vanilla_idr: correctness ----------------------------------------------------


@pytest.mark.unit
def test_fit_vanilla_idr_returns_named_dataclass() -> None:
    """Returns a VanillaIDRResult with expected fields."""
    X = np.random.default_rng(0).normal(size=(50, 2))
    res = fit_vanilla_idr(X)
    assert isinstance(res, VanillaIDRResult)
    assert res.idr.shape == (50,)
    assert 0.0 < res.pi_hat < 1.0
    assert -1.0 < res.rho_hat < 1.0


@pytest.mark.unit
def test_fit_vanilla_idr_recovers_rho_on_clean_S1() -> None:
    """On clean S1 data (K=2, n=2000, rho=0.85, pi=0.30) the EM recovers rho.

    Tight on rho (the EM identifies the correlation cleanly) and looser
    on pi (Vanilla IDR has well-known small-sample bias on pi).
    """
    from py_idr.simulation.scenarios import simulate_S1

    sim = simulate_S1(K=2, n=2000, pi_star=0.30, rho=0.85, seed=0)
    res = fit_vanilla_idr(np.asarray(sim.X), max_iter=200)
    assert abs(res.rho_hat - 0.85) < 0.05  # within 5 percentage points
    assert abs(res.pi_hat - 0.30) < 0.10  # within 10 percentage points


@pytest.mark.unit
def test_fit_vanilla_idr_idr_in_unit_interval() -> None:
    """Output local idr values are in [0, 1]."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(100, 2))
    res = fit_vanilla_idr(X)
    assert np.all((res.idr >= 0.0) & (res.idr <= 1.0))


@pytest.mark.unit
def test_fit_vanilla_idr_separates_reproducible_from_irreproducible() -> None:
    """Reproducible features get lower idr on average than irreproducible ones."""
    from py_idr.simulation.scenarios import simulate_S1

    sim = simulate_S1(K=2, n=2000, pi_star=0.30, rho=0.85, seed=0)
    Z = np.asarray(sim.true_Z)
    res = fit_vanilla_idr(np.asarray(sim.X), max_iter=200)
    assert res.idr[Z == 1].mean() < res.idr[Z == 0].mean()


@pytest.mark.unit
def test_fit_vanilla_idr_deterministic() -> None:
    """Same input → same output (EM is deterministic given the data + init)."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(50, 2))
    r1 = fit_vanilla_idr(X, max_iter=50)
    r2 = fit_vanilla_idr(X, max_iter=50)
    assert r1.pi_hat == r2.pi_hat
    assert r1.rho_hat == r2.rho_hat
    assert np.array_equal(r1.idr, r2.idr)


# ---- fit_vanilla_idr_pairwise --------------------------------------------------------


@pytest.mark.unit
def test_pairwise_K2_matches_single_call() -> None:
    """For K=2 the pairwise function reduces to a single fit_vanilla_idr call."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(80, 2))
    pair = fit_vanilla_idr_pairwise(X, aggregation="mean")
    single = fit_vanilla_idr(X).idr
    assert np.allclose(pair, single)


@pytest.mark.unit
def test_pairwise_K3_returns_correct_shape() -> None:
    """For K=3 the output is length n."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(60, 3))
    out = fit_vanilla_idr_pairwise(X)
    assert out.shape == (60,)


@pytest.mark.unit
def test_pairwise_max_is_more_conservative_than_mean() -> None:
    """The max aggregation is element-wise >= the mean aggregation."""
    from py_idr.simulation.scenarios import simulate_S1

    sim = simulate_S1(K=4, n=200, pi_star=0.30, rho=0.85, seed=0)
    mean_idr = fit_vanilla_idr_pairwise(np.asarray(sim.X), aggregation="mean")
    max_idr = fit_vanilla_idr_pairwise(np.asarray(sim.X), aggregation="max")
    assert np.all(max_idr >= mean_idr - 1e-9)


@pytest.mark.unit
def test_pairwise_rejects_K_lt_2() -> None:
    """K must be at least 2."""
    with pytest.raises(ValueError, match="K >= 2"):
        fit_vanilla_idr_pairwise(np.zeros((10, 1)))


@pytest.mark.unit
def test_pairwise_rejects_unknown_aggregation() -> None:
    """aggregation must be 'mean' or 'max'."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(20, 3))
    with pytest.raises(ValueError, match="aggregation"):
        fit_vanilla_idr_pairwise(X, aggregation="median")  # type: ignore[arg-type]
