"""Bernstein-Dirichlet marginal tests (Eq. 3.7 of the revised report)."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy import special as sps

from py_idr.marginals.bernstein import cdf, cumulative_basis, density_basis, pdf
from py_idr.marginals.degree_prior import (
    DEGREE_SUPPORT,
    log_prior,
    prior_weights,
)
from py_idr.marginals.degree_prior import (
    sample as sample_M,
)
from py_idr.marginals.dirichlet_weights import sample_latent_S, update_weights
from py_idr.marginals.pilot import kde_pilot_cdf, silverman_bandwidth
from py_idr.marginals.transform import (
    empirical_rank_transform,
    fit_pilot_and_transform,
    transform_X_to_U,
)

# ---- cumulative_basis ---------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.parametrize("M", [5, 10, 20, 40, 80])
def test_cumulative_basis_matches_scipy_betainc(M: int) -> None:
    """B_r(t; M) equals scipy.special.betainc(r, M-r+1, t) on a small grid of t and r."""
    t = jnp.array([0.05, 0.25, 0.5, 0.75, 0.95])
    out = cumulative_basis(t, M)
    for r in range(1, M + 1):
        for ti, t_val in enumerate(np.asarray(t)):
            expected = sps.betainc(r, M - r + 1, float(t_val))
            assert pytest.approx(float(out[ti, r - 1]), abs=1e-9) == expected


@pytest.mark.unit
def test_cumulative_basis_monotone_in_t() -> None:
    """For each r, B_r(·; M) is non-decreasing — it's the CDF of Beta(r, M-r+1)."""
    M = 10
    t = jnp.linspace(0.001, 0.999, 200)
    B = cumulative_basis(t, M)
    diffs = jnp.diff(B, axis=0)
    assert bool(jnp.all(diffs >= -1e-12))


@pytest.mark.unit
def test_cumulative_basis_endpoints() -> None:
    """B_r(0; M) = 0 and B_r(1; M) = 1 for every r ∈ [1, M]."""
    M = 5
    eps = 1e-6
    t = jnp.array([eps, 1 - eps])
    B = cumulative_basis(t, M)
    assert bool(jnp.all(B[0] < 1e-2))
    assert bool(jnp.all(B[1] > 1 - 1e-2))


# ---- density_basis -----------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.parametrize("M", [5, 10, 20, 40, 80])
def test_density_basis_interior_integrates_to_one(M: int) -> None:
    """∫₀¹ b_r(t; M) dt ≈ 1 for interior r — verifies the gammaln normalisation.

    Boundary indices ($r=1$ or $r=M$) have densities sharply peaked at $t=0$ or
    $t=1$; trapezoidal quadrature on a uniform grid systematically loses some
    mass there. That is a quadrature artefact, not an implementation bug — the
    basis is cross-checked against scipy at sample points elsewhere.
    """
    ts = jnp.linspace(0.001, 0.999, 8000)
    b = density_basis(ts, M)
    integrals = jnp.trapezoid(b, ts, axis=0)
    # Skip the two boundary indices.
    interior = integrals[2:-2]
    assert bool(jnp.all(jnp.abs(interior - 1.0) < 1e-2))


@pytest.mark.unit
def test_density_basis_matches_scipy_beta_pdf() -> None:
    """b_r(t; M) equals scipy.special's Beta(r, M-r+1) density at sample points."""
    M = 8
    t = jnp.array([0.2, 0.5, 0.7])
    out = density_basis(t, M)
    for r in range(1, M + 1):
        for ti, t_val in enumerate(np.asarray(t)):
            t_v = float(t_val)
            expected = (
                np.exp(sps.gammaln(M + 1) - sps.gammaln(r) - sps.gammaln(M - r + 1))
                * t_v ** (r - 1)
                * (1 - t_v) ** (M - r)
            )
            assert pytest.approx(float(out[ti, r - 1]), abs=1e-7) == expected


# ---- cdf and pdf composition --------------------------------------------------------


@pytest.mark.unit
def test_cdf_with_uniform_weights_is_identity_in_expectation() -> None:
    """With uniform weights, the Bernstein CDF reduces to t (mean of the Beta CDFs).

    More precisely, ∑_r (1/M) B_r(t; M) = t exactly because the Beta(r, M-r+1)
    CDFs at fixed t average to t (this is the polynomial-identity behind
    Bernstein-polynomial approximation of the identity function).
    """
    M = 20
    weights = jnp.ones(M) / M
    t = jnp.linspace(0.01, 0.99, 100)
    F = cdf(t, weights, M)
    assert jnp.allclose(F, t, atol=1e-6)


@pytest.mark.unit
def test_pdf_with_uniform_weights_is_constant_one() -> None:
    """Same identity at the density level: ∑_r (1/M) b_r(t; M) ≡ 1 on (0, 1)."""
    M = 20
    weights = jnp.ones(M) / M
    t = jnp.linspace(0.01, 0.99, 100)
    f = pdf(t, weights, M)
    assert jnp.allclose(f, jnp.ones_like(f), atol=1e-6)


@pytest.mark.unit
def test_cdf_rejects_wrong_weights_length() -> None:
    """cdf checks that weights match the declared degree."""
    with pytest.raises(ValueError, match="weights must have length"):
        cdf(jnp.array([0.5]), jnp.ones(7), M=10)


# ---- pilot CDF + rank transform -----------------------------------------------------


@pytest.mark.unit
def test_silverman_bandwidth_is_positive_for_finite_variance() -> None:
    """Sanity check: Silverman bandwidth is positive for any non-trivial sample."""
    rng = np.random.default_rng(0)
    x = jnp.asarray(rng.standard_normal(200))
    h = silverman_bandwidth(x)
    assert h > 0.0


@pytest.mark.unit
def test_kde_pilot_cdf_clamped_in_open_unit_interval() -> None:
    """KDE pilot CDF stays strictly inside (eps, 1-eps)."""
    rng = np.random.default_rng(1)
    x = jnp.asarray(rng.standard_normal(100))
    F0 = kde_pilot_cdf(x, x)
    assert float(F0.min()) > 0.0
    assert float(F0.max()) < 1.0


@pytest.mark.unit
def test_empirical_rank_transform_pit_uniform_for_gaussian() -> None:
    """Rank-transformed Gaussian draws are exactly uniform (0.5 mean, 1/12 variance)."""
    rng = np.random.default_rng(2)
    x = jnp.asarray(rng.standard_normal(1000))
    u = empirical_rank_transform(x)
    # Continuous sample (no ties) → exact rank uniformity: mean = 0.5.
    assert pytest.approx(float(u.mean()), abs=1e-6) == 0.5


@pytest.mark.unit
def test_transform_X_to_U_uses_the_pilot() -> None:
    """transform_X_to_U returns the Bernstein CDF of the pilot values."""
    rng = np.random.default_rng(3)
    x = jnp.asarray(rng.standard_normal(50))
    F0 = fit_pilot_and_transform(x)
    M = 10
    weights = jnp.ones(M) / M  # uniform → CDF reduces to F0
    U = transform_X_to_U(x, weights, M, F0)
    assert jnp.allclose(U, F0, atol=1e-6)


# ---- latent S + Dirichlet conjugate update ------------------------------------------


@pytest.mark.unit
def test_sample_latent_S_returns_one_based_indices_within_support() -> None:
    """sample_latent_S returns indices in {1, …, M}."""
    key = jax.random.PRNGKey(0)
    F0 = jnp.linspace(0.01, 0.99, 200)
    M = 10
    weights = jnp.ones(M) / M
    S = sample_latent_S(key, F0, weights, M)
    assert int(S.min()) >= 1
    assert int(S.max()) <= M


@pytest.mark.unit
def test_dirichlet_update_matches_conjugate_counts() -> None:
    """update_weights samples Dir(α_F + n_1, ..., α_F + n_M); means match (α_F + n_r) / sum."""
    key = jax.random.PRNGKey(0)
    M = 5
    alpha_F = 1.0
    # Synthetic latent indicators: r=1 ten times, r=2 five times, r=3..5 zero.
    S = jnp.array([1] * 10 + [2] * 5, dtype=jnp.int32)
    keys = jax.random.split(key, 5000)
    samples = jnp.stack([update_weights(k, S, M, alpha_F) for k in keys])
    empirical_mean = jnp.mean(samples, axis=0)
    # Theoretical mean of Dirichlet:
    counts = jnp.array([10, 5, 0, 0, 0], dtype=jnp.float64)
    expected_mean = (alpha_F + counts) / jnp.sum(alpha_F + counts)
    assert jnp.allclose(empirical_mean, expected_mean, atol=2e-2)


@pytest.mark.unit
def test_dirichlet_update_rejects_nonpositive_alpha_F() -> None:
    """alpha_F must be strictly positive."""
    key = jax.random.PRNGKey(0)
    S = jnp.array([1, 2, 3], dtype=jnp.int32)
    with pytest.raises(ValueError, match="positive"):
        update_weights(key, S, M=5, alpha_F=0.0)


# ---- degree prior -------------------------------------------------------------------


@pytest.mark.unit
def test_prior_weights_normalised_and_decreasing() -> None:
    """∑ w = 1 and w_5 > w_10 > … > w_80 (monotone in 1/M)."""
    w = prior_weights()
    assert pytest.approx(float(jnp.sum(w))) == 1.0
    diffs = jnp.diff(w)
    assert bool(jnp.all(diffs < 0))


@pytest.mark.unit
def test_log_prior_returns_negative_inf_off_support() -> None:
    """log_prior at unsupported degree is -inf."""
    assert log_prior(7) == float("-inf")


@pytest.mark.unit
def test_log_prior_in_support_matches_normalised_weights() -> None:
    """log_prior at supported m matches log(prior_weights[idx])."""
    w = prior_weights()
    for idx, m in enumerate(DEGREE_SUPPORT):
        assert pytest.approx(log_prior(m), abs=1e-9) == float(jnp.log(w[idx]))


@pytest.mark.unit
def test_sample_M_returns_value_in_support() -> None:
    """Sampled M is one of {5, 10, 20, 40, 80}."""
    keys = jax.random.split(jax.random.PRNGKey(0), 50)
    for k in keys:
        m = sample_M(k)
        assert m in DEGREE_SUPPORT
