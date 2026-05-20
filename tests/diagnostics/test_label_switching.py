"""Tests for the Stephens (2000) label-switching diagnostic (W10.12)."""

from __future__ import annotations

import numpy as np
import pytest

from py_idr.eval.diagnostics_report import build_diagnostics_report
from py_idr.eval.label_switching import (
    build_label_invariant_summary,
    sorted_cluster_sizes,
    stephens_relabel,
    wasserstein_cluster_summary,
)
from py_idr.utils.arviz_compat import idata_from_posterior

# ---- sorted_cluster_sizes ----------------------------------------------------------


@pytest.mark.unit
def test_sorted_cluster_sizes_shape_and_dtype() -> None:
    """Output shape is (n_samples, T_max) and integer-typed."""
    rng = np.random.default_rng(0)
    cluster_assignments = rng.integers(low=-1, high=4, size=(50, 30))
    out = sorted_cluster_sizes(cluster_assignments, T_max=6)
    assert out.shape == (50, 6)
    assert out.dtype.kind in ("i", "u")


@pytest.mark.unit
def test_sorted_cluster_sizes_is_permutation_invariant() -> None:
    """Relabelling clusters within an iteration leaves the sorted-size row unchanged."""
    cluster_assignments = np.array([[0, 0, 1, 2, 2, 2], [3, 3, 0, 0, 1, 1]])
    out = sorted_cluster_sizes(cluster_assignments, T_max=4)
    # Iteration 0: cluster 0=2, cluster 1=1, cluster 2=3 -> sorted [3, 2, 1, 0]
    # Iteration 1: cluster 3=2, cluster 0=2, cluster 1=2  -> sorted [2, 2, 2, 0]
    assert np.array_equal(out[0], [3, 2, 1, 0])
    assert np.array_equal(out[1], [2, 2, 2, 0])

    # Apply a random permutation to iteration 0's labels — sorted output is the same.
    perm = np.array([2, 0, 1, 3])
    z0_perm = perm[cluster_assignments[0]]
    out_perm = sorted_cluster_sizes(z0_perm[None, :], T_max=4)[0]
    assert np.array_equal(out_perm, [3, 2, 1, 0])


@pytest.mark.unit
def test_sorted_cluster_sizes_handles_all_negative_iteration() -> None:
    """An iteration with no reproducible features → zero row."""
    cluster_assignments = np.array([[-1, -1, -1, -1]])
    out = sorted_cluster_sizes(cluster_assignments, T_max=3)
    assert np.array_equal(out[0], [0, 0, 0])


@pytest.mark.unit
def test_sorted_cluster_sizes_rejects_bad_shape() -> None:
    with pytest.raises(ValueError, match="2-D"):
        sorted_cluster_sizes(np.zeros(10, dtype=int), T_max=3)


@pytest.mark.unit
def test_sorted_cluster_sizes_rejects_nonpositive_T_max() -> None:
    with pytest.raises(ValueError, match="T_max"):
        sorted_cluster_sizes(np.zeros((5, 10), dtype=int), T_max=0)


# ---- stephens_relabel -------------------------------------------------------------


@pytest.mark.unit
def test_stephens_relabel_fixes_a_known_permutation() -> None:
    """If iteration 1 is a column-permuted version of iteration 0, the
    relabeller maps them to the same canonical layout."""
    n, K = 20, 3
    rng = np.random.default_rng(0)
    z_truth = rng.integers(0, K, size=n)
    P0 = np.eye(K)[z_truth]  # (n, K) one-hot
    # P1's column k holds P0's column perm_true[k] — i.e. P1 = P0[:, perm_true].
    # Relabelling P1 with σ⁻¹ = argsort(perm_true) recovers P0.
    perm_true = np.array([1, 2, 0])
    P1 = P0[:, perm_true]
    P = np.stack([P0, P1], axis=0)
    perms, P_relabeled = stephens_relabel(P, max_iters=20)
    # Both iterations should converge to the same canonical labelling.
    assert np.allclose(P_relabeled[0], P_relabeled[1])


@pytest.mark.unit
def test_stephens_relabel_perms_are_permutations() -> None:
    """Each row of `perms` is a valid permutation of range(K)."""
    n, K, T = 15, 4, 10
    rng = np.random.default_rng(1)
    P = rng.dirichlet(np.ones(K), size=(T, n))
    perms, _ = stephens_relabel(P, max_iters=10)
    for t in range(T):
        assert sorted(perms[t].tolist()) == list(range(K))


@pytest.mark.unit
def test_stephens_relabel_rejects_bad_shape() -> None:
    with pytest.raises(ValueError, match="3-D"):
        stephens_relabel(np.zeros((5, 4)))


# ---- wasserstein_cluster_summary --------------------------------------------------


@pytest.mark.unit
def test_wasserstein_cluster_summary_self_distance_is_zero() -> None:
    """W1 to the mean-of-self is 0; the mean/std fields are non-negative."""
    rng = np.random.default_rng(0)
    sizes = rng.integers(0, 10, size=(20, 5))
    out = wasserstein_cluster_summary(sizes)
    assert out["wasserstein1_to_reference"] == pytest.approx(0.0, abs=1e-9)
    for r in range(5):
        assert out[f"rank_{r}_std"] >= 0
        assert out[f"rank_{r}_mean"] >= 0


@pytest.mark.unit
def test_wasserstein_cluster_summary_against_concrete_reference() -> None:
    """W1 between [3, 0, 0] and [0, 3, 0] (normalised) is 1 (one rank-step)."""
    sizes = np.array([[3, 0, 0]])  # iteration mean → [3, 0, 0]
    out = wasserstein_cluster_summary(sizes, reference=np.array([0.0, 3.0, 0.0]))
    # CDFs: p = [1, 1, 1]; q = [0, 1, 1]; |diff| sums to 1.
    assert out["wasserstein1_to_reference"] == pytest.approx(1.0, abs=1e-9)


# ---- DiagnosticsReport integration ----------------------------------------------


@pytest.mark.unit
def test_build_diagnostics_report_with_cluster_assignments() -> None:
    """DiagnosticsReport.label_invariant_summary is populated when
    cluster_assignments + T_max are supplied."""
    rng = np.random.default_rng(0)
    n_samples, num_chains, n = 50, 4, 30
    posterior = {
        "pi": rng.normal(0.5, 0.05, size=(num_chains, n_samples)),
    }
    idata = idata_from_posterior(posterior)
    cluster_assignments = rng.integers(low=-1, high=4, size=(n_samples, n))

    report = build_diagnostics_report(
        idata,
        cluster_assignments=cluster_assignments,
        T_max=5,
    )
    assert isinstance(report.label_invariant_summary, dict)
    assert "sorted_cluster_size__wasserstein1_to_reference" in report.label_invariant_summary
    assert "sorted_cluster_size__rank_0_mean" in report.label_invariant_summary


@pytest.mark.unit
def test_build_diagnostics_report_without_cluster_assignments_has_empty_summary() -> None:
    """T=1 path: no cluster_assignments → label_invariant_summary is {}."""
    rng = np.random.default_rng(0)
    posterior = {"pi": rng.normal(0.5, 0.05, size=(4, 50))}
    idata = idata_from_posterior(posterior)
    report = build_diagnostics_report(idata)
    assert report.label_invariant_summary == {}


@pytest.mark.unit
def test_build_diagnostics_report_requires_T_max_with_cluster_assignments() -> None:
    """Passing cluster_assignments without T_max raises a clear error."""
    rng = np.random.default_rng(0)
    posterior = {"pi": rng.normal(0.5, 0.05, size=(4, 50))}
    idata = idata_from_posterior(posterior)
    with pytest.raises(ValueError, match="T_max is required"):
        build_diagnostics_report(
            idata,
            cluster_assignments=rng.integers(low=-1, high=4, size=(50, 20)),
        )


@pytest.mark.unit
def test_build_label_invariant_summary_namespaces_keys() -> None:
    """Top-level helper prefixes keys with 'sorted_cluster_size__'."""
    rng = np.random.default_rng(0)
    z = rng.integers(low=-1, high=3, size=(10, 20))
    summary = build_label_invariant_summary(z, T_max=4)
    for k in summary:
        assert k.startswith("sorted_cluster_size__")
    # Sanity-check at least one of the expected per-rank entries lives.
    assert "sorted_cluster_size__rank_0_mean" in summary
