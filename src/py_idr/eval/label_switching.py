r"""Label-switching diagnostics for the multi-cluster Pólya-urn MCMC.

A Pitman-Yor mixture posterior is invariant under permutations of the cluster
labels, so any per-cluster summary (cluster means, cluster correlations,
cluster sizes) is identifiable only up to relabelling. Raw posterior averages
of per-cluster scalars are therefore meaningless without first **deciding
which cluster is which** across MCMC iterations.

This module provides two complementary tools:

1. :func:`sorted_cluster_sizes` — a fast, deterministic, label-invariant
   summary that does **not** require relabelling. Per iteration we sort
   cluster sizes in descending order and pad with zeros to a fixed
   length; the resulting trace lives in $\\R^{T_\\max}$ and the per-component
   mean / std / Wasserstein-1 to a reference is a well-defined posterior
   summary.

2. :func:`stephens_relabel` — the iterative relabelling algorithm of
   \\citet{Stephens2000Relabel}. Given per-iteration matrices
   $P^{(t)}_{i, k} = \\Pr(z_i = k \\mid \\text{iter}\\ t)$ over a fixed cluster
   count $K$, returns the permutation per iteration that minimises the
   total KL divergence to the running mean. Used by callers that need
   per-cluster scalar summaries (e.g. cluster-mean correlation).

The cluster-correlation Wasserstein-1 summary that plan 12 calls out is
:func:`wasserstein_cluster_summary`, which takes a sorted-size trace and
computes the 1-Wasserstein distance against either the prior expectation
(uniform under the multi-cluster Pólya-urn at small $T$) or a user-supplied
reference.

The :class:`DiagnosticsReport` field ``label_invariant_summary`` is populated
from :func:`build_label_invariant_summary` which composes the helpers above.

See plans/12_diagnostics.md and Stephens (2000), "Dealing with label switching
in mixture models", JRSS-B 62(4): 795-809.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment


def sorted_cluster_sizes(
    cluster_assignments: np.ndarray,
    T_max: int,
) -> np.ndarray:
    r"""Per-iteration descending-sorted cluster-size vectors.

    Parameters
    ----------
    cluster_assignments
        Integer array of shape ``(n_samples, n)`` from
        :class:`py_idr.inference.mcmc.run_chain.ChainResultMulti.cluster_assignments`.
        Entries are ``-1`` for irreproducible features and ``0..T-1`` for
        reproducible features at the corresponding iteration.
    T_max
        Cluster truncation (the same ``T_max`` passed to the chain driver).
        The output has shape ``(n_samples, T_max)`` so per-component
        summaries across iterations are well-defined.

    Returns
    -------
    Array of shape ``(n_samples, T_max)`` where row ``t`` is the
    descending-sorted cluster sizes at iteration ``t``, zero-padded if the
    iteration has fewer than ``T_max`` active clusters.

    Notes
    -----
    The sorted-size vector is permutation-invariant by construction:
    relabelling clusters within an iteration leaves the sorted vector
    unchanged. This is the cheapest label-invariant summary; for richer
    per-cluster scalars use :func:`stephens_relabel`.
    """
    if cluster_assignments.ndim != 2:
        raise ValueError(
            f"cluster_assignments must be 2-D (n_samples, n); got shape {cluster_assignments.shape}"
        )
    if T_max <= 0:
        raise ValueError(f"T_max must be > 0; got {T_max}")

    n_samples, _n = cluster_assignments.shape
    out = np.zeros((n_samples, T_max), dtype=np.int64)
    for t in range(n_samples):
        z_t = cluster_assignments[t]
        # Count occurrences of each non-negative label.
        if z_t.size == 0:
            continue
        # `minlength=T_max` ensures the buffer is exactly T_max long even
        # if z_t has no reproducible features.
        positive = z_t[z_t >= 0]
        if positive.size == 0:
            continue
        counts = np.bincount(positive, minlength=T_max)
        # Drop any indices beyond T_max if the chain ever spawned more
        # (defensive — shouldn't happen because chain enforces T <= T_max).
        counts = counts[:T_max]
        # Pad if counts is shorter than T_max (shouldn't happen given
        # minlength, but guard).
        if counts.size < T_max:
            counts = np.pad(counts, (0, T_max - counts.size))
        out[t] = np.sort(counts)[::-1]
    return out


def stephens_relabel(
    P: np.ndarray,
    *,
    max_iters: int = 50,
    tol: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray]:
    r"""Stephens (2000) iterative relabeller for an MCMC partition trace.

    Parameters
    ----------
    P
        Array of shape ``(n_samples, n, K)`` where
        $P[t, i, k] = \\Pr(z_i = k \\mid \\text{iter}\\ t)$ — typically a
        one-hot matrix for hard-assignment chains. The cluster axis is of
        fixed width $K$; iterations with fewer than $K$ active clusters
        contribute zero columns.
    max_iters
        Cap on the alternating-minimisation outer loop. Stephens (2000) Algorithm 2.
        Default 50 is generous; convergence typically takes < 10.
    tol
        Stop when the change in the per-iteration assignment matrix Frobenius
        norm falls below ``tol``.

    Returns
    -------
    A 2-tuple ``(perms, P_relabeled)``:

    - ``perms`` of shape ``(n_samples, K)``, where ``perms[t]`` is a permutation
      of ``range(K)`` and ``z_relabeled = perms[t][z_raw]``.
    - ``P_relabeled`` of shape ``(n_samples, n, K)`` is ``P`` with the
      permutation applied along the cluster axis.

    Notes
    -----
    Algorithm:

    1. Initialise reference $Q_{i, k} = (1/T) \\sum_t P^{(t)}_{i, k}$.
    2. For each iteration $t$, find the permutation $\\sigma_t$ of
       $\\{1, \\ldots, K\\}$ minimising the KL divergence
       $\\sum_{i, k} P^{(t)}_{i, \\sigma_t(k)} \\log(P^{(t)}_{i, \\sigma_t(k)} / Q_{i, k})$.
       This reduces to a linear-sum-assignment with cost matrix
       $C_{a, b} = \\sum_i P^{(t)}_{i, a} \\log(P^{(t)}_{i, a} / Q_{i, b})$.
    3. Update $Q$ to the mean of the relabeled $P^{(t)}$.
    4. Iterate steps 2--3 until $Q$ stabilises.

    The implementation uses :func:`scipy.optimize.linear_sum_assignment` for
    the per-iteration permutation search.
    """
    if P.ndim != 3:
        raise ValueError(f"P must be 3-D (n_samples, n, K); got shape {P.shape}")
    n_samples, n, K = P.shape
    if K < 1:
        raise ValueError(f"K must be >= 1; got {K}")

    # Numerical floor on probabilities; KL is undefined at P=0 or Q=0.
    eps = 1e-12
    P_safe = np.clip(P, eps, 1.0)

    # Initialise Q as the cluster-averaged P (no relabelling yet).
    Q = P_safe.mean(axis=0)
    Q = np.clip(Q, eps, 1.0)

    perms = np.tile(np.arange(K), (n_samples, 1))
    last_Q = Q.copy()

    for _outer in range(max_iters):
        # For every iteration, find the permutation that best matches Q.
        new_perms = np.empty_like(perms)
        for t in range(n_samples):
            Pt = P_safe[t]  # (n, K)
            # Cost C[a, b] = sum_i P_t[i, a] log(P_t[i, a] / Q[i, b])
            # = sum_i P_t[i, a] log(P_t[i, a]) - sum_i P_t[i, a] log(Q[i, b])
            # The first term doesn't depend on b → drops out of the
            # assignment problem. Keep it anyway for cost-magnitude
            # interpretability.
            log_P = np.log(Pt)  # (n, K)
            log_Q = np.log(Q)  # (n, K)
            # C[a, b] = sum_i P_t[i, a] * (log_P[i, a] - log_Q[i, b])
            # = (P_t * log_P).sum(0)[a]   -   (P_t.T @ log_Q)[a, b]
            self_term = (Pt * log_P).sum(0)  # (K,)
            cross_term = Pt.T @ log_Q  # (K, K) — a-th row, b-th col
            C = self_term[:, None] - cross_term
            row_ind, col_ind = linear_sum_assignment(C)
            # row_ind is range(K); col_ind[a] = b means raw cluster a
            # maps to relabeled cluster b under σ(a) = b. To re-index
            # the columns of P[t] so that "the relabeled column b
            # holds the data of the original cluster σ⁻¹(b)", we store
            # σ⁻¹ in perms[t] (which is np.argsort(col_ind)). Then
            # relabeled[t][:, b] = P[t][:, σ⁻¹(b)] = P[t][:, perms[t][b]],
            # which is the numpy-native form below.
            new_perms[t] = np.argsort(col_ind)
        perms = new_perms
        # Apply permutations and recompute Q.
        relabeled = np.empty_like(P_safe)
        for t in range(n_samples):
            relabeled[t] = P_safe[t][:, perms[t]]
        Q_new = relabeled.mean(axis=0)
        Q_new = np.clip(Q_new, eps, 1.0)
        if np.linalg.norm(Q_new - last_Q) < tol:
            Q = Q_new
            break
        Q = Q_new
        last_Q = Q.copy()

    P_relabeled = np.empty_like(P)
    for t in range(n_samples):
        P_relabeled[t] = P[t][:, perms[t]]
    return perms, P_relabeled


def wasserstein_cluster_summary(
    sorted_sizes: np.ndarray,
    reference: np.ndarray | None = None,
) -> dict[str, float]:
    r"""Wasserstein-1 + per-rank summary statistics for a sorted-size trace.

    Parameters
    ----------
    sorted_sizes
        Output of :func:`sorted_cluster_sizes`; shape ``(n_samples, T_max)``.
        Each row is descending-sorted cluster sizes for one MCMC iteration.
    reference
        Optional ``(T_max,)`` reference (e.g., the prior expected sizes) to
        compare against. If ``None`` the reference is the empirical mean
        across iterations — useful as a "did the chain mix on its sorted-
        size summary" diagnostic.

    Returns
    -------
    A dict with keys:

    - ``"rank_0_mean"`` ... ``"rank_{T_max-1}_mean"``: per-rank mean size
    - ``"rank_0_std"`` ... ``"rank_{T_max-1}_std"``: per-rank std
    - ``"wasserstein1_to_reference"``: scalar 1-Wasserstein distance between
      the **iteration-averaged** sorted-size distribution and ``reference``.
      Computed via the closed-form 1-Wasserstein on the empirical CDFs of
      the discrete distributions.
    """
    if sorted_sizes.ndim != 2:
        raise ValueError(
            f"sorted_sizes must be 2-D (n_samples, T_max); got shape {sorted_sizes.shape}"
        )
    _n_samples, T_max = sorted_sizes.shape
    mean_per_rank = sorted_sizes.mean(axis=0)
    std_per_rank = sorted_sizes.std(axis=0)

    if reference is None:
        # Use the cross-iteration mean as the reference. The Wasserstein
        # distance is then trivially zero — useful when callers want
        # only the rank-wise mean/std without an external reference.
        ref = mean_per_rank
    else:
        ref = np.asarray(reference, dtype=np.float64)
        if ref.shape != (T_max,):
            raise ValueError(f"reference must have shape ({T_max},); got {ref.shape}")

    # 1-Wasserstein between two non-negative weight vectors of equal
    # length, treating them as histograms over rank-positions 0..T_max-1:
    # = sum_i |cumsum(p)[i] - cumsum(q)[i]| * (1)  — bin width 1.
    p = mean_per_rank / max(mean_per_rank.sum(), 1.0)
    q = ref / max(ref.sum(), 1.0)
    w1 = float(np.abs(np.cumsum(p) - np.cumsum(q)).sum())

    out = {}
    for r in range(T_max):
        out[f"rank_{r}_mean"] = float(mean_per_rank[r])
        out[f"rank_{r}_std"] = float(std_per_rank[r])
    out["wasserstein1_to_reference"] = w1
    return out


def build_label_invariant_summary(
    cluster_assignments: np.ndarray,
    *,
    T_max: int,
    reference: np.ndarray | None = None,
) -> dict[str, float]:
    """Compose the sorted-size summary into the ``DiagnosticsReport`` dict.

    The returned dict is :class:`DiagnosticsReport.label_invariant_summary`
    in its ``"sorted_cluster_size_..."`` namespace; values are floats so the
    pydantic ``dict[str, float]`` schema validates without coercion.

    Parameters
    ----------
    cluster_assignments
        Integer array of shape ``(n_samples, n)`` from the multi-cluster
        chain driver.
    T_max
        Cluster truncation passed to the chain driver.
    reference
        Optional prior reference (``(T_max,)`` array) for the Wasserstein-1
        distance. ``None`` → self-reference (W1 = 0).

    Returns
    -------
    Dict whose keys are prefixed ``sorted_cluster_size__`` to disambiguate
    from any future label-invariant summaries (e.g., per-cluster correlation
    distributions, once the chain exposes them).
    """
    sizes = sorted_cluster_sizes(cluster_assignments, T_max=T_max)
    base = wasserstein_cluster_summary(sizes, reference=reference)
    return {f"sorted_cluster_size__{k}": v for k, v in base.items()}
