# ---
# jupyter:
#   jupytext:
#     formats: py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
# ---

# %% [markdown]
# # 07 — End-to-end S1 walkthrough with diagnostics
#
# This notebook ties together everything from the previous six:
#
# 1. Generate one S1 dataset.
# 2. Run a multi-chain PY-IDR fit (T = 1 path).
# 3. Inspect the posterior on $\pi$ and $\rho$.
# 4. Inspect the diagnostics report.
# 5. Apply Sun-Cai at $\alpha = 0.05$ and compare to truth.
#
# If only one notebook in this directory runs to completion on your
# machine, make it this one.

# %%
from __future__ import annotations

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

# %% [markdown]
# ## Step 1: Synthetic data
# %%
from py_idr.simulation.scenarios import simulate_S1  # type: ignore[attr-defined]

sim = simulate_S1(K=3, n=400, pi_star=0.30, rho=0.85, seed=0)
print(f"X shape: {sim.X.shape}, true pi: {sim.true_pi}, true rho: {sim.true_rho}")

# %% [markdown]
# ## Step 2: Pilot CDF + multi-chain fit
#
# Use the empirical-rank transform as the pilot, then run two short
# chains. This will take ~60-90 seconds on CPU.

# %%
from py_idr.inference.mcmc.run_chain import run_multi_chain_simple  # type: ignore[attr-defined]
from py_idr.marginals.transform import empirical_rank_transform  # type: ignore[attr-defined]

F0 = jnp.stack(
    [empirical_rank_transform(sim.X[:, j]) for j in range(sim.K)],
    axis=1,
)

result = run_multi_chain_simple(
    F0=F0,
    M=5,
    key=jax.random.PRNGKey(0),
    num_chains=2,
    n_warmup=50,
    n_samples=200,
    nuts_warmup=20,
)
print("Posterior keys:", list(result.idata.posterior.data_vars))
print("z_samples shape:", result.z_samples.shape)  # (chains, samples, n)

# %% [markdown]
# ## Step 3: Posterior on $\pi$ and $\rho$

# %%
pi_samples = np.asarray(result.idata.posterior["pi"]).reshape(-1)
rho_samples = np.asarray(result.idata.posterior["rho"]).reshape(-1)
print(f"posterior mean pi:  {pi_samples.mean():.3f}  (true {sim.true_pi})")
print(f"posterior mean rho: {rho_samples.mean():.3f}  (true {sim.true_rho})")

fig, axes = plt.subplots(1, 2, figsize=(13, 4))
axes[0].hist(pi_samples, bins=40, alpha=0.7)
axes[0].axvline(sim.true_pi, color="red", lw=2, label="truth")
axes[0].set_xlabel("$\\pi$")
axes[0].set_ylabel("count")
axes[0].legend()
axes[0].set_title("Posterior on $\\pi$")

axes[1].hist(rho_samples, bins=40, alpha=0.7)
axes[1].axvline(sim.true_rho, color="red", lw=2, label="truth")
axes[1].set_xlabel("$\\rho$")
axes[1].set_ylabel("count")
axes[1].legend()
axes[1].set_title("Posterior on $\\rho$")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Step 4: Diagnostics
#
# The bundled diagnostics check Rhat, ESS, divergence fraction, and
# chain count.

# %%
from py_idr.eval.diagnostics_report import build_diagnostics_report  # type: ignore[attr-defined]

report = build_diagnostics_report(result.idata, num_chains=2)
print(f"verdict: {report.verdict}")
print(f"  rhat_max:         {report.rhat_max:.4f}")
print(f"  ess_bulk_min:     {report.ess_bulk_min:.0f}")
print(f"  ess_tail_min:     {report.ess_tail_min:.0f}")
print(f"  divergence_frac:  {report.divergence_fraction:.4f}")
print(f"  num_chains:       {report.num_chains}")
if report.failed_checks:
    print(f"  failed checks:    {report.failed_checks}")

# %% [markdown]
# A `warn` verdict here is *expected* because the chain is short and
# uses only 2 chains. For real work, use 4 chains and at least
# `n_samples = 500`.

# %% [markdown]
# ## Step 5: Local idr + Sun-Cai
#
# Flatten chains, average, apply Sun-Cai.

# %%
from py_idr.decision.local_idr import from_mcmc
from py_idr.simulation.evaluation import evaluate_recovery

z_flat = result.z_samples.reshape(-1, result.z_samples.shape[-1])
idr = from_mcmc(jnp.asarray(z_flat))

alpha = 0.05
rec = evaluate_recovery(idr, sim.true_Z, alpha=alpha)
print(f"At alpha = {alpha}:")
print(f"  k_alpha       = {rec.k_alpha}")
print(f"  realised FDR  = {rec.realized_fdr:.3f}")
print(f"  power         = {rec.power:.3f}")

# %% [markdown]
# ## Visualise the call set
#
# Overlay called features on the pairwise scatter of replicates 1 and 2.

# %%
from py_idr.decision.sun_cai import step_up_threshold

dec = step_up_threshold(idr, alpha=alpha)
called_idx = np.asarray(dec.selected)
called = np.zeros(sim.n, dtype=bool)
called[called_idx] = True
X = np.asarray(sim.X)
Z = np.asarray(sim.true_Z)

fig, ax = plt.subplots(figsize=(6, 6))
ax.scatter(
    X[~called & (Z == 0), 0],
    X[~called & (Z == 0), 1],
    s=8,
    c="lightgrey",
    label="rejected (truly irrep)",
)
ax.scatter(
    X[~called & (Z == 1), 0],
    X[~called & (Z == 1), 1],
    s=8,
    c="orange",
    label="rejected (truly rep)",
)
ax.scatter(
    X[called & (Z == 1), 0], X[called & (Z == 1), 1], s=15, c="green", label="called correctly"
)
ax.scatter(
    X[called & (Z == 0), 0],
    X[called & (Z == 0), 1],
    s=20,
    c="red",
    marker="x",
    label="false discoveries",
)
ax.set_xlabel("rep 1")
ax.set_ylabel("rep 2")
ax.set_title(f"S1 calls at alpha = {alpha} (T = 1 path)")
ax.legend()
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Where to go next
#
# - **Run more chains and more samples.** This notebook used `num_chains=2,
#   n_samples=200` for speed. For a research-grade run use `num_chains=4,
#   n_samples=1000, n_warmup=500`.
# - **Try the other regimes.** Swap `simulate_S1` for `simulate_S2`
#   (weak signal), `simulate_S3` (sparse + heterogeneous), or
#   `simulate_S4` (skewed marginals).
# - **Switch to the multi-cluster sampler.** For S5-style data with
#   heavy-tail dependence, use `run_multi_chain` (not `run_multi_chain_simple`)
#   and inspect the posterior on $T$.
# - **Use `run_replicate_S1` for the one-liner.** Everything above is
#   bundled in `py_idr.simulation.replicates.run_replicate_S1`, which
#   returns a flat `ReplicateRow` for CSV output.
