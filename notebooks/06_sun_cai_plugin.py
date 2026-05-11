# ---
# jupyter:
#   jupytext:
#     formats: py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
# ---

# %% [markdown]
# # 06 — Sun-Cai step-up on a synthetic local idr distribution
#
# We:
#
# 1. Generate a "plausible" local idr distribution by hand — a mixture
#    of "reproducible" features (idr near 0) and "irreproducible" features
#    (idr near 1).
# 2. Run Sun-Cai at several $\alpha$.
# 3. Sweep $\alpha$ and check the realised-FDR vs nominal-FDR curve.
#
# This isolates the decision rule from the rest of the chain so you can
# develop intuition for *what FDR control buys you* and *what it doesn't*.

# %%
from __future__ import annotations

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from py_idr.decision.sun_cai import step_up_threshold
from py_idr.simulation.evaluation import evaluate_recovery

# %% [markdown]
# ## Synthetic local idr + truth
#
# - 1000 features, 30% reproducible.
# - Reproducible idr: Beta(2, 50) — concentrated near 0 but with some
#   leakage toward higher values.
# - Irreproducible idr: Beta(50, 2) — concentrated near 1 but with some
#   leakage toward lower values.

# %%
rng = np.random.default_rng(0)
n = 1000
pi_star = 0.30
n_rep = int(pi_star * n)
n_irrep = n - n_rep

idr_rep = rng.beta(2.0, 50.0, size=n_rep)
idr_irrep = rng.beta(50.0, 2.0, size=n_irrep)
idr_all = np.concatenate([idr_rep, idr_irrep])
Z_true = np.concatenate([np.ones(n_rep, dtype=int), np.zeros(n_irrep, dtype=int)])

# Shuffle (the rule is permutation-invariant; this just stress-tests
# the implementation).
perm = rng.permutation(n)
idr_all, Z_true = idr_all[perm], Z_true[perm]

fig, ax = plt.subplots(figsize=(7, 4))
ax.hist(idr_all[Z_true == 0], bins=50, alpha=0.6, label="irreproducible (truth)")
ax.hist(idr_all[Z_true == 1], bins=50, alpha=0.6, label="reproducible (truth)")
ax.set_xlabel("local idr")
ax.set_ylabel("count")
ax.legend()
ax.set_title("Synthetic local idr distribution")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Sun-Cai at three alphas
#
# For each $\alpha$ we report $k_\alpha$ (rank cutoff), the local-idr
# threshold $\tau_\alpha$, and the call set size.

# %%
for alpha in [0.01, 0.05, 0.10]:
    dec = step_up_threshold(jnp.asarray(idr_all), alpha=alpha)
    called_idx = np.asarray(dec.selected)
    called = np.zeros(n, dtype=bool)
    called[called_idx] = True
    fdr = ((Z_true == 0) & called).sum() / max(1, called.sum())
    pwr = ((Z_true == 1) & called).sum() / max(1, (Z_true == 1).sum())
    print(
        f"alpha = {alpha:.2f}:  k_alpha = {int(dec.k_alpha):4d}  "
        f"gamma_alpha = {float(dec.gamma_alpha):.3f}  "
        f"called = {called.sum():4d}  "
        f"realised FDR = {fdr:.3f}  power = {pwr:.3f}"
    )

# %% [markdown]
# ## Calibration: sweep $\alpha$
#
# Average realised FDR over 30 Monte Carlo replicates of the same setup;
# plot against nominal $\alpha$.


# %%
def one_replicate(seed: int, n: int, pi_star: float) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    n_rep = int(pi_star * n)
    n_irrep = n - n_rep
    idr_rep = rng.beta(2.0, 50.0, size=n_rep)
    idr_irrep = rng.beta(50.0, 2.0, size=n_irrep)
    idr = np.concatenate([idr_rep, idr_irrep])
    Z = np.concatenate([np.ones(n_rep, dtype=int), np.zeros(n_irrep, dtype=int)])
    perm = rng.permutation(n)
    return idr[perm], Z[perm]


alpha_grid = np.linspace(0.01, 0.20, 20)
fdr_grid = np.zeros_like(alpha_grid)
pwr_grid = np.zeros_like(alpha_grid)
n_rep_mc = 30

for s in range(n_rep_mc):
    idr_s, Z_s = one_replicate(s, n=1000, pi_star=0.30)
    for i, alpha in enumerate(alpha_grid):
        rec = evaluate_recovery(jnp.asarray(idr_s), jnp.asarray(Z_s), alpha=float(alpha))
        fdr_grid[i] += rec.realized_fdr / n_rep_mc
        pwr_grid[i] += rec.power / n_rep_mc

fig, axes = plt.subplots(1, 2, figsize=(13, 4))
axes[0].plot(alpha_grid, fdr_grid, "o-", label="realised FDR")
axes[0].plot(alpha_grid, alpha_grid, "k--", label="nominal $\\alpha$")
axes[0].set_xlabel("nominal $\\alpha$")
axes[0].set_ylabel("realised FDR")
axes[0].legend()
axes[0].set_title("FDR calibration")

axes[1].plot(alpha_grid, pwr_grid, "o-")
axes[1].set_xlabel("nominal $\\alpha$")
axes[1].set_ylabel("power")
axes[1].set_title("Power vs alpha")
plt.tight_layout()
plt.show()

# %% [markdown]
# **What you should see.** The realised-FDR curve sits *at or slightly
# below* the nominal $\alpha$ line — Sun-Cai is slightly conservative
# for this well-separated distribution. Power rises sharply with $\alpha$
# and saturates around $\alpha = 0.10$.
#
# In real PY-IDR runs the local idr is a posterior summary; the same
# Sun-Cai rule applies, with the same calibration guarantees *as long
# as the chain's posterior is well-calibrated*. The diagnostic verdict
# (Notebook 07) is what tells you whether to trust it.
