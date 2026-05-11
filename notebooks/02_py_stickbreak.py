# ---
# jupyter:
#   jupytext:
#     formats: py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
# ---

# %% [markdown]
# # 02 — Pitman-Yor stick-breaking
#
# We draw stick-breaking weights from the Pitman-Yor process and
# visualise:
#
# 1. How the weights decay along the stick.
# 2. The induced prior on the number of clusters $T$ in a sample of
#    size $n$.
# 3. How the discount parameter $\sigma$ controls the power-law tail.

# %%
from __future__ import annotations

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

# %% [markdown]
# ## Stick-breaking weights
#
# Recall: $V_t \sim \mathrm{Beta}(1 - \sigma, \alpha + t \sigma)$, then
# $w_t = V_t \prod_{s < t}(1 - V_s)$. PY-IDR exposes this as
# :func:`py_idr.pym.stickbreak.gem_sample`.
# %%
from py_idr.pym.stickbreak import gem_sample

K_trunc = 30  # truncation level for visualisation
n_draws = 500
key = jax.random.PRNGKey(0)

settings = [
    ("DP (sigma=0, alpha=1)", 1.0, 0.0),
    ("PY (sigma=0.5, alpha=1)", 1.0, 0.5),
    ("PY (sigma=0.7, alpha=1)", 1.0, 0.7),
]

fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
for ax, (label, alpha, sigma) in zip(axes, settings, strict=False):
    sub_keys = jax.random.split(key, n_draws)
    W = np.stack([np.asarray(gem_sample(k, alpha, sigma, K_trunc)) for k in sub_keys])
    ax.plot(np.arange(1, K_trunc + 1), W.mean(axis=0), "o-", label="mean weight")
    ax.fill_between(
        np.arange(1, K_trunc + 1),
        np.quantile(W, 0.1, axis=0),
        np.quantile(W, 0.9, axis=0),
        alpha=0.2,
        label="10-90% band",
    )
    ax.set_title(label)
    ax.set_xlabel("stick index $t$")
    ax.set_yscale("log")
    ax.legend()
axes[0].set_ylabel("$w_t$")
plt.tight_layout()
plt.show()

# %% [markdown]
# **Take-away.** At $\sigma = 0$ (Dirichlet process) the weights decay
# exponentially. At $\sigma > 0$ they decay like a power law — more mass
# in the tail, meaning more "rare" clusters get non-negligible
# probability a priori. This is what PY brings over DP for the IDR
# multi-cluster case.

# %% [markdown]
# ## Prior on $T$, the number of clusters in $n$ samples
#
# Given the PY weights, the number of *distinct* clusters that appear in
# a sample of size $n$ is itself random. The CRP simulation below draws
# the partition and counts blocks.


# %%
def crp_block_count(alpha: float, sigma: float, n: int, rng: np.random.Generator) -> int:
    """Simulate the Chinese-restaurant analogue and return the block count."""
    counts: list[int] = []
    for i in range(n):
        T = len(counts)
        denom = alpha + i
        probs = np.array([(c - sigma) / denom for c in counts] + [(alpha + T * sigma) / denom])
        probs = probs / probs.sum()
        t = int(np.random.default_rng(rng.integers(0, 2**31)).choice(T + 1, p=probs))
        if t == T:
            counts.append(1)
        else:
            counts[t] += 1
    return len(counts)


n = 200
n_sim = 200
rng = np.random.default_rng(42)

fig, ax = plt.subplots(figsize=(7, 4))
for alpha, sigma, label in [
    (1.0, 0.0, "DP a=1"),
    (1.0, 0.5, "PY a=1, s=0.5"),
    (1.0, 0.7, "PY a=1, s=0.7"),
]:
    Ts = [crp_block_count(alpha, sigma, n, rng) for _ in range(n_sim)]
    ax.hist(Ts, bins=range(1, max(Ts) + 2), alpha=0.5, label=label, density=True)
ax.set_xlabel("$T$ (number of clusters in n = 200)")
ax.set_ylabel("density")
ax.legend()
plt.tight_layout()
plt.show()

# %% [markdown]
# **Take-away.** PY puts mass on a much wider range of $T$ than DP does
# at the same concentration $\alpha$. For PY-IDR this matters when the
# data really do contain multiple copula types — the prior is happy to
# spawn new clusters as evidence accumulates.

# %% [markdown]
# ## The EPPF as a function of $(\alpha, \sigma)$
#
# Hold a partition fixed; vary the PY hyperparameters; see how the log
# EPPF changes. This is the surface that NUTS samples in step 7 of
# Algorithm 1.

# %%
from py_idr.pym.eppf import log_eppf  # type: ignore[attr-defined]

# A partition of n = 100 into 5 clusters of sizes (40, 30, 15, 10, 5).
cluster_sizes = jnp.array([40, 30, 15, 10, 5])
n = int(cluster_sizes.sum())

sigma_grid = np.linspace(0.01, 0.9, 50)
alpha_fixed = 1.0
logE = np.array([float(log_eppf(alpha_fixed, float(s), cluster_sizes, n)) for s in sigma_grid])

fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(sigma_grid, logE)
ax.set_xlabel("$\\sigma$")
ax.set_ylabel("log EPPF")
ax.set_title("log EPPF surface in $\\sigma$ (alpha=1, partition of 100)")
plt.tight_layout()
plt.show()

# %% [markdown]
# The EPPF favours intermediate $\sigma$ for this partition — extreme
# values of $\sigma$ are penalised. NUTS will track this surface
# faithfully because it's smooth and concave-looking.
