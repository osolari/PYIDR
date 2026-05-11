# ---
# jupyter:
#   jupytext:
#     formats: py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
# ---

# %% [markdown]
# # 00 — The IDR mixture, by hand
#
# This notebook re-derives the *irreproducible discovery rate* (IDR)
# mixture in ~30 lines of numpy. The point is to give you the smallest
# possible working model so the rest of the PY-IDR machinery has
# something concrete to generalise.
#
# We do *not* import `py_idr` here — everything is from scratch. The next
# notebook (`01_gaussian_copula.py`) shows the corresponding PY-IDR
# entry points.

# %%
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as sps

# %% [markdown]
# ## The model
#
# Two replicates ($K = 2$). Each feature has a hidden indicator
# $Z_i \in \{0, 1\}$:
#
# - $Z_i = 1$ (reproducible): draw $(X_{i1}, X_{i2})$ from $\mathcal N_2(0, R)$
#   with $R = \big(\begin{smallmatrix} 1 & \rho \\ \rho & 1 \end{smallmatrix}\big)$
#   and $\rho = 0.7$.
# - $Z_i = 0$ (irreproducible): draw the two coordinates *independently*
#   from $\mathcal N(0, 1)$.
#
# The reproducible fraction is $\pi^* = 0.30$ (n = 1000).

# %%
rng = np.random.default_rng(0)

n = 1000
pi_star = 0.30
rho = 0.7
n_rep = int(pi_star * n)
n_irrep = n - n_rep

R = np.array([[1.0, rho], [rho, 1.0]])
X_rep = rng.multivariate_normal(mean=[0, 0], cov=R, size=n_rep)
X_irrep = rng.standard_normal(size=(n_irrep, 2))

X = np.vstack([X_rep, X_irrep])
Z_true = np.concatenate([np.ones(n_rep, dtype=int), np.zeros(n_irrep, dtype=int)])

# Shuffle so the block layout isn't trivially visible.
perm = rng.permutation(n)
X, Z_true = X[perm], Z_true[perm]
print("X shape:", X.shape, "  Z=1 count:", int(Z_true.sum()))

# %% [markdown]
# ## Visualising the two components
#
# Reproducible features sit along the diagonal (positive correlation).
# Irreproducible features form a noise cloud.

# %%
fig, ax = plt.subplots(figsize=(5, 5))
ax.scatter(X[Z_true == 0, 0], X[Z_true == 0, 1], s=8, alpha=0.4, label="Z=0 (noise)")
ax.scatter(X[Z_true == 1, 0], X[Z_true == 1, 1], s=8, alpha=0.7, label="Z=1 (signal)")
ax.set_xlabel("Replicate 1 score")
ax.set_ylabel("Replicate 2 score")
ax.set_aspect("equal")
ax.legend()
plt.tight_layout()
plt.show()

# %% [markdown]
# ## The local idr
#
# For a *known* $(\pi^*, \rho)$ the posterior $\PP(Z_i = 0 \mid X_i)$ —
# the **local idr** — has a closed form:
#
# $$
# \mathrm{idr}_i = \frac{(1 - \pi^*) f_0(X_i)}{(1 - \pi^*) f_0(X_i) + \pi^* f_1(X_i)},
# $$
#
# where $f_0$ is the standard-bivariate-normal density (independence)
# and $f_1$ is the correlated-bivariate-normal density.

# %%
f0 = sps.multivariate_normal(mean=[0, 0], cov=np.eye(2)).pdf
f1 = sps.multivariate_normal(mean=[0, 0], cov=R).pdf

num = (1 - pi_star) * f0(X)
den = num + pi_star * f1(X)
idr = num / den
print("idr min/mean/max:", idr.min().round(3), idr.mean().round(3), idr.max().round(3))

# %% [markdown]
# ## Sun-Cai step-up rule
#
# Sort `idr` ascending, take running mean $Q_k$, find the largest $k$
# with $Q_k \le \alpha$. That's our call set.

# %%
alpha = 0.05
order = np.argsort(idr)
idr_sorted = idr[order]
Q = np.cumsum(idr_sorted) / np.arange(1, n + 1)

mask = alpha >= Q
k_alpha = int(np.argmax(~mask)) if not mask.all() else n  # largest k with Q_k <= alpha
called_idx = order[:k_alpha]
called_mask = np.zeros(n, dtype=bool)
called_mask[called_idx] = True

realized_fdr = float(((Z_true == 0) & called_mask).sum() / max(1, called_mask.sum()))
power = float(((Z_true == 1) & called_mask).sum() / max(1, (Z_true == 1).sum()))
print(f"k_alpha = {k_alpha},  realised FDR = {realized_fdr:.3f},  power = {power:.3f}")

# %% [markdown]
# ## What you should see
#
# At $\rho = 0.7$, $\pi^* = 0.30$, $K = 2$, $n = 1000$:
#
# - **Realised FDR** is at or below the nominal 0.05.
# - **Power** is *small* — often only a handful of features get called.
#   This is correct: with only $K = 2$ replicates and moderate $\rho$,
#   the local idr's running mean exceeds 0.05 very quickly. Sun-Cai is
#   doing exactly what we want.
# - The local idr distribution is **bimodal** in shape — reproducible
#   features cluster near 0, irreproducible near 1 — but the
#   overlap between the modes at $K = 2$ keeps the *step-up* rule
#   conservative.
#
# Try bumping $K$ to 3 and observing the jump in power. Or fix $K = 2$
# and raise $\rho$ to 0.9. Both expose more separation between the
# reproducible and irreproducible local idr distributions.
#
# In PY-IDR the same Sun-Cai rule is applied to a *posterior* local idr
# obtained from the Z-trace of a Bayesian chain — the rest of the
# notebooks build up to that.
