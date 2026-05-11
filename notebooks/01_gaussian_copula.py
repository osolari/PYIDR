# ---
# jupyter:
#   jupytext:
#     formats: py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
# ---

# %% [markdown]
# # 01 — The Gaussian copula
#
# We build intuition for the Gaussian copula by:
#
# 1. Drawing from $\mathcal N_K(0, R)$, then applying $\Phi$ per coordinate
#    to land on the copula scale $[0, 1]^K$.
# 2. Visualising the joint copula density on the unit square.
# 3. Mapping back to the observed scale via $\Phi^{-1}$ — recovering the
#    original Gaussian sample (Sklar at work).
# 4. Computing the copula log-density via PY-IDR.

# %%
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as sps

# %%
rng = np.random.default_rng(0)
K = 2
n = 5000
rho = 0.8

# Step 1: draw latent normals
R = (1 - rho) * np.eye(K) + rho * np.ones((K, K))
Z = rng.multivariate_normal(mean=np.zeros(K), cov=R, size=n)

# Step 2: copula scale — Phi(Z) is uniform per coordinate
U = sps.norm.cdf(Z)

# Step 3: map back via inverse CDF of arbitrary marginal.
# Here we use chi-square(3) for replicate 0 and exponential(1) for replicate 1.
X_rep0 = sps.chi2.ppf(U[:, 0], df=3)
X_rep1 = sps.expon.ppf(U[:, 1])

# %%
fig, axes = plt.subplots(1, 3, figsize=(13, 4))
axes[0].scatter(Z[:, 0], Z[:, 1], s=4, alpha=0.3)
axes[0].set_title(f"Latent: $\\mathcal{{N}}_2(0, R), \\rho = {rho}$")
axes[0].set_xlabel("$Z_1$")
axes[0].set_ylabel("$Z_2$")
axes[0].set_aspect("equal")

axes[1].scatter(U[:, 0], U[:, 1], s=4, alpha=0.3)
axes[1].set_title("Copula scale: $U = \\Phi(Z)$")
axes[1].set_xlabel("$U_1$")
axes[1].set_ylabel("$U_2$")
axes[1].set_aspect("equal")

axes[2].scatter(X_rep0, X_rep1, s=4, alpha=0.3)
axes[2].set_title("Marginals: $\\chi^2_3$ and $\\mathrm{Exp}(1)$")
axes[2].set_xlabel("$X_1$")
axes[2].set_ylabel("$X_2$")

plt.tight_layout()
plt.show()

# %% [markdown]
# Notice the middle panel: the marginals are perfectly uniform on $[0, 1]$
# (probability integral transform), but the dependence pattern of the
# Gaussian copula is preserved. The right panel shows the *same* dependence,
# now mapped to two very different marginal distributions.
#
# This is the entire point of copulas: dependence and marginals are
# separable concerns.

# %% [markdown]
# ## Computing the copula log-density via PY-IDR
#
# `py_idr.copulas.gaussian` exposes the Gaussian copula density. The
# inputs are on the copula scale (i.e. `U`, not `Z`).

# %%
import jax.numpy as jnp

from py_idr.copulas.gaussian import GaussianCopula

L = np.linalg.cholesky(R)
copula = GaussianCopula(jnp.asarray(L))
# Clip U away from {0, 1} to avoid Phi^{-1} infinities at the endpoints.
U_safe = jnp.clip(jnp.asarray(U), 1e-6, 1 - 1e-6)
log_pdf = np.asarray(copula.log_density(U_safe))
print("log_pdf shape:", log_pdf.shape)
print("mean log copula density at rho =", rho, ":", float(log_pdf.mean()))

# %% [markdown]
# Under the independence copula, $\log c(U) = 0$ for every $U$ (the
# density is the constant 1 on the unit cube). The mean log-density above
# is positive, reflecting that the data have non-trivial positive
# dependence under the Gaussian copula at $\rho = 0.8$.

# %% [markdown]
# ## What changes for $K \ge 3$
#
# Everything above generalises. The only $K$-specific moving part is
# the correlation matrix: at $K = 2$ it is one scalar; at $K \ge 3$ it
# is an LKJ-distributed positive-definite matrix, optionally
# PLOD-restricted (see `04_polya_urn_step_by_step.py` and the
# [LKJ + PLOD primer](../docs/math/lkj_structured.md)).
