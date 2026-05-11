# ---
# jupyter:
#   jupytext:
#     formats: py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
# ---

# %% [markdown]
# # 04 — Pólya urn cluster reassignment, step by step
#
# We trace one Pólya-urn cluster reassignment for a single feature, by
# hand. The point is to make Algorithm 1 step 2 concrete: there is no
# magic, it is just a Categorical draw whose probabilities come from
# the PY predictive plus the current copula densities.

# %%
from __future__ import annotations

import jax.numpy as jnp
import numpy as np

# %% [markdown]
# ## The setup
#
# - Two existing clusters, sizes 60 and 40.
# - Both Gaussian-copula with different $\rho$:
#   cluster 1: $\rho = 0.85$, cluster 2: $\rho = 0.40$.
# - PY hyperparameters: $\alpha = 1.0, \sigma = 0.5$.
# - The feature we are reassigning has copula-scale data
#   $u_i = (0.85, 0.80, 0.75)$ — a moderately reproducible signal across
#   $K = 3$ replicates.
# %%
from py_idr.copulas.gaussian import GaussianCopula

alpha_PY = 1.0
sigma_PY = 0.5

n_total = 100  # total reproducible features
counts = np.array([60, 40])  # currently in clusters 1 and 2
K = 3
u_i = jnp.array([[0.85, 0.80, 0.75]])  # shape (1, K)


def build_R(rho: float, K: int) -> np.ndarray:
    return (1 - rho) * np.eye(K) + rho * np.ones((K, K))


L_cluster1 = jnp.asarray(np.linalg.cholesky(build_R(0.85, K)))
L_cluster2 = jnp.asarray(np.linalg.cholesky(build_R(0.40, K)))

log_c1 = float(GaussianCopula(L_cluster1).log_density(u_i)[0])
log_c2 = float(GaussianCopula(L_cluster2).log_density(u_i)[0])
print(f"log c_1(u_i) = {log_c1:.3f}  (cluster 1, strong dependence)")
print(f"log c_2(u_i) = {log_c2:.3f}  (cluster 2, weak dependence)")

# %% [markdown]
# Cluster 1 (strong dependence) puts much higher likelihood on the
# observation — unsurprising, since $u_i$ is clearly correlated across
# replicates.

# %% [markdown]
# ## The PY predictive prefactors
#
# Suppose we are reassigning the feature *after* removing it from its
# current cluster. The PY predictive for each option is:
#
# - existing cluster $t$: $\dfrac{n_t - \sigma}{\alpha + n - 1}$
# - new cluster: $\dfrac{\alpha + T \sigma}{\alpha + n - 1}$
#
# where $T$ is the current cluster count.

# %%
# Assume the feature was previously in cluster 1.
# After removal, n_1 = 59, n_2 = 40, n_total - 1 = 99.
n_after_remove = n_total - 1
T = 2

pre_c1 = (counts[0] - 1 - sigma_PY) / (alpha_PY + n_after_remove)
pre_c2 = (counts[1] - sigma_PY) / (alpha_PY + n_after_remove)
pre_new = (alpha_PY + T * sigma_PY) / (alpha_PY + n_after_remove)
print("PY prefactors:")
print(f"  cluster 1: {pre_c1:.4f}")
print(f"  cluster 2: {pre_c2:.4f}")
print(f"  new      : {pre_new:.4f}")

# %% [markdown]
# ## New-cluster marginal likelihood
#
# For the "new cluster" option we need the integral
# $\int c_\theta(u_i) G_0(d\theta)$. PY-IDR approximates this by drawing
# one $\theta_{\text{new}}$ from the base measure and using
# $c_{\theta_{\text{new}}}(u_i)$ as the estimate. We do the same here.

# %%
rng = np.random.default_rng(0)
# Sample a fresh rho from a weakly-informative prior (Uniform on (0, 1)).
rho_new = float(rng.uniform(0.05, 0.95))
L_new = jnp.asarray(np.linalg.cholesky(build_R(rho_new, K)))
log_c_new = float(GaussianCopula(L_new).log_density(u_i)[0])
print(f"sampled rho_new = {rho_new:.3f}, log c_new(u_i) = {log_c_new:.3f}")

# %% [markdown]
# ## Combine into a Categorical
#
# Multiply prefactor times likelihood, normalise.

# %%
log_options = np.array(
    [
        np.log(pre_c1) + log_c1,
        np.log(pre_c2) + log_c2,
        np.log(pre_new) + log_c_new,
    ]
)
log_options -= log_options.max()
probs = np.exp(log_options)
probs /= probs.sum()
print("Reassignment probabilities:")
print(f"  cluster 1: {probs[0]:.3f}")
print(f"  cluster 2: {probs[1]:.3f}")
print(f"  new      : {probs[2]:.3f}")

# %% [markdown]
# Cluster 1 dominates — both the PY predictive prefactor and the copula
# likelihood favour it. The new-cluster option has tiny mass because the
# sampled $\rho_{\text{new}}$ happened to be a bad fit; another sample
# might give a different number.

# %% [markdown]
# ## What changes per feature
#
# Three things vary across features within one sweep:
#
# 1. The copula densities $c_t(u_i)$ — depend on the feature's $u_i$.
# 2. The cluster counts — *if* a previous feature in this sweep was
#    reassigned to a different cluster, the counts shift.
# 3. The PY prefactor's denominator $\alpha + n - 1$ — constant within
#    one sweep.
#
# In code, the per-feature loop is in Python (the atom set is
# variable-length), but each iteration is a small batched JAX computation
# of copula densities, plus the Categorical draw.

# %% [markdown]
# ## A common misconception
#
# *"Doesn't this assume the order of features matters?"* No — the Gibbs
# sweep is exact only because at each step we condition on **everything
# else**. After many sweeps the posterior is invariant under feature
# permutations. The shuffled order at the top of the sweep is for
# *mixing speed*, not correctness.
