# ---
# jupyter:
#   jupytext:
#     formats: py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
# ---

# %% [markdown]
# # 03 — Bernstein-Dirichlet marginals
#
# We:
#
# 1. Draw a heavy-tailed sample on $[0, 1]$.
# 2. Fit a Bernstein-Dirichlet marginal with a small number of basis
#    functions ($M = 8$).
# 3. Watch the conjugate update converge across a handful of sweeps.
# 4. Plot the fitted density against the truth.

# %%
from __future__ import annotations

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as sps

# %% [markdown]
# ## The data
#
# We generate $n = 1000$ samples from a Beta(2, 8) density (heavy on the
# left). The Bernstein basis should be able to recover the shape.

# %%
rng = np.random.default_rng(0)
n = 1000
U_true = rng.beta(2.0, 8.0, size=n)

# %% [markdown]
# ## The basis
#
# `py_idr.marginals.bernstein` evaluates the Bernstein basis. For each
# point $u$ and basis index $m \in \{0, \dots, M\}$:
#
# $$
# b_{m, M}(u) = \binom{M}{m} u^m (1 - u)^{M-m}.
# $$

# %%
from py_idr.marginals.bernstein import density_basis

M = 8
u_grid = np.linspace(0.001, 0.999, 200)
# density_basis returns (n, M) — basis indices r = 1, ..., M (1-based in the report).
basis = np.asarray(density_basis(jnp.asarray(u_grid), M))  # (200, M)

fig, ax = plt.subplots(figsize=(7, 4))
for m in range(M):
    ax.plot(u_grid, basis[:, m], label=f"r={m + 1}")
ax.set_title(f"Bernstein density basis (M = {M})")
ax.set_xlabel("u")
ax.set_ylabel("$b_r(u; M)$")
ax.legend(ncol=3, fontsize=8)
plt.tight_layout()
plt.show()

# %% [markdown]
# Each basis function is a tiny pmf concentrated near $m/M$. Mixing them
# with weights $w_m$ on the simplex gives a density on $[0, 1]$.

# %% [markdown]
# ## Conjugate update
#
# Initial weights = uniform. The update is:
#
# 1. Compute basis responsibilities $\gamma_{i, m} = w_m b_{m, M}(u_i) /
#    \sum_{m'} w_{m'} b_{m', M}(u_i)$.
# 2. Sum to basis counts $N_m = \sum_i \gamma_{i, m}$.
# 3. Sample $w \sim \mathrm{Dirichlet}(\beta_0 + N)$ for prior
#    concentration $\beta_0$.

# %%
from py_idr.marginals.dirichlet_weights import sample_latent_S, update_weights

alpha_F = 0.5  # weakly informative Dirichlet prior concentration
w = jnp.ones(M) / M  # weights live on the M-simplex (basis indices 1..M)
key = jax.random.PRNGKey(0)
F0_X = jnp.asarray(U_true)  # data are already on [0, 1] — use directly as the pilot CDF

snapshots = [np.asarray(w)]
for _ in range(10):
    key, sub_s, sub_w = jax.random.split(key, 3)
    # Two-step sweep: latent membership S, then conjugate Dirichlet on w.
    S = sample_latent_S(sub_s, F0_X, w, M=M)
    w = update_weights(sub_w, S, M=M, alpha_F=alpha_F)
    snapshots.append(np.asarray(w))

snapshots_arr = np.stack(snapshots)
print("Weight trace shape:", snapshots_arr.shape)  # (11, M)

# %% [markdown]
# ## Visualising convergence
#
# Plot the fitted density at iterations 0, 1, 3, 10 against the true
# Beta(2, 8) density.


# %%
def density_at(weights: np.ndarray, u_grid: np.ndarray) -> np.ndarray:
    """f(u) = sum_r w_r b_r(u; M)."""
    basis_eval = np.asarray(density_basis(jnp.asarray(u_grid), M))
    return basis_eval @ weights


true_pdf = sps.beta.pdf(u_grid, 2.0, 8.0)

fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(u_grid, true_pdf, "k", lw=2, label="truth: Beta(2, 8)")
for it in [0, 1, 3, 10]:
    ax.plot(u_grid, density_at(snapshots_arr[it], u_grid), alpha=0.7, label=f"iter {it}")
ax.set_xlabel("u")
ax.set_ylabel("density")
ax.legend()
plt.tight_layout()
plt.show()

# %% [markdown]
# After a handful of sweeps the Bernstein-Dirichlet density tracks the
# Beta(2, 8) truth closely. The remaining bias is the *truncation error*
# of degree $M = 8$ — pushing $M$ to 15 would nearly eliminate it, at
# the cost of higher Monte Carlo variance in the weights.
#
# In the full PY-IDR chain this update is one step out of eight, runs
# in conjugate closed form per replicate, and is essentially free.

# %% [markdown]
# ## Trying a harder marginal
#
# Try the same code with `U_true = rng.beta(0.5, 0.5, size=n)` —
# the arcsine distribution. The fit is much worse at $M = 8$ because
# the truth has divergences at 0 and 1; you need $M \gtrsim 30$ to
# recover it. This is the right intuition for choosing $M$ on real data.
