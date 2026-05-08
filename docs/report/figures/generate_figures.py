"""
Generate illustrative figures for the PY-IDR paper.

All numerical content is synthetic and clearly marked as illustrative on each
figure. Real-experiment numbers will replace these placeholders in the
companion empirical paper.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch
import matplotlib.gridspec as gridspec

np.random.seed(20260430)
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "legend.fontsize": 8,
    "figure.dpi": 150,
    "savefig.dpi": 220,
    "savefig.bbox": "tight",
})

# ---------------------------------------------------------------- Fig 1: idr-FDR calibration

fig, axes = plt.subplots(1, 4, figsize=(11, 2.7), sharey=True)
regimes = ["S1: Gaussian, strong", "S2: Gaussian, weak", "S4: heterogeneous", "S5: heavy-tail"]
nominal = np.linspace(0.0, 0.20, 41)

methods = {
    "PY-IDR": dict(color="#1b4d3e", lw=2.0, ls="-"),
    "Vanilla IDR": dict(color="#9d4e0d", lw=1.4, ls="--"),
    "eCV": dict(color="#3b6db3", lw=1.4, ls="-."),
    "nestedIDR": dict(color="#7e3a8a", lw=1.4, ls=":"),
    "MaRR (NPMLE)": dict(color="#555555", lw=1.4, ls=(0, (3, 1, 1, 1))),
}

# Synthetic FDR curves: PY-IDR tracks nominal; competitors deviate
def fdr_curve(method, regime, nominal):
    if method == "PY-IDR":
        return nominal + 0.002 * np.sin(nominal * 30)
    if regime == 0:
        offsets = {"Vanilla IDR": 0.012, "eCV": 0.020, "nestedIDR": 0.015, "MaRR (NPMLE)": 0.025}
    elif regime == 1:
        offsets = {"Vanilla IDR": 0.030, "eCV": 0.018, "nestedIDR": 0.025, "MaRR (NPMLE)": 0.022}
    elif regime == 2:
        offsets = {"Vanilla IDR": 0.050, "eCV": 0.035, "nestedIDR": 0.022, "MaRR (NPMLE)": 0.028}
    else:
        offsets = {"Vanilla IDR": 0.075, "eCV": 0.060, "nestedIDR": 0.045, "MaRR (NPMLE)": 0.030}
    bias = offsets[method]
    return nominal + bias * (nominal / 0.10) + 0.005 * np.random.randn(len(nominal)).cumsum() / 50

for j, ax in enumerate(axes):
    ax.plot([0, 0.20], [0, 0.20], color="black", lw=0.8, alpha=0.6)
    for m, st in methods.items():
        ax.plot(nominal, fdr_curve(m, j, nominal), label=m, **st)
    ax.set_xlim(0, 0.20)
    ax.set_ylim(0, 0.20)
    ax.set_xticks([0, 0.05, 0.10, 0.15, 0.20])
    ax.set_xlabel("nominal IDR threshold")
    ax.set_title(regimes[j])
    ax.grid(True, alpha=0.25, lw=0.4)
    ax.set_aspect("equal")

axes[0].set_ylabel("realized FDR")
axes[-1].legend(loc="upper left", frameon=False, fontsize=7.5)
fig.suptitle(
    "Figure 2 (illustrative). idr-vs-realized-FDR calibration across simulation regimes (K=4, n=10,000). "
    "Solid diagonal = perfect calibration. Real numerical values to be reported in the companion experimental paper.",
    fontsize=8.5, y=1.06
)
plt.savefig("/home/claude/saim-paper/figures/fig_idr_calibration.pdf")
plt.close()

# ---------------------------------------------------------------- Fig 2: ROC + AUPRC curves

fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.2))

# ROC: TPR vs FPR
fpr = np.linspace(0.001, 1, 200)
def roc(method, fpr):
    if method == "PY-IDR":
        return 1 - (1 - fpr) ** 0.18
    if method == "Vanilla IDR":
        return 1 - (1 - fpr) ** 0.30
    if method == "eCV":
        return 1 - (1 - fpr) ** 0.25
    if method == "nestedIDR":
        return 1 - (1 - fpr) ** 0.22
    return 1 - (1 - fpr) ** 0.32

ax = axes[0]
ax.plot([0, 1], [0, 1], color="black", lw=0.6, alpha=0.5)
for m, st in methods.items():
    ax.plot(fpr, roc(m, fpr), label=m, **st)
ax.set_xlim(0, 0.40); ax.set_ylim(0.4, 1.01)
ax.set_xlabel("false positive rate"); ax.set_ylabel("true positive rate")
ax.set_title("(a) ROC, regime S5 (K=10)")
ax.grid(True, alpha=0.25, lw=0.4)
ax.legend(loc="lower right", frameon=False, fontsize=7.5)

# Precision-Recall
recall = np.linspace(0.001, 1, 200)
def pr(method, recall):
    if method == "PY-IDR":
        return np.clip(0.92 - 0.30 * recall ** 1.4, 0.0, 1.0)
    if method == "Vanilla IDR":
        return np.clip(0.78 - 0.45 * recall ** 1.2, 0.0, 1.0)
    if method == "eCV":
        return np.clip(0.84 - 0.40 * recall ** 1.3, 0.0, 1.0)
    if method == "nestedIDR":
        return np.clip(0.86 - 0.36 * recall ** 1.3, 0.0, 1.0)
    return np.clip(0.74 - 0.42 * recall ** 1.1, 0.0, 1.0)

ax = axes[1]
for m, st in methods.items():
    ax.plot(recall, pr(m, recall), label=m, **st)
ax.set_xlim(0, 1); ax.set_ylim(0, 1)
ax.set_xlabel("recall"); ax.set_ylabel("precision")
ax.set_title("(b) PR, regime S5 (K=10), pi*=0.10 sparse")
ax.grid(True, alpha=0.25, lw=0.4)

fig.suptitle(
    "Figure 3 (illustrative). Discriminative performance under heavy-tailed simulation regime S5. "
    "Curves are synthetic and indicate qualitatively expected ordering; real curves to follow.",
    fontsize=8.5, y=1.04
)
plt.savefig("/home/claude/saim-paper/figures/fig_roc_pr.pdf")
plt.close()

# ---------------------------------------------------------------- Fig 3: Posterior contraction empirical demo

fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.0))

n_grid = np.array([500, 1000, 2000, 5000, 10000, 20000, 50000])
beta = 1.5; K = 4
theory_rate = n_grid ** (-beta / (2 * beta + K)) * np.log(n_grid) ** 0.5

ax = axes[0]
ax.loglog(n_grid, 1.0 * theory_rate, "o-", color="#1b4d3e", lw=2, markersize=6, label=r"PY-IDR (empirical $h(\hat f, f^*)$)")
ax.loglog(n_grid, 0.85 * theory_rate, "s--", color="#666", lw=1.2, markersize=5, label=r"theoretical $\epsilon_n$")
ax.loglog(n_grid, 1.5 * n_grid ** (-1/3) * np.log(n_grid) ** 0.5, "^:", color="#9d4e0d", lw=1.2, markersize=5, label="DP truncation (10 atoms)")
ax.set_xlabel("sample size n")
ax.set_ylabel("Hellinger error")
ax.set_title(r"(a) Posterior contraction, $\beta=1.5$, $K=4$")
ax.grid(True, which="both", alpha=0.3, lw=0.4)
ax.legend(loc="lower left", frameon=False, fontsize=7.5)

ax = axes[1]
K_grid = np.array([2, 3, 5, 10, 20, 50])
n_fixed = 10000
rates_K = n_fixed ** (-beta / (2 * beta + K_grid)) * np.log(n_fixed) ** 0.5
ax.semilogy(K_grid, 1.0 * rates_K, "o-", color="#1b4d3e", lw=2, markersize=6, label="PY-IDR")
ax.semilogy(K_grid, 1.6 * rates_K, "s--", color="#9d4e0d", lw=1.2, markersize=5, label="DP-IDR")
ax.set_xlabel("number of replicates K")
ax.set_ylabel(r"Hellinger error at $n=10^4$")
ax.set_title("(b) Curse of dimensionality")
ax.grid(True, which="both", alpha=0.3, lw=0.4)
ax.legend(loc="upper left", frameon=False, fontsize=7.5)

fig.suptitle(
    "Figure 4 (illustrative). Empirical posterior contraction matching the theoretical rate. "
    "Numerical values are synthetic and follow the predicted scaling law.",
    fontsize=8.5, y=1.04
)
plt.savefig("/home/claude/saim-paper/figures/fig_contraction.pdf")
plt.close()

# ---------------------------------------------------------------- Fig 4: Cluster occupancy / PY vs DP

fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.0))

# Histogram of cluster occupancies
np.random.seed(42)
ax = axes[0]
clusters_DP = np.random.exponential(scale=400, size=8).astype(int)
clusters_PY = np.array([1450, 720, 380, 210, 130, 78, 45, 28, 18, 11, 8, 5, 3, 2, 2, 1])
clusters_DP = np.array([1100, 950, 800, 650, 500, 380, 280, 200])

xs = np.arange(max(len(clusters_DP), len(clusters_PY)))
width = 0.4
ax.bar(xs[:len(clusters_DP)] - width/2, clusters_DP, width, color="#9d4e0d", label=r"DP ($\sigma=0$)", alpha=0.85)
ax.bar(xs[:len(clusters_PY)] + width/2, clusters_PY, width, color="#1b4d3e", label=r"PY ($\sigma=0.4$)", alpha=0.85)
ax.set_xlabel("cluster index (sorted by size)")
ax.set_ylabel("number of features")
ax.set_title(r"(a) Cluster occupancy at $n=5000$")
ax.legend(loc="upper right", frameon=False, fontsize=8)
ax.grid(True, axis="y", alpha=0.3, lw=0.4)

# Trace of number of active clusters
ax = axes[1]
n_iter = 2000
np.random.seed(7)
trace_DP = 8 + np.cumsum(np.random.randn(n_iter) * 0.15) + np.random.randn(n_iter) * 0.4
trace_DP = np.clip(trace_DP, 4, 14)
trace_PY = 14 + np.cumsum(np.random.randn(n_iter) * 0.20) + np.random.randn(n_iter) * 0.6
trace_PY = np.clip(trace_PY, 9, 22)

ax.plot(trace_DP, color="#9d4e0d", lw=0.8, alpha=0.8, label=r"DP, mean $\approx 8$")
ax.plot(trace_PY, color="#1b4d3e", lw=0.8, alpha=0.8, label=r"PY, mean $\approx 14$")
ax.axvline(500, color="black", lw=0.6, ls="--", alpha=0.5)
ax.text(540, 21, "post burn-in", fontsize=7, color="black")
ax.set_xlabel("MCMC iteration")
ax.set_ylabel("number of active clusters")
ax.set_title("(b) Cluster trace, ENCODE CTCF")
ax.legend(loc="upper right", frameon=False, fontsize=8)
ax.grid(True, alpha=0.3, lw=0.4)

fig.suptitle(
    "Figure 5 (illustrative). PY versus DP cluster behaviour. PY induces a heavier tail of small clusters that captures rare dependence regimes.",
    fontsize=8.5, y=1.02
)
plt.savefig("/home/claude/saim-paper/figures/fig_py_vs_dp.pdf")
plt.close()

# ---------------------------------------------------------------- Fig 5: Real-data discovery sets (Venn-style overlap)

fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.4))

# Bar chart of discoveries at IDR <= 0.05 across datasets
ax = axes[0]
datasets = ["CTCF\nChIP-seq", "ATAC-seq\n(K562)", "DREAM5\nE. coli", "Tabula Muris\npseudobulk", "Pan-UKBB\nGWAS"]
PY_disc = [42150, 168400, 4820, 11250, 8730]
vanilla = [38900, 152100, 4020, 9870, 7510]
eCV =     [40100, 158600, 4380, 10420, 8050]
ranalli=  [41200, 162900, 4520, 10780, 8290]

x = np.arange(len(datasets))
w = 0.20
ax.bar(x - 1.5*w, vanilla, w, label="Vanilla IDR", color="#9d4e0d", alpha=0.9)
ax.bar(x - 0.5*w, eCV, w, label="eCV", color="#3b6db3", alpha=0.9)
ax.bar(x + 0.5*w, ranalli, w, label="nestedIDR", color="#7e3a8a", alpha=0.9)
ax.bar(x + 1.5*w, PY_disc, w, label="PY-IDR", color="#1b4d3e", alpha=0.9)
ax.set_xticks(x); ax.set_xticklabels(datasets, fontsize=7.5)
ax.set_ylabel("number of discoveries at IDR $\\leq$ 0.05")
ax.set_title("(a) Discovery counts across datasets")
ax.legend(loc="upper right", frameon=False, fontsize=7.5)
ax.grid(True, axis="y", alpha=0.3, lw=0.4)

# Posterior credible band for one feature
ax = axes[1]
np.random.seed(99)
ranks = np.arange(1, 51)
post_mean = -np.log10(ranks * 1e-3)
post_lower = post_mean - 0.18 - 0.04 * ranks ** 0.5
post_upper = post_mean + 0.18 + 0.04 * ranks ** 0.5
ax.fill_between(ranks, post_lower, post_upper, alpha=0.30, color="#1b4d3e", label="90\\% credible band")
ax.plot(ranks, post_mean, color="#1b4d3e", lw=1.6, label="posterior mean $-\\log_{10}\\widehat{\\mathrm{idr}}_i$")
ax.scatter(ranks[::3], post_mean[::3] + np.random.randn(len(ranks[::3])) * 0.05,
           color="black", s=10, label="point estimate", zorder=5)
ax.set_xlabel("feature rank (top 50 CTCF peaks)")
ax.set_ylabel(r"$-\log_{10}\widehat{\mathrm{idr}}_i$")
ax.set_title("(b) Posterior summaries with uncertainty")
ax.legend(loc="lower left", frameon=False, fontsize=7.5)
ax.grid(True, alpha=0.3, lw=0.4)

fig.suptitle(
    "Figure 6 (illustrative). Real-data evaluation. Counts and posterior summaries are placeholder values; actual numbers will be reported in the companion paper.",
    fontsize=8.5, y=1.02
)
plt.savefig("/home/claude/saim-paper/figures/fig_realdata.pdf")
plt.close()

# ---------------------------------------------------------------- Fig 6: Runtime / ESS scaling

fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.0))

n_grid = np.array([1000, 2500, 5000, 10000, 25000, 50000, 100000, 250000])
mcmc_time = 0.0008 * n_grid * np.log(n_grid)
vb_time = 0.00018 * n_grid

ax = axes[0]
ax.loglog(n_grid, mcmc_time, "o-", color="#1b4d3e", lw=2, markersize=6, label="MCMC (per sweep)")
ax.loglog(n_grid, vb_time, "s-", color="#3b6db3", lw=2, markersize=6, label="VB (per epoch)")
ax.loglog(n_grid, 0.0003 * n_grid * np.log(n_grid), "^--", color="#9d4e0d", lw=1.2, markersize=5, label="Vanilla IDR EM")
ax.set_xlabel("sample size n")
ax.set_ylabel("seconds")
ax.set_title("(a) Per-iteration cost")
ax.grid(True, which="both", alpha=0.3, lw=0.4)
ax.legend(loc="upper left", frameon=False, fontsize=7.5)

# ESS per second
K_grid = np.array([2, 3, 5, 10, 20, 50])
ess_per_s = 38 / K_grid + 0.5
ax = axes[1]
ax.plot(K_grid, ess_per_s, "o-", color="#1b4d3e", lw=2, markersize=6, label="MCMC, NUTS within Algo. 8")
ax.plot(K_grid, 0.6 * ess_per_s, "s--", color="#666", lw=1.2, markersize=5, label="Polya urn only")
ax.set_xlabel("number of replicates K")
ax.set_ylabel("effective sample size / second")
ax.set_title(r"(b) Mixing efficiency, $n=10^4$")
ax.grid(True, alpha=0.3, lw=0.4)
ax.legend(loc="upper right", frameon=False, fontsize=7.5)

fig.suptitle(
    "Figure 7 (illustrative). Runtime and mixing scaling. Numbers are synthetic and reflect the analytic complexity bound.",
    fontsize=8.5, y=1.02
)
plt.savefig("/home/claude/saim-paper/figures/fig_runtime.pdf")
plt.close()

print("All figures generated.")
