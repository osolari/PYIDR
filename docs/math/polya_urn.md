# Pólya urn predictive (step-by-step)

This page works through the Pólya-urn cluster-reassignment step that
appears at the top of the multi-cluster Gibbs sweep in
[`py_idr.inference.mcmc.sweep_multi`][]. The formulas are in the
[Pitman-Yor primer](pitman_yor.md); the goal here is the *operational*
sequence, with the precise edge cases that the code handles.

## What this step does

After step 1 (resampling $Z_i$) we know which features are
"reproducible." Step 2 re-distributes those features among the active
clusters and lets some of them spawn new clusters. The mechanism is a
**Gibbs sweep** over the cluster assignments $L_i$, one feature at a
time, with each conditional given by the PY predictive after *removing*
feature $i$ from its current cluster.

## Pseudocode

```
inputs:  L   ∈ {1, ..., T+}^n_rep         current cluster labels
         U   ∈ [0, 1]^{n_rep × K}         copula-scale data
         atoms = {θ_t : t = 1, ..., T+}    cluster parameters
         α, σ                              PY hyperparameters

for i in shuffle(indices with Z_i = 1):
    t_old = L[i]
    # 1. Remove feature i from its current cluster.
    n_{t_old} -= 1
    if n_{t_old} == 0:
        drop cluster t_old; relabel; T+ -= 1

    # 2. Build the unnormalised posterior predictive across (T+ + 1) options.
    for t = 1, ..., T+:
        log_p[t] = log(n_t - σ) - log(α + n_rep - 1) + log c_t(U_i)
    log_p[new] = log(α + T+ * σ) - log(α + n_rep - 1) + log m_new(U_i)
    normalise log_p  → p

    # 3. Sample new label.
    t_new = Categorical(p)

    if t_new is "new":
        θ_new ~ G_0
        T+ += 1
        L[i] = T+
        atoms[T+] = θ_new
    else:
        L[i] = t_new

    n_{L[i]} += 1
```

The two ingredients are:

- $c_t(U_i)$ — the **density of cluster $t$**'s copula evaluated at
  $U_i$. Lives in `copula.density(U_i, θ_t)`.
- $m_{\text{new}}(U_i)$ — the **marginal likelihood under the base
  measure** $G_0$. This is the integral
  $\int c_\theta(U_i) \, G_0(d\theta)$ — the predictive of a fresh
  cluster. PY-IDR approximates this with a single Monte Carlo draw
  from $G_0$, which is biased but in practice fine because new-cluster
  proposals are rare and the next sweep refines them.

## Why this works

The Pólya-urn predictive *is* the conditional of $L_i$ given the rest
under the PY prior — this is the Blackwell-MacQueen characterisation of
PY. So running a Gibbs sweep over the $L_i$'s leaves the posterior
invariant *exactly* (modulo the $m_{\text{new}}$ approximation).

The shuffle at the top is for mixing: deterministic order can lead to
slow autocorrelation when the data have block structure.

## Edge cases the code handles

1. **Empty clusters after removal.** Whenever $n_{t_{\text{old}}}$ hits 0
   we drop that cluster and relabel. The atoms tuple in
   `MultiClusterState` is variable-length, so we can do this cheaply.

2. **Stale counts.** Step 1 of the sweep (class assignment) flips some
   $Z_i$'s. Before entering the Pólya-urn step we recompute
   $\{n_t\}_{t}$ from $L$ filtered by the current $Z$ — see
   `_counts_from_z` inside `sweep_multi`. Earlier versions of the code
   had a subtle bug where stale counts were reused; the fix is the
   recompute.

3. **Singleton last cluster.** If after the removal the *only* feature
   left in cluster $t_{\text{old}}$ was the one we just removed (so
   $n_{t_{\text{old}}}$ went from 1 to 0), we drop it. The next sample
   then either lands in an existing cluster or proposes a new one;
   either way the chain is correct.

4. **No reproducible features.** If $Z_i = 0$ for all $i$, the
   Pólya-urn loop has nothing to do and we skip straight to step 3 of
   the sweep. The clusters effectively go "dormant" until $Z$ resamples
   some 1's.

## Cost

Each Pólya-urn step is $O(n_{\text{rep}} \cdot T_{\text{avg}})$ copula
density evaluations. On the simulation grid with $n = 4000$ and
$T_{\text{avg}} \approx 1.5$, this is the inner loop the chain spends
most of its time in — the NUTS atom updates are cheaper.

Vectorisation: the per-cluster densities are batched across features
inside each cluster, so the constant factor is small. The outer per-feature
loop is in Python, but each iteration only touches one row of $U$.

## What you should see in posterior diagnostics

- The histogram of $T^{(t)}$ across draws should concentrate. If it is
  flat between 1 and 8, the chain has not mixed.
- The traceplot of any cluster-level summary (size, atom parameter)
  should not drift. If a cluster appears and never gets revisited, it
  is a singleton stranded by a bad new-cluster proposal — usually fine,
  but lots of these inflate the apparent variability of $T$.
- The proportion of `is_new` events per sweep — accessible via the
  per-sweep summary returned by `sweep_multi` — should be a small but
  non-negligible fraction (say 1–5%). If 0%, the prior is too tight; if
  > 20%, the data are not supporting the partition you have and
  $G_0$ may be too diffuse.
