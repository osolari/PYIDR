# Choosing $K$ (replicate count)

The replicate count $K$ is the most consequential design choice in the
IDR setting. PY-IDR works for any $K \ge 2$, but the **statistical
behaviour** changes substantially as $K$ grows. This guide collects the
rules of thumb that the simulation grid confirms.

## The mechanical lower bound: $K = 2$

You need at least $K = 2$ to talk about reproducibility at all. With one
replicate, $\PP(Z_i = 0 \mid X_i)$ collapses to the prior $1 - \pi^*$
for every feature, and the model has no signal.

At $K = 2$:

- The Gaussian copula correlation $\rho$ is a single scalar.
- The Gumbel copula is supported (the type-update step admits it).
- The chain is fastest because the copula density evaluation is $K = 2$
  small.
- The local idr distribution often has substantial overlap between
  reproducible and irreproducible — power is modest unless $\rho$ is
  large.

This is what the original IDR (Li et al., 2011) was designed for and
remains the most common setting in practice.

## $K = 3$: the sweet spot

At $K = 3$ the model gains:

- **Off-diagonal LKJ structure.** The exchangeable parameterisation
  ($R = (1-\rho)I + \rho \mathbf{1}\mathbf{1}^\top$) is still a single
  scalar, but the full LKJ-Cholesky parameterisation has 3 free
  off-diagonals. The chain can detect that one replicate disagrees
  with the other two.
- **Stronger separation.** With three replicates, a feature has to be
  high-ranked in *all three* to look reproducible. The local idr
  distribution becomes much more bimodal; power at fixed $\alpha$ goes
  up by a wide margin.
- **Gumbel falls off.** The current implementation only supports
  $K = 2$ Gumbel. The type-update step's `supported_types_for_K`
  excludes Gumbel when $K \ge 3$. This is a temporary limitation.

For most experimental designs, $K = 3$ is the right default.

## $K = 4$ and above

The marginal value of a fourth replicate is smaller than the third —
diminishing returns kick in. But:

- **Multi-cluster detection.** If the reproducible signal is genuinely
  mixed (e.g. two pathways with different dependence patterns), $K = 4$
  is the smallest $K$ at which the multi-cluster sampler can reliably
  distinguish them. With $K = 3$ the model can usually fit any
  multi-cluster data with one Gaussian copula.
- **Cost.** The Gaussian copula density costs $O(K^3)$ per evaluation
  (Cholesky-based). For $K = 4$ this is still fast; for $K \ge 8$ the
  NUTS step becomes the bottleneck.

The simulation grid in §4.1 runs $K \in \{2, 3, 4\}$ for S1-S4 and
$K \in \{3, 4\}$ for S5 (multi-cluster).

## What if my $K$ is large (say $\ge 10$)?

This is unusual for the IDR setting but happens in some genomics
applications (e.g. 10 ChIP-seq replicates from one cell line). Two
considerations:

1. **The LKJ + PLOD posterior may concentrate too tightly.** At
   $K = 10$, even small data sets identify $R$ very precisely. The PLOD
   projection becomes nearly inactive (all entries already positive).
   Consider increasing the LKJ $\eta$ to keep the prior regularising.
2. **The Bernstein-Dirichlet update is per-replicate.** Each of the 10
   marginals gets its own $w_j$. Memory and compute scale linearly in
   $K$. Practical, but slow.
3. **The type-update is `K`-dependent.** Gumbel is excluded; the
   Clayton copula needs the multivariate generalisation. This part of
   the codebase is well-tested only up to $K = 5$.

If you have $K \ge 10$, consider running the chain in **CPU mode with
many chains in parallel** rather than a single GPU chain; the per-chain
inner loops are slow but embarrassingly parallel across chains.

## $K = 1$? Use IRLS instead

If you only have one replicate, IDR/PY-IDR is the wrong tool. You want a
standard FDR procedure on the one column of scores — Storey-q-value,
Benjamini-Hochberg. The reproducibility framing requires at least two
columns to compare.

## Posterior diagnostics that depend on $K$

A few sanity checks are $K$-specific:

- **$K = 2$.** The posterior on $\rho$ is the most prior-sensitive of
  all $K$. Run a prior-only chain (set $\pi^* \to 0$) and confirm the
  posterior on $\rho$ moves substantially.
- **$K \ge 3$.** Compare the per-pair empirical correlations in the
  reproducible part of the data against the posterior LKJ. If one pair
  is much weaker than the others, the multi-cluster sampler may put
  posterior mass on $T = 2$ (one cluster for the strong pairs, one for
  the weak).
- **$K \ge 4$.** The cluster-count posterior should be informative —
  $T$ rarely above 3 — and the per-cluster Gaussian-vs-Clayton-vs-$t_5$
  type proportions should be stable. If they drift, the type-update
  step is mixing poorly; consider longer warmup.
