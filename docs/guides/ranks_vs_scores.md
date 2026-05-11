# Ranks vs raw scores

PY-IDR consumes a pilot CDF $F_0 \in [0, 1]^{n \times K}$, not raw scores
$X \in \mathbb R^{n \times K}$. This page is the practical answer to:
how do I build $F_0$, when does the choice matter, and what should I
do if my "scores" are not on a sensible scale?

## The two transforms PY-IDR ships

[`py_idr.marginals.transform`][] exposes two ways to build $F_0$:

### `empirical_rank_transform(x)` — default

$$
F_{0, j}(x_{ij}) = \frac{\mathrm{rank}(x_{ij})}{n + 1},
$$

with the `mid_rank` tie convention (a tie of $k$ features all receive
the average of their would-be ranks). This is the original IDR pilot
and is fine for any score whose only assumption is "larger is more
likely real."

```python
import jax.numpy as jnp
from py_idr.marginals.transform import empirical_rank_transform

F0_col = empirical_rank_transform(jnp.asarray(X[:, j]))
```

### `cdf_pilot(x, cdf_fn)` — explicit marginal

When you know the marginal CDF analytically (e.g. p-values which are
$\mathrm{Uniform}(0, 1)$ under the null, or scores you have calibrated
externally), you can pass it directly:

$$
F_{0, j}(x_{ij}) = F_j(x_{ij}),
$$

where $F_j$ is supplied as a callable.

## When the choice matters

The Bernstein-Dirichlet marginal model **corrects** the pilot CDF
toward the true marginal as the chain runs. So in principle the choice
of pilot is just an initial condition. In practice:

- **For shared marginals (S1, S2, real data with consistent
  experimental scale)**, the empirical rank pilot is already very close
  to the truth and the Bernstein-Dirichlet correction is tiny. Use the
  empirical rank.
- **For heterogeneous marginals (S3, S4)**, the empirical rank pilot is
  still close to the truth *per replicate* — that is the whole point of
  rank transforms. The Bernstein-Dirichlet correction matters more
  here, but the starting point is still good.
- **For "scores" that are not really comparable across features**
  (e.g., logit-transformed probabilities that have been thresholded),
  the empirical rank can be misleading because ties are massive. In
  that case, jitter the scores before passing them in, or use the
  `tie_policy` knob in [`py_idr.data.schema`][] to record what
  convention you used so the manifest reflects it.

## Tied scores

The default `mid_rank` policy assigns each tied feature the average of
the ranks the tie would have taken. Mathematically this is the
*right* thing — it does not bias the pilot in either direction.

Two alternative policies are tracked in the `DatasetManifest` schema:

- `random` — break ties by random tie-breaking. This destroys the
  *information* that the features tied, but smooths the pilot. Useful
  for legacy datasets where ties are an artefact.
- `truncate` — drop tied features. Use only when ties are clearly
  spurious (e.g., values capped at a software ceiling).

The manifest records whichever policy you used. Reproducibility
requires the same policy across replicate runs.

## Missing data

Missing scores are recorded via the `missingness_policy` field of the
`DatasetManifest`:

- `mean_rank` (default) — replace each missing score with the mean rank
  of the column. Equivalent to "this feature is not informative in
  this replicate." Conservative for the IDR question (no signal can come
  from a missing replicate).
- `drop_feature` — drop any feature with at least one missing replicate.
  Lossier but more conservative.

The chain does not see "missingness" directly — it operates on $F_0$
which has been filled in by whichever policy. The manifest records the
choice.

## What if my scores really are not rankable?

Sometimes "score" is a categorical label or a confidence tier (e.g.,
\{low, medium, high\}). Two options:

1. **Add jitter** to the raw scores within each tier so the empirical
   rank can break ties. This is a defensible move when the tier is the
   only meaningful information per feature.
2. **Re-frame the problem.** PY-IDR is designed for *continuous* scores
   whose ranking carries information. If your scores are genuinely
   ordinal with only a few levels, the multinomial-logistic IDR variants
   are a better fit; PY-IDR is overkill.

## Sanity checks before running the chain

Before you fit, plot the column histograms of $F_0$. They should each
look roughly uniform on $[0, 1]$ under the null component.

- If one column is heavily concentrated at the bottom (most $F_0$ values
  near 0), that replicate has many features below the "noise floor" —
  consider re-thresholding before ranking.
- If one column shows a strong bimodality, the empirical rank pilot
  has been forced into a piecewise-uniform shape by tie clusters; check
  your tie policy.
