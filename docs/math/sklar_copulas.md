# Sklar's theorem & copulas

This primer is a one-page refresher on **copulas** and the trick that
lets PY-IDR decouple "what each replicate looks like marginally" from
"how the replicates depend on each other." If you have seen copulas
before, skim it; if not, this is the load-bearing piece of background for
the rest of the model.

## Sklar's theorem

For any joint distribution $F$ on $\mathbb R^K$ with marginals
$F_1, \dots, F_K$, there exists a **copula** $C : [0, 1]^K \to [0, 1]$
such that

$$
F(x_1, \dots, x_K) = C\!\left(F_1(x_1), \dots, F_K(x_K)\right).
$$

If the marginals are continuous (which is the only case PY-IDR deals
with), the copula $C$ is **unique**. It is itself a CDF — specifically the
CDF of the vector $(F_1(X_1), \dots, F_K(X_K))$, which is uniform in each
coordinate by the probability integral transform.

This is one of the most useful theorems in applied probability because it
says you can **separate concerns**:

- choose any marginal model you like, independently per coordinate;
- choose any copula you like, independently of the marginals;
- glue them together via Sklar.

## Why it is the right object for IDR

In the reproducibility setting we have $K$ replicate experiments. The
*marginals* — how feature scores are distributed in each replicate — vary
across replicates for boring reasons: experiment-specific noise, scaling,
binning. The *across-replicate dependence* — does a high score in
replicate 1 predict a high score in replicate 2? — is the *only*
quantity we actually care about for reproducibility.

Sklar lets us absorb everything boring into the marginals $F_j$ and study
the copula in isolation. Two replicates are "reproducible" iff their
copula has substantial positive dependence; "irreproducible" iff the
copula is the independence copula

$$
C^{\bot}(u) = \prod_{j=1}^K u_j.
$$

## Three copula families used in PY-IDR

PY-IDR has four copula families in the registry. Three are summarised
below; the $t_\nu$ copula is essentially the Gaussian copula with a
heavier-tail bridge and is covered in the report.

### Gaussian copula

Parametrised by a positive-definite correlation matrix $R$ with unit
diagonal. The copula is

$$
C_R(u) = \Phi_R\!\left(\Phi^{-1}(u_1), \dots, \Phi^{-1}(u_K)\right),
$$

where $\Phi_R$ is the joint CDF of $\mathcal N_K(0, R)$. Tail dependence
is **zero** in every direction — extreme co-movements are exactly as
likely under the Gaussian copula as under independence asymptotically.

In PY-IDR the Gaussian copula uses an **exchangeable** correlation
structure $R = (1 - \rho) I + \rho \, \mathbf 1 \mathbf 1^\top$ for the
single-cluster path, and a full LKJ-prior-constrained matrix in the
multi-cluster path. See [LKJ + PLOD](lkj_structured.md).

Code: [`py_idr.copulas.gaussian`][].

### Clayton copula

A one-parameter Archimedean copula with **lower-tail** dependence:
extreme *low* co-movements are more likely than the Gaussian copula
predicts. For $\theta > 0$,

$$
C_\theta^{\text{Cl}}(u) = \Big(\sum_{j=1}^K u_j^{-\theta} - K + 1\Big)^{-1/\theta}.
$$

The lower-tail dependence coefficient is $\lambda_L = 2^{-1/\theta}$.

Code: [`py_idr.copulas.clayton`][].

### Gumbel copula

The reflection of Clayton — **upper-tail** dependence. For $\theta \ge 1$,

$$
C_\theta^{\text{Gu}}(u) = \exp\!\left(-\Big(\sum_{j=1}^K (-\log u_j)^\theta\Big)^{1/\theta}\right).
$$

The upper-tail dependence coefficient is $\lambda_U = 2 - 2^{1/\theta}$.

PY-IDR's Gumbel implementation is **$K = 2$ only** today (the general
Hofert recursion for $K > 2$ is in plan 11). The type-update step in
[`py_idr.inference.mcmc.type_update`][] excludes Gumbel from the
proposal pool when $K \ge 3$.

Code: [`py_idr.copulas.gumbel`][].

## Mixing copulas

A finite mixture of copulas is itself a copula:

$$
C(u) = \sum_{t=1}^T w_t \, C_t(u), \quad w_t \ge 0, \sum w_t = 1.
$$

PY-IDR's reproducible component is exactly such a mixture, but with
$T$ unknown and a Pitman-Yor prior on the weights $w_t$. See the
[Pitman-Yor primer](pitman_yor.md) for how the mixture is parametrised.

## Generating from a copula

Two patterns appear repeatedly in the code:

**Pattern 1: draw on the latent scale, then apply $\Phi$.**

For the Gaussian copula, draw $Z \sim \mathcal N_K(0, R)$ and set
$U = \Phi(Z)$. This is what `_gaussian_copula_mixture_X` does in
[`py_idr.simulation.scenarios`][] — see the S1 generator.

**Pattern 2: marginal-by-marginal inverse CDF.**

Given $U$ on the copula scale, apply $F_j^{-1}$ to column $j$ to get the
observed-scale data. PY-IDR uses this every time it converts copula
draws to "what the user actually sees." The simulation regimes vary which
$F_j^{-1}$ they apply:

- S1, S2: $F_j^{-1} = \Phi^{-1}$ (standard normal).
- S3: $F_j^{-1}(u) = \sigma_j \Phi^{-1}(u)$ (per-replicate scale).
- S4: $F_j^{-1}$ = skew-normal inverse CDF with per-replicate skewness.

## What the chain sees

The chain never sees the raw $X$. It consumes the **pilot CDF**
$F_0(x_i, j) \in [0, 1]^K$ — for synthetic data, this is the empirical
rank transform; for real data, the same. The Bernstein-Dirichlet
marginals on top of $F_0$ correct any residual mismatch between
"empirical rank" and "true marginal CDF" — see
[Bernstein-Dirichlet](bernstein_dirichlet.md).

In code:

```python
from py_idr.marginals.transform import empirical_rank_transform
F0_col = empirical_rank_transform(X[:, j])  # rank/(n+1) with ties tied
```
