# LKJ + PLOD-constrained correlations

For Gaussian-copula clusters with $K \ge 3$, PY-IDR needs a prior on the
$K \times K$ correlation matrix that is (a) tractable for NUTS, and
(b) restricted to the **positive-lower-orthant-dependence** (PLOD)
cone, because the reproducible component is by definition positively
dependent across replicates. This page is the short answer to: how do
we get both?

## The LKJ prior

The **LKJ distribution** (Lewandowski, Kurowicka, Joe, 2009) is the
canonical prior on correlation matrices. It places density

$$
p(R \mid \eta) \propto |R|^{\eta - 1}
$$

on the space of $K \times K$ positive-definite matrices with unit
diagonal. The shape parameter $\eta > 0$ controls how concentrated the
prior is on the identity:

- $\eta = 1$ — uniform over correlation matrices.
- $\eta > 1$ — concentrated near the identity (small off-diagonals).
- $\eta < 1$ — pushed toward boundary (large $|\rho|$).

LKJ is **invariant under permutation** of the coordinates and admits a
clean Cholesky parameterisation that NumPyro implements natively
(`numpyro.distributions.LKJCholesky`).

## The Cholesky parameterisation

Every correlation matrix factors as $R = L L^\top$ for a *lower-triangular*
$L$ with positive diagonal. The off-diagonals of $L$ are free
real numbers up to a unit-row-norm constraint. NUTS works much better in
this parameterisation than directly on $R$, because the constraint
"correlation matrix" is replaced by the much friendlier "rows of $L$ are
unit-norm in the Euclidean sense."

PY-IDR's Gaussian copula atom is *always* parameterised by its Cholesky
factor $L$; the dense $R$ is reconstructed on demand for density
evaluation. See [`py_idr.algebra.correlation`][].

## The PLOD constraint

**Positive lower-orthant dependence.** A $K$-vector $U \in [0, 1]^K$ has
PLOD if

$$
\PP(U_1 \le u_1, \dots, U_K \le u_K) \ge \prod_{j=1}^K u_j
\quad \forall u \in [0, 1]^K.
$$

For the Gaussian copula, PLOD is equivalent to $R$ having **non-negative
off-diagonals** — i.e. every pairwise correlation $\rho_{jk}$ is $\ge 0$.
This is the right constraint for the reproducible component: replicate
scores cannot *anti*-correlate under reproducibility.

The PLOD constraint is implemented as a **projection step** on the
candidate $R$ produced by NUTS: any negative off-diagonal entry is
clamped to zero and the matrix is re-projected onto the
positive-definite cone. See [`py_idr.algebra.plod`][].

The projection is differentiable (in the sense that NUTS can step
through it) because the clamp is applied to the constructed $R$ *after*
the Cholesky reparameterisation, not during. The NUTS step proposes a
free Cholesky $L$; we reconstruct $R = L L^\top$; we PLOD-project; we
evaluate the copula density on the projected $R$.

## Why not just constrain LKJ directly?

You could try to impose "non-negative off-diagonals" in the LKJ
parameterisation directly, but it is awkward — LKJ samples
unconstrained off-diagonals naturally and re-parameterising under the
PLOD constraint loses the LKJ's invariance. The post-hoc projection is
operationally cleaner and the bias is negligible when the data
genuinely have positive dependence (which they do, by assumption).

## Exchangeable special case ($K = 2$ or $T = 1$)

For the T = 1 fast path PY-IDR uses an **exchangeable correlation
matrix**

$$
R_\rho = (1 - \rho) I + \rho \, \mathbf 1 \mathbf 1^\top,
$$

parameterised by the single scalar $\rho \in (0, 1)$. The PLOD constraint
reduces to $\rho \ge 0$ which is the prior support anyway.

The chain samples $\rho$ via a NumPyro Beta prior with a small
concentration. See [`py_idr.inference.mcmc.nuts_atoms`][].

## Multi-cluster ($T > 1$) and atomic types

In the multi-cluster path, each Gaussian-copula cluster carries its own
LKJ-Cholesky $L_t$. NUTS updates them in batch via NumPyro
(`atom_dispatch.update_atom_via_nuts`), one cluster at a time. The PLOD
projection runs after every accepted NUTS step.

When the type-update step (`type_update.py`) proposes converting a
Gaussian cluster into a Clayton cluster (or vice versa), the LKJ-Cholesky
representation is discarded and the Clayton-$\theta$ representation
replaces it. The reverse move re-initialises $L$ from a near-identity
prior draw. This is fine for the chain because the type-update is an
**independence-Metropolis** step — the proposal is the family prior, not
a continuous reparameterisation.

## A note on identifiability

When $K = 2$, the LKJ prior is just $\mathrm{Beta}$ on $(\rho + 1)/2$.
The PLOD constraint reduces to a half-line. Nothing exotic.

When $K \ge 3$, the LKJ + PLOD combination *is* informative: the
posterior on $R$ is shaped both by the data (through the copula
likelihood) and by the PLOD cone. The cone restriction prevents the
chain from finding "false" reproducible clusters where some pairs are
positively correlated and others are not — those would model
*mode-locked* clusters, which are not interesting for the IDR setting.

## Where to look in the code

- `py_idr.algebra.correlation.cholesky_to_correlation` — $L \mapsto R$.
- `py_idr.algebra.correlation.correlation_to_cholesky` — $R \mapsto L$ via Cholesky.
- `py_idr.algebra.plod.plod_project` — the projection onto the PLOD cone.
- `py_idr.inference.mcmc.nuts_atoms` — the NUTS update for Gaussian-copula atoms.
- `py_idr.copulas.gaussian.gaussian_copula_log_pdf` — the copula density that NUTS evaluates.
