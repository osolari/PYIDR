# Bernstein-Dirichlet marginals

The Bernstein-Dirichlet marginal model is how PY-IDR learns the
*marginal* CDF on each replicate $j$ alongside the copula. This page
states the model, points to the implementation, and explains the
conjugate update that makes the corresponding sweep step nearly free.

## Why a marginal model at all

After applying the pilot CDF $F_0$ to each column of $X$ we have data
$U = F_0(X) \in [0, 1]^{n \times K}$ that *should* be approximately
uniform under the irreproducible component (by the probability integral
transform — see [Sklar](sklar_copulas.md)). The empirical-rank pilot is
exactly uniform on the irreproducible *sample*, but only approximately
uniform on the *population*; the small mismatch is enough to bias the
copula posterior. The Bernstein-Dirichlet correction lets the chain
**adjust each marginal $F_j$** to whatever shape is consistent with the
data plus the current copula state.

## The model

Let $b_{m, M}(u) = \binom{M}{m} u^m (1 - u)^{M - m}$ be the $m$-th
**Bernstein polynomial** of degree $M$. For each replicate $j$ we model
the marginal CDF $F_j$ as a convex combination

$$
F_j(u) = \sum_{m=0}^{M} w_{j, m} \, B_{m, M}(u),
$$

where $B_{m, M}(u) = \sum_{k \le m} b_{k, M}(u)$ is the integrated
Bernstein basis (the CDF associated with the $m$-th basis pmf) and
$w_j = (w_{j, 0}, \dots, w_{j, M})$ is a vector on the simplex.

Equivalently, the *density* is

$$
f_j(u) = \sum_{m=0}^{M} w_{j, m} \, b_{m, M}(u),
$$

a Bernstein density of degree $M$ with weights $w_j$. Any continuous
density on $[0, 1]$ can be uniformly approximated by such a mixture as
$M \to \infty$.

Prior on the weights: $w_j \sim \mathrm{Dirichlet}(\beta_0)$ with
$\beta_0 = (\beta_0, \dots, \beta_0)$ a symmetric concentration vector.
The default $\beta_0$ is weakly informative.

Code: [`py_idr.marginals.bernstein`][] (basis evaluation),
[`py_idr.marginals.dirichlet_weights`][] (the conjugate update),
[`py_idr.marginals.pilot`][] (the glue between the pilot CDF and the
weighted mixture).

## Why this is the right basis

Bernstein polynomials have three properties that make them the natural
nonparametric marginal:

1. **Shape preserving.** A convex combination of CDFs is a CDF; we never
   leave the space of valid marginals during the update.
2. **Local support.** The $m$-th basis is concentrated near $m/M$, so a
   weight update at index $m$ only affects the marginal near that point.
   This is what makes the chain mix — the marginal can shift mass in the
   upper tail without disturbing the lower tail.
3. **Conjugate to a Dirichlet prior under multinomial sampling.** The
   weights enter the likelihood through a discrete sufficient statistic
   (the per-feature posterior probability that it falls in basis $m$);
   the Dirichlet prior makes the weights' full conditional another
   Dirichlet. **This is the entire sweep step.**

## The conjugate update

At sweep iteration $s$, for each replicate $j$:

1. Evaluate the **basis responsibilities**

   $$
   \gamma_{i, m}^{(j)} = \frac{w_{j, m}^{(s)} \, b_{m, M}(U_{ij})}
                              {\sum_{m'} w_{j, m'}^{(s)} \, b_{m', M}(U_{ij})}.
   $$

2. Sum to **basis counts**:
   $N_{j, m} = \sum_{i=1}^{n} \gamma_{i, m}^{(j)}$.

3. Sample $w_j^{(s+1)} \sim \mathrm{Dirichlet}(\beta_0 + N_j)$.

Each step is $O(n M)$ and embarrassingly parallel across $j$. The
implementation in `dirichlet_weights.update_weights` vectorises both
loops.

## Updating $M$

The degree $M$ is itself a random integer with a truncated geometric
prior, see [`py_idr.marginals.degree_prior`][]. The update is a
Metropolis step that proposes $M \pm 1$ with appropriate Jacobian; small
moves keep the acceptance rate near 0.4–0.6 in practice. Because the
sweep step for $w_j \mid M$ is conjugate, the $M$-update only needs the
marginal likelihood of $w_j$ integrated out — a closed-form Dirichlet
multinomial.

## Practical guidance

- **What does $M$ control?** Roughly, the *flexibility* of each
  marginal. Small $M$ (3–5) is fine for shared standard-Gaussian
  marginals (S1, S2). Large $M$ (8–15) lets the chain learn heavy or
  skewed marginals (S3, S4).
- **What if I set $M$ too small?** The marginal will be smoother than
  truth, and the copula posterior will overcompensate by inflating the
  correlation. This shows up as a *biased* posterior mean for $\rho$
  in S3/S4 if $M = 3$.
- **What if I set $M$ too large?** The marginal will overfit (especially
  on the tails); local idr estimates for high-rank features will be
  noisy. The degree prior in `degree_prior.py` pushes back, but a
  too-diffuse prior here can hurt.

## Initialisation from the pilot

We initialise $w_j$ at the projection of the empirical CDF onto the
Bernstein basis — `pilot.initial_weights`. Mathematically this is

$$
w_{j, m}^{(0)} = \int_0^1 b_{m, M}(u) \, \hat F_n(u) \, du,
$$

which integrates the basis against the empirical CDF $\hat F_n$. The
projection is closed form; no inversion needed. The chain then refines
$w_j$ away from this starting point.
