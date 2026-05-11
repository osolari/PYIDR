# The IDR problem

This primer re-derives the *irreproducible discovery rate* (IDR) of
Li, Brown, Huang, and Bickel (2011, *Ann. Appl. Stat.*) in a form that
exposes exactly which assumptions PY-IDR relaxes. Reading it once is
enough to know which line of the model each subsequent primer is talking
about.

## Setup

We have $n$ features and $K$ independent replicate experiments. Each
experiment returns a score for every feature; stack them into

$$
X \in \mathbb R^{n \times K}, \quad X_{ij} = \text{score of feature } i
\text{ in replicate } j.
$$

The order of features inside each column carries information (high score
= "more likely real signal"), but the *raw* scores across replicates are
not directly comparable — replicate 1 might have a heavier upper tail than
replicate 2, simply because of an experimental scaling difference.

We want a per-feature decision: *is feature $i$'s high score reproducible
across replicates, or did it appear in the top of one column by chance?*

## The two-component mixture

The original IDR model is a **two-component mixture**, indexed by a
latent indicator $Z_i \in \{0, 1\}$:

$$
\begin{aligned}
Z_i &\stackrel{\text{iid}}{\sim} \mathrm{Bernoulli}(\pi^*) \\
X_i \mid Z_i = 1 &\sim \text{(reproducible distribution)} \\
X_i \mid Z_i = 0 &\sim \text{(irreproducible distribution)}.
\end{aligned}
$$

$\pi^*$ is the **reproducible mass** — the fraction of features that are
truly shared signal. The reproducible distribution has *positive*
dependence across the $K$ replicate columns; the irreproducible
distribution is independent across columns (the per-replicate marginals
might still match, but the across-replicate correlation is zero).

## Sklar's decomposition

The key modelling move is to use **Sklar's theorem** (covered properly in
the next primer): any joint distribution factors into

$$
F(x_1, \dots, x_K) = C\!\left(F_1(x_1), \dots, F_K(x_K)\right),
$$

where $F_j$ are the marginals and $C$ is the **copula** capturing pure
dependence on the unit hypercube $[0, 1]^K$. The IDR mixture becomes

$$
F(x) = \pi^* \, C_{\text{rep}}\!\left(F_1(x_1), \dots, F_K(x_K)\right) +
       (1 - \pi^*) \, C_{\text{irrep}}\!\left(F_1(x_1), \dots, F_K(x_K)\right).
$$

Critically the *same* marginals $F_j$ appear in both components — the only
thing that differs between reproducible and irreproducible features is
the dependence structure.

## The original IDR choices

Li et al. fix:

- $C_{\text{irrep}}$ = the **independence copula** $C(u) = \prod_j u_j$.
- $C_{\text{rep}}$ = the $K$-variate **Gaussian copula** with exchangeable
  correlation $\rho$.
- Marginals $F_j$ = the empirical CDF on the observed ranks (pilot rank
  transform).

The model is then parametrised by $(\pi^*, \rho)$ plus the empirical
marginals. The fit is by EM on $(\pi^*, \rho)$ with the per-feature
posteriors $\PP(Z_i = 0 \mid X_i)$ computed as a by-product. Those
posteriors are the **local idr**:

$$
\mathrm{idr}_i = \PP(Z_i = 0 \mid X_i).
$$

The decision rule sorts features by their local idr ascending (most
likely reproducible first) and reports the global IDR — the running mean
$\overline{\mathrm{idr}}_i = i^{-1} \sum_{k \le i} \mathrm{idr}_{(k)}$ — as
a Bayesian FDR proxy.

## Where PY-IDR enters

PY-IDR keeps the mixture structure and the Sklar decomposition but
relaxes three pinch-points:

1. **$C_{\text{rep}}$ is not forced to be a single Gaussian copula.** A
   Pitman-Yor prior over copula clusters lets multiple copula
   families (Gaussian, Clayton, Gumbel, $t_\nu$) share posterior mass.
   The number of clusters $T$ is not fixed.
2. **Marginals $F_j$ are not the pilot rank.** Bernstein-Dirichlet
   polynomials learn each $F_j$ alongside the copula. The pilot rank is
   still the initial condition, but the chain corrects it.
3. **The decision rule is not the running-mean IDR proxy.** Sun-Cai's
   step-up rule, applied to the posterior local idr, gives exact mFDR
   control at any nominal level $\alpha$.

The remaining primers spell out each of these in turn.

## Conventions used throughout

| Symbol     | Meaning                                                | Code surface                                     |
|------------|--------------------------------------------------------|--------------------------------------------------|
| $n$        | Number of features                                     | `sim.n`, `F0.shape[0]`                           |
| $K$        | Number of replicates                                   | `sim.K`, `F0.shape[1]`                           |
| $Z_i$      | Reproducibility indicator                              | `result.z_samples[..., i]`                       |
| $\pi^*$    | Reproducible mass                                      | `posterior["pi"]`                                |
| $\rho$     | Gaussian-copula correlation (T = 1)                    | `posterior["rho"]`                               |
| $T$        | Number of active copula clusters                       | `multi_state.num_active_clusters()`              |
| $\alpha, \sigma$ | PY concentration & discount                      | `posterior["alpha"]`, `posterior["sigma"]`       |
| $M$        | Bernstein polynomial degree                            | `posterior["M"]`                                 |
| $F_j$      | Replicate-$j$ marginal CDF                             | `marginals.pilot.evaluate_F0`                    |
| $U_{ij}$   | $F_j(X_{ij})$, the copula-scale rank                   | `marginals.transform.empirical_rank_transform`   |
| $\mathrm{idr}_i$ | Local idr                                        | `decision.local_idr.from_mcmc`                   |
| $\alpha$ (decision) | Sun-Cai nominal FDR level                   | `decision.sun_cai.sun_cai_step_up(..., alpha=)`  |

The PY hyperparameters and the decision-rule level both use the letter
$\alpha$ in the literature — context disambiguates. We try to write
"$\alpha_{\text{PY}}$" when the distinction matters.
