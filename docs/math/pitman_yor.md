# The Pitman-Yor process

The Pitman-Yor (PY) process is the prior PY-IDR uses on the number,
weights, and parameters of the **copula clusters** that make up the
reproducible component. This primer collects the three facts about PY
that the model relies on:

1. its **stick-breaking** representation,
2. its **exchangeable partition probability function** (EPPF), and
3. its **predictive (Pólya-urn) form**.

We assume you have seen the Dirichlet process; PY generalises it.

## Stick-breaking

A draw $G \sim \mathrm{PY}(\alpha, \sigma, G_0)$ is a discrete
distribution that can be written

$$
G(\cdot) = \sum_{t=1}^\infty w_t \, \delta_{\theta_t}(\cdot),
$$

where:

- $\theta_t \stackrel{\text{iid}}{\sim} G_0$ are the atom locations,
  drawn from the **base measure** $G_0$. In PY-IDR, $G_0$ is the prior
  over copula-family-and-parameter pairs: the family is uniform over the
  registry, and the parameters are drawn from the family-specific prior.
- $w_t$ are the **stick-breaking weights**: $V_t \sim \mathrm{Beta}(1 - \sigma, \alpha + t \sigma)$
  independently, and
  $w_t = V_t \prod_{s<t}(1 - V_s)$.

Two hyperparameters control the prior:

- $\sigma \in [0, 1)$ — the **discount**. $\sigma = 0$ recovers the
  Dirichlet process; positive $\sigma$ gives the prior a **power-law
  tail**, so the number of clusters in $n$ observations scales like
  $O(n^\sigma)$ instead of $O(\log n)$. For PY-IDR this matters when $n$
  is large (10⁵+ peaks) and a few rare-but-real copula clusters need
  posterior mass.
- $\alpha > -\sigma$ — the **concentration**. Holds the same role as in
  the Dirichlet process: larger $\alpha$ means more clusters at any fixed
  $n$. Default prior: a weakly informative $\mathrm{Exponential}(1)$.

Code: [`py_idr.pym.stickbreak`][].

## The EPPF

When you marginalise out the cluster identities and ask for the
distribution over the *partition* of $\{1, \dots, n\}$ that PY induces,
you get the EPPF:

$$
p(n_1, \dots, n_T \mid \alpha, \sigma) =
\frac{1}{(\alpha + 1)_{n - 1}}
\prod_{t=1}^{T-1}(\alpha + t \sigma)
\prod_{t=1}^{T}(1 - \sigma)_{n_t - 1},
$$

where $T$ is the number of nonempty blocks, $n_t = |\{i : Z_i = t\}|$ are
the block sizes, and $(a)_n$ is the Pochhammer rising factorial
$(a)_n = a(a + 1)\cdots(a + n - 1)$.

The EPPF is symmetric in the block sizes and depends only on the
*counts*, not on which features are in which block. We use it for the
$(\alpha, \sigma)$ update: holding the partition fixed, the EPPF as a
function of $(\alpha, \sigma)$ is a clean posterior factor that NumPyro's
NUTS can sample efficiently.

Code: [`py_idr.pym.eppf`][]. The log-EPPF is `log_eppf(alpha, sigma, cluster_sizes, n)`.
The NUTS update is [`py_idr.inference.mcmc.py_hyperparam_update`][].

## The Pólya urn (predictive)

Conditional on the partition of $\{1, \dots, i - 1\}$, the predictive
distribution of $Z_i$ is

$$
\PP(Z_i = t \mid Z_{1:i-1}) =
\begin{cases}
\dfrac{n_t - \sigma}{\alpha + i - 1}, & t \text{ is an existing cluster}, \\[6pt]
\dfrac{\alpha + T \sigma}{\alpha + i - 1}, & t \text{ is a new cluster}.
\end{cases}
$$

Two observations:

- Compared to the Dirichlet process ($\sigma = 0$), the PY predictive
  **discounts** existing-cluster sizes by $\sigma$ and adds $T \sigma$ to
  the new-cluster mass. The discount is what makes the cluster-count
  growth power-law.
- This is the exact formula used by the **Pólya-urn cluster
  reassignment** step of Algorithm 1. For each feature with $Z_i = 1$,
  we remove it from its current cluster, evaluate the predictive over
  every existing cluster and one fresh-cluster slot, and resample
  $Z_i$ from that categorical. The reassignment is exact under exchangeability.

Code: [`py_idr.pym.polya_urn`][]. The cluster reassignment that uses it
is at the top of [`py_idr.inference.mcmc.sweep_multi`][].

## How PY fits into PY-IDR

Putting the pieces together:

1. The reproducible component is a mixture of copula clusters
   $C_1, \dots, C_T$ with mixing weights $w_1, \dots, w_T$ drawn from PY.
2. Each feature $i$ with $Z_i = 1$ is assigned to one of these clusters
   via the latent label $L_i \in \{1, \dots, T\}$.
3. Conditional on $L_i = t$, the copula-scale data $U_i$ is drawn from
   $C_t$.
4. The PY hyperparameters $(\alpha, \sigma)$ are themselves resampled
   each sweep from their posterior, which is just the prior times the EPPF.

The number of clusters $T$ is *random* and grows like $O(n^\sigma)$ a
priori. In practice the posterior concentrates strongly: when the data
really are from a single Gaussian copula (S1), the chain spends nearly
all its time in $T = 1$ states; when there is genuine multi-cluster
structure (S5), it explores $T = 2$ or $T = 3$ states.

## What you actually need to read from the posterior

- `posterior["alpha"]`, `posterior["sigma"]`: the PY hyperparameters,
  scalars. Useful for understanding how prior-vs-posterior the cluster
  count is.
- `multi_state.num_active_clusters()` (per-draw): the random
  $T^{(t)}$. The histogram of this across draws is the posterior on $T$.
- The cluster atoms themselves (per-draw, variable in number): which
  copula families and parameters are active. Summarising these is
  application-specific and lives downstream of the chain.
