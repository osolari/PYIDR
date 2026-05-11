# The Sun-Cai step-up rule

The decision step PY-IDR uses is **Sun and Cai (2007, *JASA*)**'s
step-up rule applied to the posterior local idr. This page gives the
formula, the rationale, and the precise way the code handles the edge
cases (ties, empty rejection sets, the $\max(1, \cdot)$ denominator).

## Setup

After the chain, every feature $i$ has a posterior local idr

$$
\mathrm{idr}_i = \PP(Z_i = 0 \mid \text{data}) \in [0, 1].
$$

Small $\mathrm{idr}_i$ = high posterior probability of being reproducible.
We want to **call** a set $\mathcal R \subseteq \{1, \dots, n\}$ of
reproducible features while controlling the **marginal FDR** at a
user-specified level $\alpha$ (typically 0.05).

## The rule

Sort the local idr values ascending so that
$\mathrm{idr}_{(1)} \le \mathrm{idr}_{(2)} \le \dots \le \mathrm{idr}_{(n)}$.
For each $k \in \{1, \dots, n\}$, let

$$
Q_k = \frac{1}{\max(1, k)} \sum_{i=1}^{k} \mathrm{idr}_{(i)}.
$$

$Q_k$ is the average posterior probability that the top-$k$ features are
*irreproducible*; intuitively, it is the Bayesian FDR of the top-$k$ set.
Sun-Cai select

$$
k_\alpha = \max\big\{k : Q_k \le \alpha\big\},
$$

with the convention $k_\alpha = 0$ if no $k$ satisfies the inequality.

The call set is

$$
\mathcal R = \big\{i : \mathrm{idr}_i \le \mathrm{idr}_{(k_\alpha)}\big\}.
$$

Code: [`py_idr.decision.sun_cai`][], specifically `sun_cai_step_up(idr, alpha)`.

## Why this controls FDR

Under the assumed Bayesian model, the posterior expected number of false
discoveries in the top-$k$ set is exactly $\sum_{i \le k} \mathrm{idr}_{(i)}$
(Sun and Cai, eq. 3.4). The posterior mFDR is therefore
$\sum_{i \le k} \mathrm{idr}_{(i)} / k$, which is $Q_k$. The step-up rule
picks the largest $k$ such that this is $\le \alpha$.

The result is *not* a frequentist FDR guarantee — it is a Bayesian one,
conditional on the model. In practice the empirical realised FDR
matches the nominal $\alpha$ closely as long as the posterior local idr
is well-calibrated, which is what the simulation grid checks (Table 4 in
the report).

## The $\max(1, \cdot)$ in the denominator

A subtle but important point. The "naive" rule

$$
Q_k^{\text{naive}} = \frac{1}{k} \sum_{i \le k} \mathrm{idr}_{(i)}
$$

is *undefined* at $k = 0$ (no features rejected). Using $\max(1, k)$ in
the denominator makes the rule well-defined for every $k$ from 0 to
$n$ and equivalent to the original Sun-Cai formulation. The code uses
this form; the test `test_evaluate_recovery.py` exercises the
$k_\alpha = 0$ corner.

## Ties

When two features have identical local idr (common after rounding from
finite Monte Carlo), the rule is **inclusive on ties**: if
$\mathrm{idr}_{(k_\alpha)} = \mathrm{idr}_{(k_\alpha + 1)}$, both go into
the call set. This is slightly conservative on the FDR side but the
right default — excluding either one of two indistinguishable features
is arbitrary.

## How it compares to the global IDR threshold

The original IDR rule (Li et al., 2011) returns the **global IDR** at
each rank:

$$
\overline{\mathrm{idr}}_k = \frac{1}{k} \sum_{i \le k} \mathrm{idr}_{(i)}
$$

and the user reads off the rank at which $\overline{\mathrm{idr}}_k$
crosses 0.05 (or 0.01). This is **the same statistic** as $Q_k$ —
Sun-Cai is just the principled extraction of a decision rule from it.
The advantage is that Sun-Cai is the exact step-up rule whereas the
"read off the rank" approach is informal and depends on the user
choosing how to handle ties / boundary cases.

Once $k_\alpha$ is selected, the **threshold** value
$\tau_\alpha = \mathrm{idr}_{(k_\alpha)}$ is exposed by
`sun_cai_step_up`. It is useful for plotting and for applying the same
threshold to a held-out validation set.

## Evaluating the rule on synthetic data

For S1–S4 we know the ground truth $Z_i$, so we can compute the
**realised** FDR and power. The helper is
[`py_idr.simulation.evaluation.evaluate_recovery`][]. It returns:

- `realized_fdr` = $\frac{|\{i \in \mathcal R : Z_i = 0\}|}{\max(1, |\mathcal R|)}$
- `power` = $\frac{|\{i \in \mathcal R : Z_i = 1\}|}{\max(1, \sum_i Z_i)}$
- `k_alpha` = the chosen rank

Across the simulation grid, the empirical $\mathbb E[\mathrm{realized\_fdr}]$
should sit at or slightly below $\alpha$. Power depends on the regime —
S2 (weak signal) has much lower power than S1 because the local idr
distribution is less separated.

## What the rule is *not* doing

A few common confusions worth flagging:

- **It is not a frequentist test.** There is no null distribution being
  thresholded. The whole machinery is conditional on the Bayesian model
  fit; if the model is misspecified the FDR guarantee is also.
- **It is not the running-IDR.** It is the step-up rule *derived from*
  the running IDR statistic. The distinction matters for the boundary
  behaviour at $k$ very small.
- **It does not adjust for multiple chains.** The local idr that goes
  into Sun-Cai is the Monte Carlo average across all draws — chains and
  samples flattened. There is no separate "chain-level" FDR.

## Alternative: Bayesian mFDR with expected discoveries

PY-IDR also exposes a slightly different decision interface in
[`py_idr.decision.bayes_mfdr`][]: given a threshold $\tau \in [0, 1]$ on
the local idr, return the posterior expected number of discoveries
$\sum_i \PP(\mathrm{idr}_i \le \tau)$ and the corresponding posterior
mFDR. This is useful when the user has a *specific* threshold in mind
(e.g. matched to a downstream pipeline) and wants to report the
posterior mFDR at that threshold, rather than picking $\tau$ by
controlling FDR at $\alpha$.
