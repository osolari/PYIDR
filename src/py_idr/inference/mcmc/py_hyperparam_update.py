"""Algorithm 1 step 7: joint Pitman--Yor $(\\alpha, \\sigma)$ update.

Per §3.3.1 of the revised report, the PY hyperparameters are updated via
"auxiliary-variable or slice updates". We follow the durable user feedback
to use mature library tooling and run **NumPyro NUTS** on the joint
posterior of $(\\alpha, \\sigma)$ given the current cluster configuration —
no hand-rolled slice sampler.

The model:

- $\\alpha \\sim \\mathrm{Exponential}(1)$ matching the report's $\\mathrm{Gamma}(1, 1)$
  hyperprior of §3.2.5.
- $\\sigma \\sim \\mathrm{Uniform}[0, 1)$ matching §3.2.5.
- A ``numpyro.factor`` adds the log-EPPF
  :func:`py_idr.pym.eppf.log_eppf` so the posterior is
  $\\propto p(\\alpha) p(\\sigma) p(\\mathrm{partition} \\mid \\alpha, \\sigma)$.

NumPyro handles the constrained-support transforms automatically:
$\\alpha$ via $\\log$, $\\sigma$ via $\\mathrm{logit}$.

The Pólya-urn step (already implemented; commit 7b) is conditioned on
$(\\alpha, \\sigma)$; conversely, the EPPF is a *sufficient statistic* of the
partition for the $(\\alpha, \\sigma)$ conditional. The joint sweep therefore
runs Algorithm 1 step 7 *after* the cluster reassignments (step 2) so the
partition is current.

See plans/04_inference_mcmc.md.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from jaxtyping import Array, Integer
from numpyro.infer import MCMC, NUTS

from py_idr.pym.eppf import log_eppf


def _py_hyperparam_model(
    cluster_sizes: Integer[Array, "T"],
    n: int,
) -> None:
    """NumPyro model declaring $\\alpha$, $\\sigma$ and the log-EPPF factor."""
    alpha = numpyro.sample("alpha", dist.Exponential(1.0))
    sigma = numpyro.sample("sigma", dist.Uniform(0.0, 1.0))
    numpyro.factor("eppf", log_eppf(alpha, sigma, cluster_sizes, n))


def update_py_hyperparams(
    key: jax.Array,
    cluster_sizes: Integer[Array, "T"],
    n: int,
    *,
    alpha_init: float = 1.0,
    sigma_init: float = 0.1,
    n_warmup: int = 50,
    n_samples: int = 1,
    target_accept_prob: float = 0.95,
) -> tuple[tuple[float, float], dict[str, int]]:
    """Joint NUTS update for the PY hyperparameters.

    Parameters
    ----------
    key
        JAX PRNG key.
    cluster_sizes
        Strictly positive cluster sizes for the active prefix (length $T$).
    n
        Total reproducible feature count.
    alpha_init, sigma_init
        Chain start. Defaults sample-rate-friendly: ``α = 1`` matches the
        prior mode of $\\mathrm{Exp}(1)$, and ``σ = 0.1`` puts us safely away
        from the boundary at 0.
    n_warmup, n_samples
        NUTS warmup and post-warmup sample counts. Default ``n_samples = 1``
        — Algorithm 1 needs only the final position.
    target_accept_prob
        NUTS adaptation target. ``0.95`` is conservative; the joint $(\\alpha,
        \\sigma)$ posterior is well-conditioned on synthetic partitions but
        can become sharp under skewed cluster-size profiles.

    Returns
    -------
    ((alpha, sigma), diagnostics) — final-sample values + NUTS diagnostics
    (``num_divergences``, ``n_samples``).
    """
    cluster_sizes = jnp.asarray(cluster_sizes, dtype=jnp.int32)
    if cluster_sizes.size == 0:
        raise ValueError("cluster_sizes must be non-empty.")
    if not bool(jnp.all(cluster_sizes > 0)):
        raise ValueError("cluster_sizes must be strictly positive (drop empty clusters).")
    if not (0.0 < sigma_init < 1.0):
        raise ValueError(f"sigma_init must be in (0, 1); got {sigma_init}.")
    if alpha_init <= 0.0:
        raise ValueError(f"alpha_init must be > 0; got {alpha_init}.")

    kernel = NUTS(_py_hyperparam_model, target_accept_prob=target_accept_prob)
    mcmc = MCMC(
        kernel,
        num_warmup=n_warmup,
        num_samples=n_samples,
        num_chains=1,
        progress_bar=False,
    )
    init_params = {
        "alpha": jnp.asarray(alpha_init),
        "sigma": jnp.asarray(sigma_init),
    }
    mcmc.run(key, cluster_sizes, n, init_params=init_params)
    samples = mcmc.get_samples()
    alpha_final = float(samples["alpha"][-1])
    sigma_final = float(samples["sigma"][-1])
    diagnostics = {
        "num_divergences": int(jnp.sum(mcmc.get_extra_fields()["diverging"])),
        "n_samples": n_samples,
    }
    return (alpha_final, sigma_final), diagnostics
