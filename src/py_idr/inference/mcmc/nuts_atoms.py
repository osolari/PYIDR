"""Algorithm 1 step 3: NUTS atom-parameter update for active reproducible clusters.

Per §3.3.1 of the revised report, every active cluster's continuous parameters
are updated by NUTS (or Metropolis where gradients are unreliable). This module
provides a generic NumPyro NUTS wrapper plus concrete per-atom-type model
factories. The current scope covers the canonical baselines used by S1–S5:

- **Exchangeable Gaussian**: scalar $\\rho \\in (0, 1)$ with a $\\mathrm{Beta}(2, 2)$ prior.
- **Clayton**: scalar $\\theta > 0$ with an $\\mathrm{Exp}(1)$ prior (matches the
  $\\mathrm{Gamma}(1, 1)$ hyperprior of §3.2.5).
- **Bivariate Gumbel**: scalar $\\theta > 1$ with $\\theta - 1 \\sim \\mathrm{Exp}(1)$.

The structured factor classes (one-factor / two-factor) and the $t_5$ atom land
in a follow-up; the API below was designed so adding a new atomic type is a
single new model factory plus an entry in the dispatcher.

Notes
-----
We do **not** ask NUTS to reject proposals based on PLOD positive-pairwise
post-checks (PLOD rejection is a discrete-step operation that breaks NUTS's
continuous trajectory). Instead, every atomic type / structural class above
uses a parameterisation whose support is *automatically* PLOD-compatible:

- Exchangeable Gaussian with $\\rho \\in (0, 1)$ has all-positive correlations.
- Clayton with $\\theta > 0$ is strictly PLOD on the open interval.
- Gumbel with $\\theta > 1$ is strictly PLOD; the prior excludes the
  independence-boundary point $\\theta = 1$.

For atomic types whose natural prior support overlaps the independence
boundary (e.g., LKJ Cholesky on a free $K \\times K$ correlation matrix), the
follow-up will use rejection at the *atom proposal* level (a Metropolis-Hastings
correction) rather than mid-trajectory.

See plans/04_inference_mcmc.md.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
from jaxtyping import Array, Float
from numpyro.infer import MCMC, NUTS

from py_idr.copulas.clayton import ClaytonCopula
from py_idr.copulas.gaussian import GaussianCopula
from py_idr.copulas.gumbel import GumbelCopula


def nuts_atom_update(
    key: jax.Array,
    model_fn: Callable[..., None],
    init_params: dict[str, Float[Array, "..."]],
    *,
    model_args: tuple[Any, ...] = (),
    model_kwargs: dict[str, Any] | None = None,
    n_warmup: int = 50,
    n_samples: int = 1,
    target_accept_prob: float = 0.95,
) -> tuple[dict[str, Float[Array, "..."]], dict[str, int]]:
    """Run a short NumPyro NUTS chain on a single cluster's atom parameters.

    Parameters
    ----------
    key
        JAX PRNG key.
    model_fn
        A NumPyro model function. Must declare ``numpyro.sample`` for each
        latent atom parameter and ``numpyro.factor`` for the cluster's data
        log-likelihood (the model factories below handle this).
    init_params
        Current values of the atom parameters; used as the NUTS chain start
        so warmup is short and the chain doesn't drift from a posterior mode.
    model_args, model_kwargs
        Positional and keyword arguments passed to ``model_fn``. The data
        observations are typically passed here.
    n_warmup
        NUTS warmup iterations. Default 50 is enough for a smooth scalar
        posterior; raise to 200+ for higher-dim factor parameterisations.
    n_samples
        Post-warmup samples. Default 1 — Algorithm 1 needs only the final
        position; the ``MCMC`` object also exposes the divergence count.
    target_accept_prob
        NUTS adaptation target. Default 0.95 (high) is conservative and tends
        to produce zero divergent transitions on the simple atom posteriors
        above; the §4.6 stress tests can re-tune this.

    Returns
    -------
    (final_params, diagnostics)
        ``final_params`` is the parameter dictionary at the last post-warmup
        sample; ``diagnostics`` carries ``num_divergences`` and ``n_samples``.
    """
    if model_kwargs is None:
        model_kwargs = {}
    kernel = NUTS(model_fn, target_accept_prob=target_accept_prob)
    mcmc = MCMC(
        kernel,
        num_warmup=n_warmup,
        num_samples=n_samples,
        num_chains=1,
        progress_bar=False,
    )
    mcmc.run(
        key,
        *model_args,
        init_params=init_params,
        **model_kwargs,
    )
    samples = mcmc.get_samples()
    final = {name: samples[name][-1] for name in samples}
    diagnostics = {
        "num_divergences": int(jnp.sum(mcmc.get_extra_fields()["diverging"])),
        "n_samples": n_samples,
    }
    return final, diagnostics


# ---- model factories ---------------------------------------------------------------


def model_exchangeable_gaussian(
    U: Float[Array, "n K"],
) -> None:
    """NumPyro model for an exchangeable-correlation Gaussian-copula atom.

    Latent parameter
    ----------------
    rho : Beta(2, 2) on (0, 1)
        The single shared off-diagonal correlation. Beta(2, 2) is concentrated
        away from 0 and 1, matching the §3.2.2 PLOD-compatible support.
    """
    K = U.shape[1]
    rho = numpyro.sample("rho", dist.Beta(2.0, 2.0))
    R = (1.0 - rho) * jnp.eye(K) + rho * jnp.ones((K, K))
    L = jnp.linalg.cholesky(R)
    log_lik = jnp.sum(GaussianCopula(L).log_density(U))
    numpyro.factor("data", log_lik)


def model_clayton(
    U: Float[Array, "n K"],
) -> None:
    """NumPyro model for a Clayton atom with scalar dependence parameter $\\theta > 0$.

    Latent parameter
    ----------------
    theta : Exponential(1) on (0, ∞)
        Matches the §3.2.5 hyperprior of $\\mathrm{Gamma}(1, 1)$.
    """
    K = U.shape[1]
    theta = numpyro.sample("theta", dist.Exponential(1.0))
    log_lik = jnp.sum(ClaytonCopula(theta=theta, K=K).log_density(U))
    numpyro.factor("data", log_lik)


def model_gumbel_k2(
    U: Float[Array, "n 2"],
) -> None:
    """NumPyro model for a bivariate Gumbel atom with $\\theta > 1$.

    Latent parameter
    ----------------
    theta_minus_one : Exponential(1) on (0, ∞)
        Reparameterised so $\\theta = 1 + \\eta$, $\\eta \\sim \\mathrm{Exp}(1)$.
        This excludes the independence boundary $\\theta = 1$ as §3.2.2 requires.

    Notes
    -----
    Multivariate Gumbel ($K \\geq 3$) needs the Hofert (2008) Stirling-recursion
    density, deferred to a follow-up; this factory is the bivariate baseline.
    """
    if U.shape[1] != 2:
        raise ValueError(f"model_gumbel_k2 requires K=2; got K={U.shape[1]}.")
    theta_minus_one = numpyro.sample("theta_minus_one", dist.Exponential(1.0))
    theta = 1.0 + theta_minus_one
    log_lik = jnp.sum(GumbelCopula(theta=theta, K=2).log_density(U))
    numpyro.factor("data", log_lik)


# ---- per-atom-type wrappers --------------------------------------------------------


def update_exchangeable_gaussian(
    key: jax.Array,
    cluster_U: Float[Array, "n K"],
    rho_init: float,
    *,
    n_warmup: int = 50,
    n_samples: int = 1,
    target_accept_prob: float = 0.95,
) -> tuple[float, dict[str, int]]:
    """Single-cluster NUTS update for the exchangeable Gaussian atom's $\\rho$."""
    init_params = {"rho": jnp.asarray(rho_init)}
    final, diag = nuts_atom_update(
        key,
        model_exchangeable_gaussian,
        init_params,
        model_args=(cluster_U,),
        n_warmup=n_warmup,
        n_samples=n_samples,
        target_accept_prob=target_accept_prob,
    )
    return float(final["rho"]), diag


def update_clayton(
    key: jax.Array,
    cluster_U: Float[Array, "n K"],
    theta_init: float,
    *,
    n_warmup: int = 50,
    n_samples: int = 1,
    target_accept_prob: float = 0.95,
) -> tuple[float, dict[str, int]]:
    """Single-cluster NUTS update for the Clayton atom's $\\theta$."""
    init_params = {"theta": jnp.asarray(theta_init)}
    final, diag = nuts_atom_update(
        key,
        model_clayton,
        init_params,
        model_args=(cluster_U,),
        n_warmup=n_warmup,
        n_samples=n_samples,
        target_accept_prob=target_accept_prob,
    )
    return float(final["theta"]), diag


def update_gumbel_k2(
    key: jax.Array,
    cluster_U: Float[Array, "n 2"],
    theta_init: float,
    *,
    n_warmup: int = 50,
    n_samples: int = 1,
    target_accept_prob: float = 0.95,
) -> tuple[float, dict[str, int]]:
    """Single-cluster NUTS update for the bivariate Gumbel atom's $\\theta$."""
    if theta_init <= 1.0:
        raise ValueError(f"theta_init must be > 1 for Gumbel; got {theta_init}")
    init_params = {"theta_minus_one": jnp.asarray(theta_init - 1.0)}
    final, diag = nuts_atom_update(
        key,
        model_gumbel_k2,
        init_params,
        model_args=(cluster_U,),
        n_warmup=n_warmup,
        n_samples=n_samples,
        target_accept_prob=target_accept_prob,
    )
    return 1.0 + float(final["theta_minus_one"]), diag
