"""Algorithm 1 step 4: atomic-type independence-Metropolis (IM) proposal.

Per §3.3.1 of the revised report, for each occupied cluster the atomic type
$t_m \\in \\{\\text{gauss}, t_5, \\text{clayton}, \\text{gumbel}\\}$ is updated by
**independence Metropolis with a uniform proposal over the four atomic
families**. We propose a fresh atomic type uniformly, draw fresh continuous
parameters from the type's prior, and accept with the standard MH ratio.

When parameters are proposed from the prior of the new type and the prior on
types is uniform, the proposal density and prior cancel symmetrically and the
acceptance ratio reduces to the **likelihood ratio**:

.. math::
   \\alpha = \\min\\!\\Bigl(1,\\,
       \\exp\\bigl[\\,\\ell(\\bU \\mid t_{\\text{new}}, \\theta_{\\text{new}}) -
                  \\ell(\\bU \\mid t_{\\text{cur}}, \\theta_{\\text{cur}})\\bigr]\\Bigr).

This is the simplest and most common IM-MH for discrete categorical state.

This module is custom because no library implements a discrete proposal over
copula families with the PY-IDR atomic-type prior support; the per-step
machinery (proposing, computing the acceptance ratio, accepting / rejecting)
is hand-rolled but the larger orchestrator that calls this lives inside a
``numpyro.infer.MCMCKernel`` (plan 11 §16).

See plans/04_inference_mcmc.md.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from py_idr.copulas.base import Copula
from py_idr.copulas.clayton import ClaytonCopula
from py_idr.copulas.gaussian import GaussianCopula
from py_idr.copulas.gumbel import GumbelCopula
from py_idr.copulas.registry import ATOMIC_TYPES
from py_idr.copulas.student_t import StudentTCopula

# Type tag → integer index for the IM categorical and the SamplerState array.
TYPE_TO_INDEX: dict[str, int] = {name: i for i, name in enumerate(ATOMIC_TYPES)}
INDEX_TO_TYPE: dict[int, str] = {i: name for name, i in TYPE_TO_INDEX.items()}


@dataclass(frozen=True)
class AtomDraw:
    """A single atomic-type draw produced by :func:`prior_propose_atom`.

    Attributes
    ----------
    type_name
        One of ``ATOMIC_TYPES`` (``"gauss"``, ``"t5"``, ``"clayton"``, ``"gumbel"``).
    type_index
        Position in :data:`ATOMIC_TYPES`.
    copula
        Concrete :class:`py_idr.copulas.base.Copula` ready for ``log_density``.
    params
        Dict of native parameters (``rho``, ``theta``, etc.) for serialisation.
    """

    type_name: str
    type_index: int
    copula: Copula
    params: dict[str, float]


def _propose_gauss(key: jax.Array, K: int) -> AtomDraw:
    """Propose an exchangeable Gaussian atom with $\\rho \\sim \\mathrm{Beta}(2, 2)$."""
    rho = float(jax.random.beta(key, 2.0, 2.0))
    R = (1.0 - rho) * jnp.eye(K) + rho * jnp.ones((K, K))
    L = jnp.linalg.cholesky(R)
    return AtomDraw("gauss", TYPE_TO_INDEX["gauss"], GaussianCopula(L), {"rho": rho})


def _propose_t5(key: jax.Array, K: int) -> AtomDraw:
    """Propose an exchangeable $t_5$ atom with $\\rho \\sim \\mathrm{Beta}(2, 2)$."""
    rho = float(jax.random.beta(key, 2.0, 2.0))
    R = (1.0 - rho) * jnp.eye(K) + rho * jnp.ones((K, K))
    L = jnp.linalg.cholesky(R)
    return AtomDraw("t5", TYPE_TO_INDEX["t5"], StudentTCopula(L), {"rho": rho})


def _propose_clayton(key: jax.Array, K: int) -> AtomDraw:
    """Propose a Clayton atom with $\\theta \\sim \\mathrm{Exp}(1)$."""
    theta = float(jax.random.exponential(key))
    # Exp(1) places mass at zero; clamp away from the boundary so the density
    # is finite. The §3.2.5 hyperprior is Gamma(1, 1) ≡ Exp(1) but the
    # reproducible component is the zero-discount limit excluded.
    theta = max(theta, 1e-3)
    return AtomDraw(
        "clayton", TYPE_TO_INDEX["clayton"], ClaytonCopula(theta=theta, K=K), {"theta": theta}
    )


def _propose_gumbel(key: jax.Array, K: int) -> AtomDraw:
    """Propose a Gumbel atom with $\\theta - 1 \\sim \\mathrm{Exp}(1)$.

    Bivariate only at this commit — the multivariate Gumbel density is the
    one open atomic-type follow-on (Hofert 2008 Stirling recursion).
    """
    theta = 1.0 + float(jax.random.exponential(key))
    return AtomDraw(
        "gumbel", TYPE_TO_INDEX["gumbel"], GumbelCopula(theta=theta, K=K), {"theta": theta}
    )


_PROPOSERS: dict[str, Callable[[jax.Array, int], AtomDraw]] = {
    "gauss": _propose_gauss,
    "t5": _propose_t5,
    "clayton": _propose_clayton,
    "gumbel": _propose_gumbel,
}


def supported_types_for_K(
    K: int,
    *,
    proposal_types: tuple[str, ...] = ATOMIC_TYPES,
) -> tuple[str, ...]:
    """Filter ``proposal_types`` to those whose copula density supports the given $K$.

    Multivariate Gumbel density ($K \\ge 3$) is the one open atomic-type follow-on
    (Hofert 2008 Stirling recursion). Until that lands, Gumbel proposals are
    excluded for $K \\ge 3$.
    """
    if K >= 3:
        return tuple(t for t in proposal_types if t != "gumbel")
    return proposal_types


def prior_propose_atom(
    key: jax.Array,
    type_name: str,
    K: int,
) -> AtomDraw:
    """Draw an atom of the requested type from its prior.

    Parameters
    ----------
    key
        JAX PRNG key.
    type_name
        One of :data:`ATOMIC_TYPES`.
    K
        Replicate dimension. Multivariate Gumbel ($K \\ge 3$) is not yet
        supported and raises through the underlying ``GumbelCopula.log_density``.
    """
    if type_name not in TYPE_TO_INDEX:
        raise ValueError(f"unknown atomic type {type_name!r}; supported: {ATOMIC_TYPES}")
    return _PROPOSERS[type_name](key, K)


def update_atomic_type(
    key: jax.Array,
    cluster_U: Float[Array, "n K"],
    current: AtomDraw,
    *,
    proposal_types: tuple[str, ...] = ATOMIC_TYPES,
) -> tuple[AtomDraw, bool, float]:
    """One independence-Metropolis step over atomic types for one cluster.

    Proposes ``new_type ~ Uniform(proposal_types)`` and ``new_params ~
    prior(new_type)``, then accepts via the likelihood ratio. Symmetric
    proposal + identical type-prior weights cancel, leaving only data
    likelihoods.

    Parameters
    ----------
    key
        JAX PRNG key.
    cluster_U
        Length-``(n_m, K)`` data matrix for the cluster being updated. Empty
        clusters (``n_m == 0``) should never be passed; the broader sweep
        handles their dropout via the Pólya-urn step.
    current
        Current atom draw (kept across sweeps in :class:`SamplerState`'s atom
        array; the test exercises this contract).
    proposal_types
        Subset of :data:`ATOMIC_TYPES` to propose from. The default is the
        full registry; restrict to enable the §4.6 Gaussian-only ablation.

    Returns
    -------
    (new_atom, accepted, log_alpha)
        ``new_atom`` is ``current`` if rejected else the freshly drawn atom;
        ``accepted`` is the boolean MH outcome; ``log_alpha`` is the
        un-clipped log acceptance ratio (useful for diagnostics).
    """
    if cluster_U.shape[0] == 0:
        raise ValueError("update_atomic_type called on an empty cluster (n_m = 0).")
    if not proposal_types:
        raise ValueError("proposal_types must be non-empty.")

    K = cluster_U.shape[1]
    # Filter out atomic types whose copula does not yet implement K.
    effective_types = supported_types_for_K(K, proposal_types=proposal_types)
    if not effective_types:
        raise ValueError(
            f"no atomic type in {proposal_types} supports K={K}. "
            "Multivariate Gumbel (K >= 3) is the one open follow-on."
        )

    type_key, param_key, accept_key = jax.random.split(key, 3)

    # Step 1: uniformly choose a proposal type.
    proposal_idx = int(jax.random.randint(type_key, (), 0, len(effective_types)))
    proposed_type = effective_types[proposal_idx]
    proposed = prior_propose_atom(param_key, proposed_type, K)

    # Step 2: log-likelihood ratio.
    log_lik_current = float(jnp.sum(current.copula.log_density(cluster_U)))
    log_lik_proposed = float(jnp.sum(proposed.copula.log_density(cluster_U)))
    log_alpha = log_lik_proposed - log_lik_current

    # Step 3: accept with probability min(1, exp(log_alpha)).
    u = float(jax.random.uniform(accept_key))
    accepted = bool(jnp.log(u) < log_alpha)
    new_atom = proposed if accepted else current
    return new_atom, accepted, log_alpha
