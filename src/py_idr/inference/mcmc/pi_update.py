"""Algorithm 1 step 8: mixing-proportion update $\\pi \\sim \\mathrm{Beta}(1 + n_1, 1 + n_0)$.

The prior is $\\pi \\sim \\mathrm{Beta}(1, 1) \\equiv \\mathrm{Uniform}[0, 1]$
(see §3.2.5 of the revised report). Given the class assignments $Z$, the
conditional posterior is the conjugate Beta.

See plans/04_inference_mcmc.md.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, Integer


def update_pi(
    key: jax.Array,
    Z: Integer[Array, "n"],
    *,
    prior_a: float = 1.0,
    prior_b: float = 1.0,
) -> Float[Array, ""]:
    r"""Sample $\pi \mid Z \sim \mathrm{Beta}(\mathrm{prior\_a} + n_1, \mathrm{prior\_b} + n_0)$.

    Parameters
    ----------
    key
        JAX PRNG key.
    Z
        Length-$n$ class indicators in $\{0, 1\}$. The function counts
        $n_1 = \sum Z_i$ and $n_0 = n - n_1$.
    prior_a, prior_b
        Beta prior shape parameters. Default ``(1.0, 1.0)`` matches §3.2.5.

    Returns
    -------
    Scalar 0-D JAX array — the new $\pi$.

    Notes
    -----
    ``jax.random.beta`` expects strictly positive concentration parameters.
    With the default prior $(1, 1)$, $n_1$ and $n_0$ can be zero, but the
    posterior $\mathrm{Beta}(1 + n_1, 1 + n_0)$ remains well-defined.
    """
    if prior_a <= 0.0 or prior_b <= 0.0:
        raise ValueError(
            f"prior_a and prior_b must be strictly positive; got ({prior_a}, {prior_b})"
        )
    n1 = jnp.sum(Z.astype(jnp.int32))
    n0 = jnp.asarray(Z.shape[0], dtype=jnp.int32) - n1
    a = prior_a + n1.astype(jnp.float32)
    b = prior_b + n0.astype(jnp.float32)
    return jax.random.beta(key, a, b)
