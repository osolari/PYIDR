"""arviz cross-version compatibility helpers.

arviz changed :func:`arviz.from_dict` between 0.x and 1.x in incompatible,
non-overlapping ways. We need code that works against either pin because the
two main deploy targets (local Python 3.12 + Apple Silicon, remote
GPU CUDA 12) end up resolving to different versions.
"""

from __future__ import annotations

import arviz as az
import numpy as np


def idata_from_posterior(posterior: dict[str, np.ndarray]) -> az.InferenceData:
    """Build an ``InferenceData`` whose ``posterior`` group holds ``posterior``.

    - **arviz 0.x** (e.g. 0.23): each group is a keyword argument
      ``az.from_dict(posterior=...)``. Passing ``{"posterior": ...}`` as the
      first positional arg does **not** raise — it silently puts the inner
      dict as a single variable named ``"posterior"`` inside the posterior
      group, and downstream code that does ``idata.posterior["pi"]`` then
      raises ``KeyError: 'pi'``.
    - **arviz 1.x** (e.g. 1.1): the first positional arg is a dict mapping
      group name to a dict of variables, ``az.from_dict({"posterior": ...})``.
      The ``posterior=`` kwarg is removed.

    Strategy: try the 1.x form first; verify the produced data_vars include
    the input keys; on mismatch, fall back to the 0.x kwarg form.

    Parameters
    ----------
    posterior
        Dict mapping variable name to a ``(num_chains, num_draws)`` array.

    Returns
    -------
    An ``arviz.InferenceData`` whose ``posterior`` group contains the input
    variables with the expected names.
    """
    idata = az.from_dict({"posterior": posterior})
    try:
        produced_vars = set(idata.posterior.data_vars)
    except AttributeError:
        produced_vars = set()
    if produced_vars >= set(posterior):
        return idata
    # 0.x path: the first form was silently wrong; use the kwarg form.
    return az.from_dict(posterior=posterior)
