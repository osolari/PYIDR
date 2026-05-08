"""StickIDR — Hierarchical nonparametric Bayesian IDR via Pitman-Yor copula mixtures.

Reference: docs/report/sections/3.method.tex (model + inference + theory),
4.experiments.tex (empirical protocol).
"""

from stick_idr._version import __version__
from stick_idr.doctor import doctor

__all__ = [
    "__version__",
    "doctor",
]
