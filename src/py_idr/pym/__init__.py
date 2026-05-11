"""Pitman-Yor process primitives.

This package collects the three representations of the Pitman-Yor
process that PY-IDR needs:

- :mod:`py_idr.pym.stickbreak` — the **stick-breaking** representation.
  Returns the truncated GEM weights $w_{1:T}$ that parameterise the
  random discrete measure. ``gem_sample`` is the public draw.
- :mod:`py_idr.pym.polya_urn` — the **predictive (Pólya urn)** form.
  Implements the per-feature cluster-assignment probabilities used by
  Algorithm 1 step 2.
- :mod:`py_idr.pym.eppf` — the **exchangeable partition probability
  function**. ``log_eppf(alpha, sigma, cluster_sizes, n)`` is the
  closed-form log probability of a partition under the PY prior;
  Algorithm 1 step 7 uses it as the NUTS target for the
  $(\\alpha, \\sigma)$ update.

The three forms are mathematically equivalent — which one is most
convenient depends on the sweep step. See the **Pitman-Yor primer**
in the docs site and §3.2.1 of the report.
"""
