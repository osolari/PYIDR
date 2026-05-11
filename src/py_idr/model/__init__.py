"""Chain-state containers.

Two state types because the multi-cluster path has variable-length
per-cluster atoms that do not fit into a fixed-shape JAX array:

- :mod:`py_idr.model.state` — :class:`SimpleSweepState` for the
  $T = 1$ fast path. Pure JAX, fully traceable; the entire sweep can
  be ``jit``-compiled.
- :mod:`py_idr.model.multi_state` — :class:`MultiClusterState` for the
  $T > 1$ path. Hybrid container: fixed-shape arrays for
  $(\\pi, M, \\mathrm{weights}, z, \\alpha, \\sigma)$ plus a Python
  tuple of dataclasses for the variable-length atom list. The sweep
  is Python on the outside, JIT-compiled per step on the inside.

The Python overhead of the multi-cluster path is small (~milliseconds
per sweep); the trade-off is correctness — variable atom counts and
mixed copula types cannot be expressed as a single JAX array without
heavy padding and indexing tricks.

See plans/04_inference_mcmc.md.
"""
