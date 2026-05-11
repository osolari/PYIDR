"""Synthetic simulation grid from §4.1 of the report.

Three submodules:

- :mod:`py_idr.simulation.scenarios` — the regime generators.
  ``simulate_S1`` through ``simulate_S4`` are implemented today;
  ``simulate_S5`` (heavy-tail Clayton/Gumbel/$t_5$ mixture) is the
  next milestone. Each generator returns a :class:`SimulationResult`
  carrying raw scores $X$, ground-truth class labels $Z$, and the
  parameters used.
- :mod:`py_idr.simulation.replicates` — end-to-end driver: simulate →
  pilot CDF → multi-chain fit → posterior local idr → Sun-Cai → flat
  :class:`ReplicateRow` ready for CSV. Both a generic ``run_replicate``
  dispatcher and ergonomic ``run_replicate_S1/S2/S3/S4`` wrappers.
- :mod:`py_idr.simulation.evaluation` — realised-FDR + power against
  the simulation truth at a given Sun-Cai operating level.

See plans/07_simulation_studies.md and §4.1 of the report.
"""
