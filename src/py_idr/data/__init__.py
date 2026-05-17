"""Pydantic schemas for dataset and run manifests.

PY-IDR enforces strict reproducibility via three manifest schemas in
:mod:`py_idr.data.schema`:

- :class:`DatasetManifest` — describes the input dataset (accession,
  release, SHA-256 of the bytes, paths to the feature universe and
  replicate-matching CSV, tie/missingness policies).
- :class:`RunManifest` — describes one chain execution (run id, git
  SHA, config hash, base seed, inference method, optional nested
  dataset manifest, software versions, hardware, diagnostics verdict).
- :class:`SeedLedger` — records the base seed and every derived seed
  used downstream.

All schemas use ``extra = "forbid"`` to catch typos early. Round-trip
JSON / YAML is supported. See plans/13_reproducibility.md and the
**reproducibility guide** in the docs site.
"""
