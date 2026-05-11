"""Small utilities used across the codebase.

:mod:`py_idr.utils.run` contains the run-id and manifest-construction
helpers:

- ``new_run_id()`` — return a fresh ``<unix_ms>-<8-hex>`` identifier;
  lexicographically sortable by start time.
- ``now_utc()`` — current UTC timestamp in ISO-8601 form.
- ``detect_git_sha()`` — best-effort detection of the repo's
  ``HEAD`` SHA; falls back to ``"unknown"`` outside a working tree.
- ``build_run_manifest(...)`` — validated constructor for
  :class:`py_idr.data.schema.RunManifest`.
"""
