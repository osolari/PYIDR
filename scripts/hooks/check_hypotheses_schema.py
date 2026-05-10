#!/usr/bin/env python
"""Pre-commit hook: validate configs/preregistration/hypotheses.yaml.

Loads `py_idr.preregistration.schema.HypothesesFile` (pydantic) and parses the YAML.
A schema mismatch is a hard failure — preregistered thresholds must not silently drift.
"""

from __future__ import annotations

import sys
from pathlib import Path


def main(argv: list[str]) -> int:
    """Validate each path against the HypothesesFile pydantic schema."""
    try:
        import yaml  # noqa: F401
    except ImportError:
        sys.stderr.write("check-hypotheses-schema: PyYAML not installed; skipping.\n")
        return 0
    try:
        from py_idr.preregistration.schema import (
            HypothesesFile,  # type: ignore[import-not-found]
        )
    except (ImportError, NotImplementedError):
        sys.stderr.write("check-hypotheses-schema: schema not yet implemented (W3); skipping.\n")
        return 0
    for path in argv[1:]:
        raw = Path(path).read_text(encoding="utf-8")
        data = yaml.safe_load(raw)
        try:
            HypothesesFile.model_validate(data)
        except Exception as e:  # pragma: no cover - hook-only path
            sys.stderr.write(f"check-hypotheses-schema: {path} failed to validate:\n{e}\n")
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
