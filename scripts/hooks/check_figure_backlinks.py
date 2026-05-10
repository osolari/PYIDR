#!/usr/bin/env python
"""Pre-commit hook: every figure/table reproducer references its LaTeX label.

Each module under src/py_idr/figures/ and src/py_idr/tables/ MUST contain its
target LaTeX label (e.g. `fig:calibration`, `tab:sim-fdr`) in the module docstring.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

LABEL_PATTERN = re.compile(r"\b(fig|tab):[a-zA-Z0-9_\-]+")


def main(argv: list[str]) -> int:
    """Scan the supplied paths and exit non-zero if any lack a fig:* or tab:* label."""
    bad: list[str] = []
    for path in argv[1:]:
        # __init__.py files are package-collectors and exempt from the backlink rule.
        if Path(path).name == "__init__.py":
            continue
        text = Path(path).read_text(encoding="utf-8")
        if not LABEL_PATTERN.search(text):
            bad.append(path)
    if bad:
        sys.stderr.write(
            "check-figure-backlinks: the following modules do not reference a "
            "fig:* or tab:* LaTeX label in their docstring:\n"
        )
        for p in bad:
            sys.stderr.write(f"  - {p}\n")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
