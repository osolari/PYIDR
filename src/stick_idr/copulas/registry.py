"""Atomic-type registry for the PY-IDR base measure.

ATOMIC_TYPES is the source of truth for the support of t_m in §3.3.2 of the report.
On import, each registered family is verified to satisfy the PLOD constraint of Eq. (3.4)
on a default u-grid; a violation aborts package import.

See plans/02_algebra_and_copulas.md.
"""

from __future__ import annotations

ATOMIC_TYPES: tuple[str, ...] = ("gauss", "t5", "clayton", "gumbel")
