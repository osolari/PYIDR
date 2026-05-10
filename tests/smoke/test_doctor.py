"""Smoke test: doctor() returns 0 on a healthy install."""

from __future__ import annotations

import pytest

import py_idr


@pytest.mark.smoke
def test_doctor_runs_and_is_green() -> None:
    """The skeleton-stage doctor() exits 0 (skipped probes do not count as failure)."""
    code = py_idr.doctor(verbose=False)
    assert code == 0
