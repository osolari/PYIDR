"""Shared pytest fixtures for the StickIDR test suite."""

from __future__ import annotations

import pytest


@pytest.fixture(scope="session")
def base_seed() -> int:
    """A deterministic seed used by every test that needs randomness."""
    return 20260508
