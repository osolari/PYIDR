"""Shared pytest fixtures for the StickIDR test suite."""

from __future__ import annotations

import os

# Pin JAX to CPU for the default test backend. JAX-Metal is experimental and trips on
# several primitive lowerings (e.g. unknown StableHLO attribute codes); GPU-marked
# tests can opt into a CUDA/Metal backend explicitly.
os.environ.setdefault("JAX_PLATFORMS", "cpu")
# Run in float64 by default — copula log-densities and PLOD slack are tight in float32.
os.environ.setdefault("JAX_ENABLE_X64", "1")

import pytest  # noqa: E402


@pytest.fixture(scope="session")
def base_seed() -> int:
    """A deterministic seed used by every test that needs randomness."""
    return 20260508
