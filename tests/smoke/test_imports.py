"""Smoke test: every public submodule imports cleanly."""

from __future__ import annotations

import importlib

import pytest

PUBLIC_MODULES = [
    "stick_idr",
    "stick_idr.algebra.correlation",
    "stick_idr.algebra.plod",
    "stick_idr.copulas.base",
    "stick_idr.copulas.registry",
    "stick_idr.marginals.bernstein",
    "stick_idr.pym.algo8",
    "stick_idr.pym.stickbreak",
    "stick_idr.inference.mcmc.sweep",
    "stick_idr.decision.sun_cai",
    "stick_idr.cli",
    "stick_idr.doctor",
]


@pytest.mark.smoke
@pytest.mark.parametrize("name", PUBLIC_MODULES)
def test_module_imports(name: str) -> None:
    """Every public submodule should import without side effects."""
    importlib.import_module(name)
