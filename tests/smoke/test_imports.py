"""Smoke test: every public submodule imports cleanly."""

from __future__ import annotations

import importlib

import pytest

PUBLIC_MODULES = [
    "py_idr",
    "py_idr.algebra.correlation",
    "py_idr.algebra.plod",
    "py_idr.copulas.base",
    "py_idr.copulas.registry",
    "py_idr.marginals.bernstein",
    "py_idr.pym.polya_urn",
    "py_idr.pym.stickbreak",
    "py_idr.inference.mcmc.sweep",
    "py_idr.decision.sun_cai",
    "py_idr.cli",
    "py_idr.doctor",
]


@pytest.mark.smoke
@pytest.mark.parametrize("name", PUBLIC_MODULES)
def test_module_imports(name: str) -> None:
    """Every public submodule should import without side effects."""
    importlib.import_module(name)
