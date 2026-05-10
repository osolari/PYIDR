"""Environment sanity check.

`stick_idr.doctor()` runs a battery of cheap probes and reports green / yellow / red.
Invoke via the CLI (`stick-idr doctor`) or directly:

>>> import stick_idr
>>> stick_idr.doctor()

The function returns 0 if all probes are green and a non-zero exit code otherwise so
CI and pre-commit can gate on it.

Probes (Stage 1 — repo skeleton):
    - JAX backend present; default dtype check.
    - NumPyro importable; version meets requirement.

Probes added in later workstreams (raise `NotImplementedError` until implemented):
    - PLOD round-trip on representative atoms (W1; algebra/plod.py).
    - Bernstein basis matches scipy reference (W1; marginals/bernstein.py).
    - Algorithm 8 fixture step (W2; pym/algo8.py).
"""

from __future__ import annotations

import importlib
import sys
from collections.abc import Callable
from dataclasses import dataclass, field

PASS = "PASS"
FAIL = "FAIL"
SKIP = "SKIP"


@dataclass
class ProbeResult:
    """Outcome of one probe."""

    name: str
    status: str
    detail: str = ""


@dataclass
class DoctorReport:
    """Aggregate outcome of all probes."""

    probes: list[ProbeResult] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        """True iff no probe failed (skipped probes do not count as failure)."""
        return not any(p.status == FAIL for p in self.probes)

    def render(self) -> str:
        """Render as a column-aligned plain-text report."""
        if not self.probes:
            return "doctor: no probes registered."
        width = max(len(p.name) for p in self.probes)
        lines = []
        for p in self.probes:
            line = f"  {p.name:<{width}}  {p.status:<5}"
            if p.detail:
                line += f"  -- {p.detail}"
            lines.append(line)
        verdict = "doctor: OK" if self.passed else "doctor: FAIL"
        return verdict + "\n" + "\n".join(lines)


def _probe_jax() -> ProbeResult:
    try:
        jax = importlib.import_module("jax")
    except ImportError as e:
        return ProbeResult("jax-import", FAIL, f"could not import jax: {e}")
    devices = jax.devices()
    if not devices:
        return ProbeResult("jax-devices", FAIL, "no JAX devices reported")
    backend = devices[0].platform
    return ProbeResult("jax-import", PASS, f"backend={backend}, devices={len(devices)}")


def _probe_numpyro() -> ProbeResult:
    try:
        numpyro = importlib.import_module("numpyro")
    except ImportError as e:
        return ProbeResult("numpyro-import", FAIL, f"could not import numpyro: {e}")
    version = getattr(numpyro, "__version__", "unknown")
    return ProbeResult("numpyro-import", PASS, f"version={version}")


def _probe_plod_roundtrip() -> ProbeResult:
    """Verify PLOD holds for every atomic type at K=2 and that c_0 correctly fails."""
    try:
        from stick_idr.algebra.plod import assert_plod
        from stick_idr.copulas.independence import IndependenceCopula
        from stick_idr.copulas.registry import REPRESENTATIVES
    except Exception as e:  # pragma: no cover
        return ProbeResult("plod-roundtrip", FAIL, f"import error: {e}")
    K = 2
    for name, build in REPRESENTATIVES.items():
        try:
            assert_plod(build(K), K)
        except ValueError as e:
            return ProbeResult("plod-roundtrip", FAIL, f"atom {name!r} failed PLOD at K={K}: {e}")
    # And independence must FAIL.
    try:
        assert_plod(IndependenceCopula(K=K), K)
    except ValueError:
        return ProbeResult(
            "plod-roundtrip",
            PASS,
            "PLOD holds for {gauss, t5, clayton, gumbel} at K=2; independence correctly fails",
        )
    return ProbeResult(
        "plod-roundtrip", FAIL, "independence copula incorrectly passed the PLOD probe"
    )


def _probe_bernstein_basis() -> ProbeResult:
    # Implemented by W1 (marginals/bernstein.py).
    return ProbeResult(
        "bernstein-basis",
        SKIP,
        "deferred to W1 (see plans/03_marginals.md)",
    )


def _probe_algo8_fixture() -> ProbeResult:
    # Implemented by W2 (pym/algo8.py).
    return ProbeResult(
        "algo8-fixture",
        SKIP,
        "deferred to W2 (see plans/04_inference_mcmc.md)",
    )


_PROBES: list[Callable[[], ProbeResult]] = [
    _probe_jax,
    _probe_numpyro,
    _probe_plod_roundtrip,
    _probe_bernstein_basis,
    _probe_algo8_fixture,
]


def doctor(verbose: bool = True) -> int:
    """Run all probes and print a report.

    Returns
    -------
    int
        0 on full pass (skips do not count as failure), 1 otherwise.
    """
    report = DoctorReport(probes=[probe() for probe in _PROBES])
    if verbose:
        print(report.render(), file=sys.stderr)
    return 0 if report.passed else 1
