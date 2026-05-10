"""Reproducibility-schema tests (plan 13)."""

from __future__ import annotations

import json
import time

import pytest
import yaml

from py_idr.data.schema import DatasetManifest, RunManifest, SeedLedger
from py_idr.utils.run import build_run_manifest, new_run_id, now_utc

_VALID_DATASET = {
    "accession": "ENCSR000DUB",
    "release": "ENCODE 4",
    "download_url": "https://www.encodeproject.org/files/ENCFF000XXX/download",
    "sha256": "a" * 64,
    "feature_universe_path": "data/encode_ctcf/feature_universe.bed",
    "replicate_matching_path": "data/encode_ctcf/replicate_match.csv",
    "score_dictionary_path": "data/encode_ctcf/score_dict.yaml",
    "tie_policy": "mid_rank",
    "missingness_policy": "mean_rank",
    "qc_summary_path": "data/encode_ctcf/qc.json",
}


# ---- DatasetManifest -----------------------------------------------------------------


@pytest.mark.unit
def test_dataset_manifest_round_trip() -> None:
    """A valid DatasetManifest serialises and parses back to itself."""
    m = DatasetManifest(**_VALID_DATASET)
    blob = m.model_dump_json()
    m2 = DatasetManifest.model_validate_json(blob)
    assert m == m2


@pytest.mark.unit
def test_dataset_manifest_rejects_unknown_field() -> None:
    """extra='forbid' catches typos in the manifest schema."""
    payload = dict(_VALID_DATASET, foo="bar")
    with pytest.raises((ValueError, TypeError)):
        DatasetManifest.model_validate(payload)


@pytest.mark.unit
def test_dataset_manifest_rejects_short_sha() -> None:
    """SHA-256 must be exactly 64 lowercase hex characters."""
    payload = dict(_VALID_DATASET, sha256="abc")
    with pytest.raises((ValueError, TypeError)):
        DatasetManifest.model_validate(payload)


@pytest.mark.unit
def test_dataset_manifest_rejects_unknown_tie_policy() -> None:
    """tie_policy is restricted to the literal set {mid_rank, random, truncate}."""
    payload = dict(_VALID_DATASET, tie_policy="exotic")
    with pytest.raises((ValueError, TypeError)):
        DatasetManifest.model_validate(payload)


@pytest.mark.unit
def test_dataset_manifest_yaml_load() -> None:
    """The manifest is loadable from YAML (the on-disk format used by run scripts)."""
    yaml_blob = yaml.safe_dump(_VALID_DATASET)
    m = DatasetManifest.model_validate(yaml.safe_load(yaml_blob))
    assert m.tie_policy == "mid_rank"


# ---- RunManifest ---------------------------------------------------------------------


@pytest.mark.unit
def test_run_manifest_basic_round_trip() -> None:
    """RunManifest with minimal required fields serialises and reloads."""
    rm = build_run_manifest(
        run_id=new_run_id(),
        git_sha="b" * 40,
        config_hash="c" * 64,
        seed=0,
        started_at=now_utc(),
        inference_method="mcmc",
    )
    blob = rm.model_dump_json()
    rm2 = RunManifest.model_validate_json(blob)
    assert rm == rm2


@pytest.mark.unit
def test_run_manifest_rejects_short_git_sha() -> None:
    """git_sha must be a full 40-character commit hash."""
    with pytest.raises((ValueError, TypeError)):
        build_run_manifest(
            run_id=new_run_id(),
            git_sha="short",
            config_hash="c" * 64,
            seed=0,
            started_at=now_utc(),
            inference_method="mcmc",
        )


@pytest.mark.unit
def test_run_manifest_rejects_unknown_inference_method() -> None:
    """inference_method is restricted to {mcmc, vi}."""
    with pytest.raises((ValueError, TypeError)):
        build_run_manifest(
            run_id=new_run_id(),
            git_sha="b" * 40,
            config_hash="c" * 64,
            seed=0,
            started_at=now_utc(),
            inference_method="abc",
        )


@pytest.mark.unit
def test_run_manifest_diagnostics_verdict_optional() -> None:
    """diagnostics_verdict is None when the run has not yet finished."""
    rm = build_run_manifest(
        run_id=new_run_id(),
        git_sha="b" * 40,
        config_hash="c" * 64,
        seed=0,
        started_at=now_utc(),
        inference_method="mcmc",
    )
    assert rm.diagnostics_verdict is None


@pytest.mark.unit
def test_run_manifest_with_dataset_manifest_round_trip() -> None:
    """A full RunManifest containing a DatasetManifest survives a JSON round-trip."""
    rm = build_run_manifest(
        run_id=new_run_id(),
        git_sha="b" * 40,
        config_hash="c" * 64,
        seed=42,
        started_at=now_utc(),
        inference_method="vi",
        dataset_manifest=_VALID_DATASET,
        software_versions={"jax": "0.4.30", "py_idr": "0.0.1"},
        hardware={"gpu": "A10", "driver": "535.183"},
    )
    blob = rm.model_dump_json()
    rm2 = RunManifest.model_validate_json(blob)
    assert rm2.dataset_manifest is not None
    assert rm2.dataset_manifest.accession == "ENCSR000DUB"
    assert rm2.software_versions["jax"] == "0.4.30"


# ---- SeedLedger ----------------------------------------------------------------------


@pytest.mark.unit
def test_seed_ledger_round_trip() -> None:
    """SeedLedger round-trips through JSON."""
    sl = SeedLedger(base_seed=42, derivations={"sweep_0": [1, 2, 3], "sweep_1": [4, 5]})
    blob = sl.model_dump_json()
    sl2 = SeedLedger.model_validate_json(blob)
    assert sl2 == sl


@pytest.mark.unit
def test_seed_ledger_rejects_empty_step_name() -> None:
    """Empty step names are ambiguous; the validator rejects them."""
    with pytest.raises((ValueError, TypeError)):
        SeedLedger(base_seed=0, derivations={"": [1]})


# ---- new_run_id --------------------------------------------------------------------


@pytest.mark.unit
def test_new_run_id_format() -> None:
    """new_run_id returns '<unix_ms>-<8-hex>' matching the RunManifest pattern."""
    rid = new_run_id()
    parts = rid.split("-")
    assert len(parts) == 2
    ts, suffix = parts
    assert ts.isdigit() and len(ts) >= 10
    assert len(suffix) == 8
    assert all(c in "0123456789abcdef" for c in suffix)


@pytest.mark.unit
def test_new_run_id_sortable_by_time() -> None:
    """Two ids generated >= 1 ms apart sort lexicographically by start time."""
    a = new_run_id()
    time.sleep(0.002)
    b = new_run_id()
    assert a < b


@pytest.mark.unit
def test_new_run_id_unique_within_millisecond() -> None:
    """The 24-bit random suffix makes within-ms collisions vanishingly unlikely."""
    ids = {new_run_id() for _ in range(2000)}
    assert len(ids) == 2000


# ---- json/yaml interop --------------------------------------------------------------


@pytest.mark.unit
def test_run_manifest_serialised_blob_is_valid_json() -> None:
    """model_dump_json output parses with the standard json module."""
    rm = build_run_manifest(
        run_id=new_run_id(),
        git_sha="b" * 40,
        config_hash="c" * 64,
        seed=0,
        started_at=now_utc(),
        inference_method="mcmc",
    )
    parsed = json.loads(rm.model_dump_json())
    assert parsed["seed"] == 0
    assert parsed["inference_method"] == "mcmc"
