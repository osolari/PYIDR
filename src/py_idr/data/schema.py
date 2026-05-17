"""Pydantic v2 schemas for the data and run manifests.

Implements IMPL-01 of the coding-agent handoff and the §4.7 + §4.9 requirements
of the revised report:

- :class:`DatasetManifest` — the pre-fit, per-dataset bundle every real-data run
  must produce before fitting any model (data manifest + feature universe +
  replicate matching + score dictionary + tie/missingness policy + QC summary).
- :class:`RunManifest` — the per-run reproducibility bundle (run id, git SHA,
  config hash, hardware, software versions, diagnostics verdict).
- :class:`SeedLedger` — the per-step PRNG-key derivations used to make a fit
  bit-identically replayable on CPU.

See plans/13_reproducibility.md and plans/08_real_data_pipelines.md.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, HttpUrl, field_validator


class DatasetManifest(BaseModel):
    """Pre-fit manifest required for every real-data dataset (§4.7 of the report).

    Per the report: "the coding agent must produce a data manifest, feature-universe
    definition, replicate-matching file, score column dictionary, tie and
    missingness policy, rank-transformation output, and dataset-specific QC
    summary before fitting any model." This class is the source of truth for that
    contract; the run script refuses to start a fit if validation fails.
    """

    # Non-strict so str -> Path / str -> HttpUrl coercion works on YAML loads;
    # extra="forbid" still catches typos in field names.
    model_config = ConfigDict(extra="forbid", strict=False)

    accession: str = Field(
        ...,
        description="Stable accession id from the originating consortium (ENCODE/DREAM5/...).",
    )
    release: str = Field(..., description="Consortium release tag, e.g. 'ENCODE 4'.")
    download_url: HttpUrl = Field(..., description="Canonical download URL.")
    sha256: str = Field(
        ...,
        description="SHA-256 of the raw downloaded archive.",
        pattern=r"^[0-9a-f]{64}$",
    )
    feature_universe_path: Path = Field(
        ...,
        description="Path to the canonical feature-universe file (n-row index, frozen).",
    )
    replicate_matching_path: Path = Field(
        ...,
        description="Path to the replicate->model-column mapping with conflict notes.",
    )
    score_dictionary_path: Path = Field(
        ...,
        description="Path to the score-column dictionary (units, monotone direction).",
    )
    tie_policy: Literal["mid_rank", "random", "truncate"] = Field(
        ...,
        description="How tied raw scores are broken before rank transformation.",
    )
    missingness_policy: Literal["mean_rank", "drop", "nan"] = Field(
        ...,
        description="How missing replicate scores are handled.",
    )
    qc_summary_path: Path = Field(
        ...,
        description="Per-replicate QC summary: histograms, missing/tie rate, suspicious flags.",
    )
    provenance_notes: str = Field(
        default="",
        description="Free-text provenance notes — human-readable history of edits.",
    )


class RunManifest(BaseModel):
    """Per-run reproducibility manifest written to ``runs/<run_id>/manifest.json``.

    The verdict ledger consumes this; the replay script derives its inputs from it.
    """

    model_config = ConfigDict(extra="forbid", strict=False)

    run_id: str = Field(
        ...,
        description="Sortable, globally unique id of the form '<unix_ts>-<6-hex>' (see utils.run.new_run_id).",
        pattern=r"^[0-9]{10,16}-[0-9a-f]{6,12}$",
    )
    git_sha: str = Field(
        ...,
        description="Full 40-character commit hash of HEAD at run start.",
        pattern=r"^[0-9a-f]{40}$",
    )
    config_hash: str = Field(
        ...,
        description="SHA-256 of the resolved Hydra config used by the run.",
        pattern=r"^[0-9a-f]{64}$",
    )
    dataset_manifest: DatasetManifest | None = Field(
        default=None,
        description="The pre-fit dataset manifest, when the run is a real-data fit.",
    )
    seed: int = Field(..., description="Base PRNG seed; recorded in the seed ledger.", ge=0)
    started_at: datetime = Field(..., description="UTC timestamp at run start.")
    finished_at: datetime | None = Field(default=None, description="UTC timestamp at run end.")
    walltime_s: float | None = Field(
        default=None, description="Wall-clock time in seconds.", ge=0.0
    )
    hardware: dict[str, str] = Field(
        default_factory=dict,
        description="Hardware metadata: gpu name, driver, host, cuda runtime.",
    )
    software_versions: dict[str, str] = Field(
        default_factory=dict,
        description="Software versions: jax, jaxlib, numpyro, py_idr.",
    )
    inference_method: Literal["mcmc", "vi"] = Field(
        ...,
        description="Inference engine used for the fit.",
    )
    diagnostics_verdict: Literal["pass", "warn", "fail"] | None = Field(
        default=None,
        description="Verdict from the diagnostics report (plan 12). Set after the fit completes.",
    )


class SeedLedger(BaseModel):
    """Per-step PRNGKey derivations from a single base seed.

    The Pólya-urn cluster reassignment, NUTS atom updates, and conjugate Bernstein
    update each consume a fresh key derived via :func:`jax.random.fold_in`. Storing
    the derivation log makes a CPU run bit-identically replayable; on GPU the
    posterior summary is replayable to within float tolerance.
    """

    model_config = ConfigDict(extra="forbid", strict=False)

    base_seed: int = Field(..., ge=0)
    derivations: dict[str, list[int]] = Field(
        default_factory=dict,
        description="Map of step name -> ordered list of fold_in inputs used.",
    )

    @field_validator("derivations")
    @classmethod
    def _no_empty_step_names(cls, v: dict[str, list[int]]) -> dict[str, list[int]]:
        """Disallow empty step names — they make the ledger ambiguous."""
        if any(not k for k in v):
            raise ValueError("derivation step names must be non-empty")
        return v
