# Reproducibility & manifests

Reproducibility in PY-IDR is enforced via three Pydantic schemas: the
`DatasetManifest`, the `RunManifest`, and the `SeedLedger`. Together they
give you the property that "if I have the run-id and the dataset
manifest, I can re-derive every byte the chain ever saw."

## The three manifests

### `DatasetManifest`

Describes the *input* dataset. Mandatory fields:

| Field                       | Meaning                                            |
|-----------------------------|----------------------------------------------------|
| `accession`                 | Dataset identifier, e.g. ENCSR000DUB.              |
| `release`                   | Release version, e.g. "ENCODE 4".                  |
| `download_url`              | Where to fetch the raw bytes.                      |
| `sha256`                    | 64-hex SHA-256 of the file at `download_url`.      |
| `feature_universe_path`     | Local path to the BED of features being scored.    |
| `replicate_matching_path`   | CSV mapping replicates to score columns.           |
| `score_dictionary_path`     | YAML defining the score column semantics.          |
| `tie_policy`                | One of `mid_rank`, `random`, `truncate`.           |
| `missingness_policy`        | One of `mean_rank`, `drop_feature`.                |
| `qc_summary_path`           | QC summary JSON for the dataset.                   |

The schema uses `extra = "forbid"` so typos in field names raise on load
rather than being silently ignored. The SHA-256 field is validated to
be exactly 64 lowercase hex characters; `tie_policy` and
`missingness_policy` are constrained to literal enums.

Code: [`py_idr.data.schema.DatasetManifest`][].

### `RunManifest`

Describes *one* execution of the chain. Mandatory fields:

| Field                  | Meaning                                                  |
|------------------------|----------------------------------------------------------|
| `run_id`               | `<unix_ms>-<8-hex>`; lexicographically sortable by time. |
| `git_sha`              | Full 40-character git commit hash.                       |
| `config_hash`          | 64-hex SHA-256 of the resolved Hydra config.             |
| `seed`                 | Integer base seed.                                       |
| `started_at`           | ISO-8601 UTC timestamp.                                  |
| `inference_method`     | One of `mcmc`, `vi`.                                     |

Optional fields:

| Field                  | Meaning                                                  |
|------------------------|----------------------------------------------------------|
| `dataset_manifest`     | Nested `DatasetManifest` (for real-data runs).           |
| `software_versions`    | `{package_name: version_string}`.                        |
| `hardware`             | `{gpu: "A10", driver: "535.183", ...}`.                  |
| `diagnostics_verdict`  | One of `pass`, `warn`, `fail`. Set after the run.        |

Helpers in `py_idr.utils.run`:

- `new_run_id()` — fresh `unix_ms-hex8` id.
- `now_utc()` — current UTC timestamp.
- `build_run_manifest(...)` — constructor with validation.

### `SeedLedger`

A small record of derived seeds:

```python
SeedLedger(
    base_seed=42,
    derivations={"sweep_0": [0, 1, 2, 3], "sweep_1": [4, 5, 6, 7]},
)
```

The base seed is the *one* integer the user supplies. Each derivation
records which derived seeds were used for which downstream sweep. This
makes it trivial to recover any specific chain from `(base_seed, step_name)`.

The validator rejects empty step names — they would be ambiguous.

## How they fit together

```
┌────────────────────────┐
│   DatasetManifest      │  ← what data the chain saw
│   (sha256, paths, ...) │
└──────────┬─────────────┘
           │ embedded in
           ▼
┌────────────────────────┐         ┌────────────────────────┐
│    RunManifest         │ ──seed──▶│      SeedLedger        │
│   (run_id, git_sha,    │         │   (base + derivations) │
│    config_hash, ...)   │         └────────────────────────┘
└──────────┬─────────────┘
           │ produced by one chain execution
           ▼
┌────────────────────────┐
│ InferenceData posterior│
│ + DiagnosticsReport    │
└────────────────────────┘
```

To reproduce a run:

1. Pin the git SHA: `git checkout <git_sha>`.
2. Re-derive the config: hash the rendered Hydra config and confirm it
   matches `config_hash`.
3. Re-fetch the dataset: download from `download_url`, verify
   `sha256`, place at the expected paths.
4. Re-run with the same `seed` and `inference_method`.

All four steps are independent. If any one fails, the run is not
reproducible from that manifest alone.

## Run IDs in practice

```python
from py_idr.utils.run import new_run_id, now_utc

rid = new_run_id()
# '1735000000000-a1b2c3d4'
```

The 64-bit Unix millisecond prefix and the 32-bit random suffix combine
to give near-certain uniqueness. The suffix was bumped from 24 to 32
bits after the birthday-paradox analysis in plan 13 — at 24 bits, a
sweep of 2000 simultaneous runs had a ~10% collision probability.

`new_run_id` is monotonic across ms boundaries: two ids generated
> 1 ms apart sort lexicographically by time.

## Writing manifests to disk

```python
from py_idr.data.schema import DatasetManifest

m = DatasetManifest(**dataset_dict)
with open("manifest.yaml", "w") as f:
    f.write(m.model_dump_json(indent=2))
```

YAML is also supported because all fields are JSON-native:

```python
import yaml
yaml_blob = yaml.safe_dump(dataset_dict)
m = DatasetManifest.model_validate(yaml.safe_load(yaml_blob))
```

## Common mistakes

1. **Storing the seed but not the git SHA.** The chain code can change
   under you; without `git_sha` the seed is meaningless.
2. **Storing the config blob but not its hash.** Two configs that look
   identical can differ by whitespace or float precision; the
   `config_hash` field forces a canonical comparison.
3. **Storing a hash of the dataset path instead of the dataset
   contents.** `sha256` should be the hash of the *bytes* at
   `download_url`, not of the URL string.
4. **Not validating on load.** Always round-trip a manifest through
   `Model.model_validate_json` before trusting it; the schema catches
   stale or corrupted manifests early.
