# Plan 08 — Real Data Pipelines

**Owner:** W5
**Depends on:** 04_inference_mcmc, 05_inference_vi, 06_decision_theory
**Implements:** §4.7–§4.9 of the revised report; Table 7 (datasets); Figure F5 (real-data summaries); Tables T7 (real-disc, real-clusters, real-runtime).

## Goal

Five reproducible real-data pipelines that take raw consortium releases and produce a posterior local idr per feature, a Sun–Cai threshold, and the figures + tables for §4.8 of the paper.

## Pre-fit dataset manifest (revised §4.7)

The revised report mandates that **before any model is fit**, every dataset must produce
the following six artifacts. The pipeline refuses to start a fit until all six exist
and pass the schema check.

| Artifact | Schema (pydantic in `src/py_idr/data/schema.py`) |
|---|---|
| **Data manifest** | `accession`, `release`, `download_url`, `sha256`, `provenance_notes` |
| **Feature universe** | the canonical $n$-row index, frozen before any score is read |
| **Replicate-matching file** | which raw replicate $\to$ which model column $j$, with conflict notes |
| **Score-column dictionary** | which raw column is the model's "score", units, monotone direction |
| **Tie / missingness policy** | `mid_rank` / `random` / `truncate`; missing $\to$ what (mean rank, NaN, exclude) |
| **Dataset QC summary** | per-replicate score histograms, missing-rate, tie-rate, suspicious-replicate flags |

These are registered in the `DatasetManifest` pydantic model and validated by a CI hook
(`scripts/hooks/check_dataset_manifest.py`). Per the report's §4.7: "the coding agent
must produce a data manifest, feature-universe definition, replicate-matching file,
score column dictionary, tie and missingness policy, rank-transformation output, and
dataset-specific QC summary before fitting any model."

## Datasets

| ID | Dataset | $K$ | Approx. $n$ | Source |
|----|---------|-----|-------------|--------|
| D1 | ENCODE 4 CTCF ChIP-seq | 2–6 reps | 30k–80k peaks | ENCODE consortium release |
| D2 | ENCODE 4 ATAC-seq | 2–4 reps | 70k–200k peaks | ENCODE consortium release |
| D3 | DREAM5 | 35 algorithms | 100k edges | DREAM consortium archive |
| D4 | Tabula Muris pseudobulk | 8 tissues | 23k genes | Tabula Muris atlas (SmartSeq2 + 10x) |
| D5 | Pan-UKBB GWAS | 6 ancestries | 10M variants | Pan-UKBB DAC release |

## Deliverables

### `src/py_idr/data/`

- `encode_ctcf.py`
  - `download(target_dir)` — pulls MACS2 narrow-peak BEDs.
  - `load(experiment_id) -> pd.DataFrame` — narrow-peak coordinates + signalValue per replicate.
  - `to_score_matrix(df) -> jnp.ndarray` — $(n, K)$ matrix of replicate-aware peak scores.
- `encode_atac.py` — analogous, broad-peak.
- `dream5.py` — loads the four reference networks; emits a $(n, 35)$ rank matrix.
- `tabula_muris.py` — pseudobulks SmartSeq2 + 10x cells per tissue; uses `scanpy` / `anndata` to aggregate.
- `pan_ukbb.py` — loads the 6-ancestry summary statistics; emits a $(n, 6)$ z-score matrix.
- `splits.py` — registry of canonical SHA-256 hashes; `verify_checksums.sh` checks them.

### `experiments/<dataset>/`

Each contains:
- `manifest.yaml` — the pre-fit `DatasetManifest` (data manifest, feature universe, replicate-matching, score dictionary, tie/missingness policy, QC summary).
- `config.yaml` — Hydra config (model + inference + dataset).
- `run.py` — argparse wrapper; runs `validate_manifest(manifest_path)` first, then reads the score matrix, fits PY-IDR, writes a Zarr trace and a CSV of local idr / discoveries.
- `evaluate.py` — computes per-dataset summary stats (discovery counts, posterior median # clusters, walltime, $\delta_n$).
- `compare.py` — runs the comparators (Vanilla IDR, eCV, nestedIDR, ChIP-R, MaRR, GMCM-AD, BNP-Archimedean) on the same input. Per the revised §4.2, comparators are run only after their package version and default settings are *verified*; the verification log lives next to `compare.py`.

### `scripts/run_real_data.sh`

```bash
#!/usr/bin/env bash
set -euo pipefail
for ds in encode_ctcf encode_atac dream5 tabula_muris pan_ukbb; do
  python -m py_idr.experiments.${ds}.run \
    --config experiments/${ds}/config.yaml \
    --out artifacts/real/${ds}/
done
```

## Tests

- `tests/integration/test_encode_ctcf_dryrun.py` — `--dry-run` flag downloads a 100-peak fixture (committed to the repo) and fits without errors. Marked `slow`.
- `tests/integration/test_dream5_smoke.py` — fits on a 1000-edge subset of the *In silico* network in < 1 min.
- `tests/data/test_splits_checksums.py` — every checksum in `splits.py` matches a committed fixture.

## Acceptance criteria

- `scripts/download_data.sh && bash scripts/verify_checksums.sh` passes on a fresh machine.
- Every loader returns a clean $(n, K)$ score matrix free of NaN, with a documented tie-breaking strategy and rank transformation.
- Comparators are wrapped in `experiments/<dataset>/compare.py` so the runtime / discovery comparison of Figure F5 is automatic.

## Notes

- **ENCODE 4** signal scores are heteroscedastic across cell lines; this is the primary motivation for the per-replicate Bernstein–Dirichlet marginals.
- **DREAM5** $K = 35$ is the high-$K$ stress test of the report. The $K = 50$ simulation cell mirrors this.
- **Pan-UKBB** is dominated by the EUR ancestry sample size; the hierarchical PY structure is essential per §5.3.
- Comparator implementations: Vanilla IDR via `idr` R package wrapped via `subprocess`; eCV / nestedIDR / MaRR / ChIP-R via their published implementations; GMCM-AD via the autograd reference code; BNP-Archimedean via the Pan, Nieto-Barajas & Craiu (2024) reference.
