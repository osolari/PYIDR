# Real-data acquisition checklist

This is the per-dataset checklist a maintainer follows before the
`py_idr.data.loaders` for that dataset can be implemented. None of the
loaders run today — they raise `NotImplementedError` with a pointer
back here.

Owned by W12 (after the W11 MCMC scan refactor lands so production-scale
fits are tractable).

---

## REAL-01: ENCODE 4 CTCF ChIP-seq

| Item | Action |
|---|---|
| **Search URL** (corrected) | https://www.encodeproject.org/search/?type=Experiment&assay_title=TF+ChIP-seq&target.label=CTCF&assembly=GRCh38&status=released&files.output_type=peaks&files.file_type=bed+narrowPeak |
| ⚠ **Common pitfall** | Do **not** include `files.preferred_default=true` — that returns only the IDR-thresholded consensus call per experiment (1 BED per experiment), not the per-replicate raw calls PY-IDR needs. |
| **Download** | Click "Download" on the search-result page → get a `files.txt`. Drop into `.data/encode_ctcf_raw.txt`. |
| **Fetch + organise** | `bash scripts/fetch_encode.sh --files-txt .data/encode_ctcf_raw.txt --out data/encode_ctcf --min-reps 2` — fetches `metadata.tsv`, filters to raw per-replicate `bed narrowPeak`, groups by experiment, drops singletons, downloads in parallel. |
| **Hash** | `sha256sum *.bed.gz` → record in `DatasetManifest.sha256` field |
| **Score column** | `signalValue` (column 7) by default; `pValue` (column 8) flag-selectable |
| **Tie policy** | `mid_rank` (matches the empirical-rank transform's mid-rank ties) |
| **Missingness** | Peak unions per experiment; absent = drop |
| **Licence** | ENCODE 4 data is CC-BY 4.0 — usable without DAC approval. |
| **Expected $n$** | 30k–80k per replicate after peak union |

Loader: `py_idr.data.loaders.load_encode_ctcf`

## REAL-02: ENCODE 4 ATAC-seq

Same workflow as REAL-01 but for ATAC-seq. Corrected search URL:

`https://www.encodeproject.org/search/?type=Experiment&assay_title=ATAC-seq&assembly=GRCh38&status=released&files.output_type=peaks&files.file_type=bed+narrowPeak`

Same `bash scripts/fetch_encode.sh ...` workflow as REAL-01. Score
column is `signalValue` (column 7) from MACS2 broad-peak calls.
$K \in \{2, 4\}$, $n \in [70k, 200k]$ peaks per replicate.

Loader: `py_idr.data.loaders.load_encode_atac`

## REAL-03: DREAM5 gene-network challenge

| Item | Action |
|---|---|
| **Source** | https://www.synapse.org/Synapse:syn2787209 (DREAM5 archive) |
| **Download** | Per-algorithm rank list TSV files; one per network |
| **Replicate matching** | Trivial — each algorithm = one replicate. $K = 35$ |
| **Score column** | Algorithm's edge-rank value |
| **Tie policy** | `mid_rank` |
| **Missingness** | Algorithms reporting fewer than the full edge universe are flagged; drop those edges from the union |
| **Licence** | DREAM5 is publicly accessible; requires a Synapse account but no DAC |
| **Expected $n$** | ~100k edges per network |

Loader: `py_idr.data.loaders.load_dream5`

## REAL-04: Tabula Muris pseudobulk

| Item | Action |
|---|---|
| **Source** | https://tabula-muris.ds.czbiohub.org/ (FACS + droplet h5ad) |
| **Download** | The two h5ad files (Smart-Seq2 + 10x); record sha256 |
| **Pseudobulk aggregation** | Sum counts within (tissue, technology) groups → pseudobulk matrix |
| **Replicate matching** | Each (tissue, technology) cell becomes one replicate column |
| **Score column** | Per-replicate log-fold-change from a chosen DE comparison (default: tissue-vs-rest) |
| **Tie policy** | `mid_rank` |
| **Missingness** | Genes with zero pseudobulk in any replicate are excluded |
| **Licence** | Tabula Muris is CC-BY 4.0 |
| **Expected $n$** | ~23k genes after intersection |

Loader: `py_idr.data.loaders.load_tabula_muris`

## REAL-05: Pan-UK Biobank GWAS

| Item | Action |
|---|---|
| **Source** | https://pan.ukbb.broadinstitute.org/ |
| **DAC approval** | **REQUIRED** — apply to the UK Biobank Access Management System; allow several weeks. Without approval, no data may be touched. |
| **Download** | Per-ancestry summary statistics TSV.gz for a chosen phenotype |
| **Replicate matching** | One file = one ancestry column. $K \le 6$ |
| **Score column** | $|Z|$ (absolute beta / SE) or $-\log_{10}(p)$ |
| **Tie policy** | `mid_rank` |
| **Missingness** | Variants absent from any ancestry's GWAS are dropped from the universe |
| **LD pruning** | $r^2 < 0.1$ within 1Mb windows to keep $n$ tractable |
| **Licence** | UK Biobank ToS — data must stay on approved infrastructure; no public re-distribution |
| **Expected $n$** | ~10M variants pre-pruning, ~1M after LD pruning |

Loader: `py_idr.data.loaders.load_pan_ukbb`

---

## After acquisition

For each loaded dataset:

1. Run `py-idr doctor` to verify the JAX backend.
2. Write the `DatasetManifest` instance via the loader and save as
   `runs/<run_id>/data_manifests/<dataset>.yaml`.
3. Fit: `py-idr fit --config configs/data/<dataset>.yaml --inference mcmc`.
4. The §4.8 tables (`tab:real-disc`, `tab:real-clusters`,
   `tab:real-runtime`) regenerate from the resulting CSV via
   `py-idr table --table t7-real`. (T7 reproducer is planned for W12.)
5. Figure F6 (`fig:realdata`) regenerates from the per-dataset CSVs
   via `py-idr figure --fig f6`. (F6 reproducer is planned for W12.)
