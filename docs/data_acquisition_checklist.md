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
| **Source** | https://www.encodeproject.org/ — search for `Assay = ChIP-seq` + `Target = CTCF` + `Assembly = GRCh38` |
| **Picking experiments** | Filter to experiments with ≥ 2 biological replicates (we plan $K \in \{2, 6\}$). Record the experiment accession (e.g. `ENCSR000DZX`). |
| **Download** | `xargs -L1 curl -O` on the per-replicate BED narrowPeak files |
| **Hash** | `sha256sum *.bed` → record in the `DatasetManifest.sha256` field |
| **Replicate matching** | One BED file = one replicate column; record file → column mapping in the manifest's `replicate_matching_path` |
| **Score column** | `signalValue` (column 7) by default; `pValue` (column 8) flag-selectable |
| **Tie policy** | `mid_rank` (matches the empirical-rank transform's mid-rank ties) |
| **Missingness** | Peak unions per experiment; absent = drop |
| **Licence** | ENCODE 4 data is CC-BY 4.0 — usable without DAC approval. |
| **Expected $n$** | 30k–80k per replicate after peak union |

Loader: `py_idr.data.loaders.load_encode_ctcf`

## REAL-02: ENCODE 4 ATAC-seq

Same workflow as REAL-01 but with `Assay = ATAC-seq` and the broadPeak
file. Score is `signalValue` from MACS2 broad-peak calls. $K \in \{2, 4\}$,
$n \in [70k, 200k]$.

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
