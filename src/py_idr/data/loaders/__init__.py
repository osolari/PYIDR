"""Real-data loaders for the §4.8 evaluation (planned).

Each dataset has its own submodule that produces:

- a frozen :class:`py_idr.data.schema.DatasetManifest` instance
- a per-replicate score matrix $X \\in \\R^{n \\times K}$
- a feature-universe index (e.g., a pandas Index of peak IDs)

All five loaders raise :class:`NotImplementedError` today; they are
blocked on data acquisition per plan~08 §"Required artifacts (revised
§4.7)". Each docstring records the upstream data source, the file
layout we expect, and the canonical preprocessing pipeline (MACS2,
pseudobulk aggregation, etc.) so the loader can be implemented in a
single self-contained commit once the data lands.

See:

- `plans/08_real_data_pipelines.md` — full design.
- `docs/data_acquisition_checklist.md` — per-dataset access steps
  (downloads, DAC applications, licensing).

The loaders all share the same signature:

>>> from py_idr.data.loaders import load_encode_ctcf
>>> manifest, X, features = load_encode_ctcf("/data/encode4/ctcf/experiment_<accession>/")

so the downstream pipeline (rank → chain → idr → Sun--Cai) is dataset
agnostic.
"""

from __future__ import annotations

__all__ = [
    "load_encode_ctcf",
    "load_encode_atac",
    "load_dream5",
    "load_tabula_muris",
    "load_pan_ukbb",
]


def _blocked(dataset_name: str, plan_section: str) -> "NotImplementedError":
    """Helper: emit a uniform "blocked on plan 08" error message."""
    return NotImplementedError(
        f"{dataset_name} loader is blocked on data acquisition (plan 08 §{plan_section}). "
        f"See docs/data_acquisition_checklist.md for the access workflow."
    )


def load_encode_ctcf(_path: str) -> None:
    """ENCODE 4 CTCF ChIP-seq loader (REAL-01, planned).

    Expected input: ENCODE 4 release directory with one BED file per
    biological replicate of a CTCF transcription-factor experiment.
    Per-replicate score = MACS2 narrow-peak ``signalValue`` (or
    ``-log10(p-value)`` if requested via a manifest flag). $K \\in \\{2, 6\\}$,
    $n \\in [30\\,000, 80\\,000]$ peaks per replicate after union over peak
    sets.
    """
    raise _blocked("ENCODE 4 CTCF ChIP-seq (REAL-01)", "Real Datasets")


def load_encode_atac(_path: str) -> None:
    """ENCODE 4 ATAC-seq loader (REAL-02, planned).

    Expected input: ENCODE 4 ATAC-seq experiment directory. Per-replicate
    score = MACS2 broad-peak ``signalValue`` or fold-enrichment.
    $K \\in \\{2, 4\\}$, $n \\in [70\\,000, 200\\,000]$ peaks.
    """
    raise _blocked("ENCODE 4 ATAC-seq (REAL-02)", "Real Datasets")


def load_dream5(_path: str) -> None:
    """DREAM5 gene-network challenge loader (REAL-03, planned).

    Expected input: DREAM archive directory with per-algorithm
    edge-ranking files. $K = 35$ (algorithms-as-replicates),
    $n \\approx 100\\,000$ edges per network. Replicate-matching = trivial
    (each algorithm is one replicate); score = the algorithm's edge
    rank.
    """
    raise _blocked("DREAM5 (REAL-03)", "Real Datasets")


def load_tabula_muris(_path: str) -> None:
    """Tabula Muris pseudobulk loader (REAL-04, planned).

    Expected input: Tabula Muris atlas h5ad files. Replicates = tissue
    × technology (Smart-Seq2 / 10x). Per-replicate score = pseudobulk
    log-fold-change of a chosen differential expression analysis.
    $K \\in \\{2, 8\\}$, $n \\approx 23\\,000$ genes after pseudobulk.
    """
    raise _blocked("Tabula Muris (REAL-04)", "Real Datasets")


def load_pan_ukbb(_path: str) -> None:
    """Pan-UK Biobank GWAS loader (REAL-05, planned).

    Expected input: Pan-UKBB summary-statistics tar.gz with per-ancestry
    Z-scores for a chosen phenotype. Replicates = ancestry groups
    passing QC. Per-replicate score = absolute Z-score.
    $K \\le 6$ ancestries, $n \\approx 10\\,000\\,000$ variants per
    phenotype (reduced via LD pruning).

    Note: requires Pan-UKBB Data Access Committee approval before any
    data is touched.
    """
    raise _blocked("Pan-UK Biobank (REAL-05)", "Real Datasets")
