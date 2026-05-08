---
name: add-real-dataset
description: Add a new real-data loader to PY-IDR (analogous to ENCODE 4 CTCF, ENCODE 4 ATAC, DREAM5, Tabula Muris, Pan-UKBB). Use when the user asks to "add support for <dataset>", "wire up a new real-data pipeline", or "register a consortium release".
---

# Skill: Add a Real Dataset

Adding a real dataset to PY-IDR requires a downloader, a loader, a checksum entry, an
experiments folder, and a docs page.

## When to invoke

Triggered by any of:

- "add support for the <name> dataset"
- "wire up <consortium release> as a real-data target"
- "register a new ChIP-seq experiment"
- "add a Bioconductor / PLINK / cellranger output as a real dataset"

Do NOT use this skill for *evaluating* an existing dataset under a new comparator —
that's a config-only change.

## Prerequisites

- A canonical URL for the raw data (or instructions on how to obtain it).
- A documented score schema (e.g., MACS2 narrowPeak signalValue, GWAS z-score, etc.).
- Replicate count $K$ and feature count $n$ ranges.

## Procedure

1. **Decide the score per replicate.** This is a one-time decision; record it in
   `docs/experiments/<name>.md` as the canonical preprocessing rule. The choice
   determines the marginal model; rank-transform is the default.
2. **Add the loader.** Create `src/stick_idr/data/<name>.py` with:
   - `download(target_dir: Path, force: bool = False)` — fetches the raw release.
   - `load(...) -> pd.DataFrame` — parses to a tidy long-format DataFrame.
   - `to_score_matrix(df) -> jnp.ndarray` — emits $(n, K)$ score matrix.
3. **Register checksums.** Add SHA-256 hashes for the canonical fixture (committed
   to the repo at `tests/fixtures/<name>/`) to `data/splits.py`.
4. **Add a smoke test.** `tests/integration/test_<name>_dryrun.py` runs `download(--dry-run)`
   on the fixture and fits PY-IDR for 100 sweeps without errors.
5. **Add an experiments folder.** Under `experiments/<name>/`:
   - `config.yaml` — Hydra config selecting model + inference + dataset defaults.
   - `run.py` — argparse wrapper that calls `inference.mcmc.sweep` (or VI) on the score matrix.
   - `evaluate.py` — computes discovery counts, posterior median # clusters, walltime.
   - `compare.py` — runs Vanilla IDR, eCV, nestedIDR, ChIP-R, MaRR, GMCM-AD,
     BNP-Archimedean on the same input where applicable.
6. **Wire to the global launcher.** Append a stanza for `<name>` to
   `scripts/run_real_data.sh`.
7. **Update the dataset table.** Add a row to `src/stick_idr/_datasets.yaml` (the
   single source of truth for `tab:datasets`); `tables/t6_datasets.py` renders it on
   the next `make_all_tables.sh` run.
8. **Add a docs page.** `docs/experiments/<name>.md` opens with the dataset table row,
   the hypothesis (if any), the command that produces it, and the comparator list.
9. **Run the smoke gate:**
   ```bash
   bash scripts/verify_checksums.sh
   pytest tests/integration/test_<name>_dryrun.py
   pre-commit run --files src/stick_idr/data/<name>.py experiments/<name>/*
   ```

## Expected outputs

- `src/stick_idr/data/<name>.py`
- `experiments/<name>/{config.yaml, run.py, evaluate.py, compare.py}`
- `tests/integration/test_<name>_dryrun.py`
- `tests/fixtures/<name>/` (committed sample input)
- `docs/experiments/<name>.md`
- updated `src/stick_idr/_datasets.yaml`, `data/splits.py`, `scripts/run_real_data.sh`

## Failure modes

- **Replicate count not stable across experiments.** Some consortium releases vary $K$
  per experiment id (e.g. ENCODE 4 CTCF has $K \in \{2, 3, 4, 6\}$ depending on the cell
  line). The loader should expose a per-experiment $K$ and the run config should iterate
  over experiments rather than forcing a fixed $K$.
- **Score schema heterogeneity.** Different consortium releases use different score
  conventions; document the chosen one explicitly and add a fixture with a known schema.
- **Public availability.** If the dataset requires DAC approval (e.g., Pan-UKBB), the
  loader's `download()` raises a clear instructional error rather than failing silently.
