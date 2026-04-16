# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This is a research project for extracting and classifying bird mentions in 18th-century English texts from the ECCO (Eighteenth Century Collections Online) corpus. The full pipeline runs on the Puhti HPC cluster (CSC Finland) where the data lives.

## Remote Workflow

All scripts run on Puhti, not locally. Use `make` to sync and connect:

```bash
make push   # sync local works/ → puhti
make pull   # sync puhti works/ → local (excludes books/)
make ssh    # SSH into Puhti at the project dir with the Python env activated
```

Remote path: `chiunhau@puhti.csc.fi:/scratch/project_2017429/chiunhau/birds`  
Python env: `/scratch/project_2017429/chiunhau/my_python_env`

## Pipeline

Scripts in `works/` implement a four-stage pipeline:

1. **`filter_books_by_keywords.py`** — Scans the full ECCO JSONL (`ecco_downloaded.jsonl`) and writes matching books to `books/keywords_match.jsonl`. Skips if output exists. Uses multiprocessing.

2. **`extract_mentions.py`** — Extracts in-context mentions (±window tokens) for target animals from the filtered JSONL. Outputs `keywords_mentions.csv`. Supports `--exclude-if` and `--require-any` for disambiguation (e.g. "canary" the bird vs. Canary Islands).

3. **`deduplicate_mentions.py`** — Removes near-duplicate OCR rows using character 3-gram Jaccard similarity. Groups by `matched_term` before comparing. Default threshold: 0.80.

4. **`classify_mentions_vllm.py`** — Classifies each mention using a local vLLM server. Categories follow Tague framing: SLAVERY/DEPRIVATION/FREEDOM, HOSPITALITY/CARING/REFUGE, OTHER. Supports resume (skips already-classified rows).

## Running Classification on Puhti (SLURM)

Submit the GPU job from Puhti:

```bash
sbatch run_classification.sh
```

The script launches `vllm` serving `Qwen/Qwen2.5-32B-Instruct` on 4× V100 GPUs, waits for it to be healthy, then runs `classify_mentions_vllm.py`.

HF model cache: `/scratch/project_2017429/chiunhau/birds/hf-cache`

## Text Normalisation

All scripts share the same normalisation logic (long-s → s, ligatures, NFKD, diacritic stripping, dehyphenation). When modifying any script's normalisation, keep them in sync.

## Data Files (on Puhti only)

- `../../../laczakol/shared/ecco_downloaded/ecco_downloaded.jsonl` — full ECCO corpus
- `books/keywords_match.jsonl` — keyword-filtered books
- `keywords.txt` — one target word per line
- `keywords_mentions.csv` — extracted mentions
- `keywords_mentions_deduped.csv` — deduplicated mentions
- `keywords_mentions_classified_vllm.csv` — classified output
