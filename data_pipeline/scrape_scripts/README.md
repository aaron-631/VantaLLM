# `data_pipeline/scrape_scripts/` — Data Acquisition & Preprocessing

This directory contains all scripts that **acquire raw data and convert it into a pipeline-ready format** compatible with `run_pipeline.sh`.

After running any script here, drop the output `.jsonl` file into `data_pipeline/raw/` and run:

```bash
./run_pipeline.sh
```

The pipeline will automatically normalize → tokenize → shard the new data and add it to the training set.

---

## Scripts Overview

| Script | Purpose |
|---|---|
| [`prep_c4.py`](#prep_c4py) | Preprocess RedPajama C4 web-text shards (**primary open-source dataset script**) |
| `scrape_arxiv.py` | Scrape cybersecurity arXiv papers |
| `scrape_rss.py` | Scrape cybersecurity RSS feeds |
| `scrape_repos.py` | Scrape public GitHub repositories |
| `parse_pdfs.py` | Parse local PDF files into text |
| `download_defcon.py` | Download DEF CON talk transcripts |
| `fix_exploitdb.py` | Fix/clean ExploitDB dataset records |
| `catch_up.py` | Catch-up ingestion for missed feeds |
| `dedup_merge.py` | Merge and deduplicate multiple raw files |

---

## `prep_c4.py`

The main script for processing **RedPajama C4** web-text data — the largest open-source pretraining dataset VantaLLM uses.

### What it does

Converts a raw RedPajama C4 JSONL shard into a clean `data_pipeline/raw/`-ready file.

```
Input  → {"text": "...", "meta": {...}, "red_pajama_subset": "c4"}
Output → {"text": "..."}
```

It applies **9 sequential filters** (cheapest first for performance):

| # | Filter | What it drops |
|---|---|---|
| 1 | Schema extraction | Records missing or non-string `text` field |
| 2 | Length | Docs < 200 chars **or** < 40 words |
| 3 | Language/script | Non-Latin-script documents (ASCII ratio < 85%) |
| 4 | Boilerplate | Footer patterns in last 15 lines (cookie banners, nav menus) |
| 5 | Repetition | SEO spam where top trigram > 25% of all trigrams |
| 6 | Punctuation density | Structured-data dumps where `:|/\` chars > 12% of text |
| 7 | URL density | Link farms where > 20% of lines are URL-dominated |
| 8 | Text cleaning | Normalize whitespace, strip control characters |
| 9 | Deduplication | MD5 exact-match dedup within this run |

---

### Requirements

No extra dependencies — uses Python 3.9+ standard library only:

```
io, os, re, sys, json, hashlib, logging, argparse, collections, dataclasses, pathlib, typing
```

---

### Getting the data

Download a single C4 shard (~500MB) from the RedPajama dataset:

```bash
# Single shard (recommended for first run):
wget https://data.together.xyz/redpajama-data-1T/v1.0.0/c4/c4-train.00000-of-01024.jsonl

# Or use the Hugging Face mirror:
# https://huggingface.co/datasets/togethercomputer/RedPajama-Data-1T
```

> **Note:** There are 1,024 shards total (~500GB combined). Start with **shard 00000** to validate the pipeline before ingesting more.

---

### Usage

#### Basic — single shard

```bash
python data_pipeline/scrape_scripts/prep_c4.py \
    --input  path/to/c4-train.00000-of-01024.jsonl \
    --output data_pipeline/raw/c4_shard_00000.jsonl
```

#### Dry-run first (recommended)

Always dry-run first to see filter stats before writing any output:

```bash
python data_pipeline/scrape_scripts/prep_c4.py \
    --input  path/to/c4-train.00000-of-01024.jsonl \
    --output data_pipeline/raw/c4_shard_00000.jsonl \
    --dry-run
```

Example output:
```
────────────────────────────────────────────────────
  File : c4-train.00000-of-01024.jsonl
  Total records       :  364,868
  ✅ Kept             :  271,340  (74.4%)
  ❌ Malformed JSON   :        0
  ❌ Empty field      :       12
  ❌ Too short        :    18,422
  ❌ Non-Latin script :     6,891
  ❌ Boilerplate      :     3,204
  ❌ Repetition spam  :     9,112
  ❌ Punct density    :     4,781
  ❌ URL density      :     1,106
  ❌ Duplicate        :        0
────────────────────────────────────────────────────
```

#### Multiple shards in a directory

```bash
python data_pipeline/scrape_scripts/prep_c4.py \
    --input_dir path/to/c4_shards/ \
    --output    data_pipeline/raw/c4_combined.jsonl
```

Processes all `.jsonl` files in the directory in sorted order, appending to a single output file. Cross-file deduplication is handled automatically.

#### Overwrite existing output

```bash
python data_pipeline/scrape_scripts/prep_c4.py \
    --input  path/to/shard.jsonl \
    --output data_pipeline/raw/c4_shard_00000.jsonl \
    --overwrite
```

---

### All CLI flags

| Flag | Type | Default | Description |
|---|---|---|---|
| `--input` | str | — | Single input `.jsonl` file *(mutually exclusive with `--input_dir`)* |
| `--input_dir` | str | — | Directory of `.jsonl` shards to process *(mutually exclusive with `--input`)* |
| `--output` | str | — | Output `.jsonl` path (**required**) |
| `--overwrite` | flag | `False` | Overwrite output if it exists (default: append) |
| `--dry-run` | flag | `False` | Print filter stats without writing anything |
| `--min_chars` | int | `200` | Minimum character count per document |
| `--max_chars` | int | `50000` | Maximum character count per document |
| `--min_words` | int | `40` | Minimum word count (approximate) per document |
| `--min_ascii` | float | `0.85` | Minimum fraction of printable ASCII characters |
| `--boilerplate_tail` | int | `15` | Lines to scan at tail for boilerplate patterns |
| `--max_trigram_ratio` | float | `0.25` | Max fraction a single trigram can represent (repetition filter) |
| `--max_punct_ratio` | float | `0.12` | Max fraction of chars that are `:|/\` (punct density filter) |
| `--max_url_line_ratio` | float | `0.20` | Max fraction of lines that are URL-dominated (link farm filter) |

---

### After running — plug into the pipeline

Once your output file is in `data_pipeline/raw/`:

```bash
./run_pipeline.sh
```

This triggers the 4-step incremental pipeline:

```
normalize_data.py  →  dedup_merge.py  →  tokenize_data.py  →  shard_data.py
```

The manifest (`data_pipeline/pipeline_manifest.json`) tracks which files have been processed — re-running is safe and idempotent.

---

### Memory behaviour

The script holds dedup hashes in RAM (`seen_hashes: Set[bytes]`):

| Scenario | RAM used |
|---|---|
| 1 shard (~365K docs) | ~20 MB |
| 10 shards (~3.6M docs) | ~200 MB |
| 100+ shards | Consider switching to SQLite-backed dedup |

For very large batch runs, replace `seen_hashes` with a `shelve` or `sqlite3` store.

---

### Design notes for contributors

- **Filter ordering is intentional** — length and language checks (O(n) in C) run before regex and n-gram checks (more expensive). Do not reorder.
- **Digit ratio filter is intentionally absent** — cybersecurity text contains CVE IDs, IP addresses, port numbers, and hashes. A digit cap would drop valid training data.
- **Dedup happens after cleaning** — so two docs differing only in whitespace are correctly identified as the same document.
- **The repetition filter strips punctuation** before computing trigrams (matching the LLaMA-2 pipeline) so `"Buy."` and `"buy"` resolve to the same token.
- **Dry-run uses `io.StringIO`** instead of `os.devnull` to avoid a Windows encoding incompatibility (`nul` is a binary device and raises `ValueError` with `encoding="utf-8"`).

---

### Tuning thresholds

If your dry-run shows an unexpectedly high drop rate on a specific filter, tune the corresponding threshold:

```bash
# Too many docs dropped by repetition filter? Raise the threshold:
python prep_c4.py --input shard.jsonl --output out.jsonl \
    --max_trigram_ratio 0.35 --dry-run

# Too many docs dropped by punctuation filter? Raise it:
python prep_c4.py --input shard.jsonl --output out.jsonl \
    --max_punct_ratio 0.18 --dry-run
```

A healthy C4 shard should keep **65–80%** of records after all filters.

---

*Last audited: June 2026 — all 9 filter functions verified correct against the LLaMA-2 data pipeline reference implementation.*
