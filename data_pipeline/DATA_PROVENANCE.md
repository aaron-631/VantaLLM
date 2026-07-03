# Data Provenance & Mixture

This manifest records **where the example pretraining corpus comes from**, its
approximate mixture, and the licensing status the operator is responsible for
confirming. It complements [`DATASET_SPEC.md`](./DATASET_SPEC.md) (which defines
the *pipeline contract* and on-disk format); this file documents the *sources*.

> **Scope disclaimer.** The public repository ships an example/showcase-scale
> ingestion path, **not** a fully materialized multi-terabyte corpus. The tables
> below describe the sources the bundled fetch scripts pull from
> (`data_pipeline/fetch_large.py`, `data_pipeline/scrape_nvd.py`,
> `scripts/build_tokenizer.py`), not a claim of a trained frontier dataset.

## Sources

All sources are public HuggingFace datasets, streamed at fetch time
(`streaming=True`). The "record cap" is the per-source limit the fetch script
applies, and drives the approximate mixture below.

| Source (HF path) | Role | Record cap | Notes |
|---|---|---:|---|
| `0-hero/OIG-small-chip2` | General instruction / chat | 500,000 | Broad assistant-style dialogue |
| `AlicanKiraz0/All-CVE-Records-Training-Dataset` | Security / vulnerability text | 300,000 | CVE records; also the tokenizer + NVD path source |
| `vanta-llm/corpus` | Cybersecurity domain corpus | 200,000 | Project-curated security text |
| `iamtarun/code_instructions_120k_alpaca` | Code instructions | 120,000 | Alpaca-format code Q&A |

## Approximate mixture (by record cap)

Computed from the caps above (total ≈ 1,120,000 records). This is a *record-count*
mixture, not a token-weighted one — actual token share depends on per-record
length and is not measured here.

| Role | Share |
|---|---:|
| General instruction / chat | ~45% |
| Security / vulnerability (CVE) | ~27% |
| Cybersecurity domain corpus | ~18% |
| Code instructions | ~11% |

## Licensing — operator responsibility

Each source carries its own license on its HuggingFace dataset page. This repo
does **not** re-license or redistribute the underlying data; the fetch scripts
pull it at runtime. **Before any non-research or redistributed use, verify the
current license and terms of each dataset directly on HuggingFace** — dataset
licenses and availability can change, and several community datasets aggregate
upstream sources with their own terms.

What this repository has **not** done (and you should, for a production run):

- Per-source license review and compatibility check for your intended use.
- Dataset revision pinning / checksums for exact reproducibility.
- Train/eval contamination screening.
- PII and safety filtering beyond the pipeline's normalization/dedup steps.

## Preprocessing applied

The bundled pipeline normalizes, deduplicates, tokenizes, and shards raw text
(`normalize_data.py → dedup_merge.py → tokenize_data.py → shard_data.py`); see
[`DATASET_SPEC.md`](./DATASET_SPEC.md) for the enforced invariants. No
source-specific cleaning beyond those shared steps is claimed.
