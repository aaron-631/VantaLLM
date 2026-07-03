# VantaLLM Dataset Specification

This document defines the data contract used by the VantaLLM pipeline. It is not a claim that the full target corpus is already packaged in this repository; it describes the intended manufacturing layout and the invariants that the pipeline tools enforce.

## Overview

VantaLLM is designed to support large-scale decoder-only pretraining with a mix of general-language, code, and security-relevant text. The repository focuses on the engineering path required to ingest, normalize, tokenize, deduplicate, and shard those corpora reproducibly.

## Terminology

The pipeline uses two distinct terms deliberately:

- **Slice:** A human-readable `.jsonl` subset used for inspection, debugging, and early smoke tests.
- **Shard:** A packed binary `.bin` tensor containing tokenized sequences used for sustained training runs.

Slices are for operator visibility. Shards are for throughput.

## Pipeline Contract

The intended pipeline flow is:

```text
raw jsonl/text
  -> normalize_data.py
  -> dedup_merge.py
  -> tokenize_data.py
  -> shard_data.py
  -> data_pipeline/shards/train_XXXX.bin
```

The repository treats this as an append-only workflow:

- `pipeline_manifest.json` tracks per-file lifecycle state.
- `global_hash_log.json` protects cross-run deduplication.
- `shard_data.py` appends new shard IDs instead of rewriting historical ones.
- `recover_state.py` can rebuild manifest state from surviving artifacts if needed.

## Expected Corpus Mix

The codebase is compatible with a broad pretraining mix such as:

- General web text
- General code corpora
- Security research and vulnerability text
- Technical documentation
- Small curated synthetic or evaluation slices for smoke testing

The public repository includes example and showcase-scale data assets for validation, not a fully materialized multi-terabyte corpus.

## Output Format

Training shards are expected to be saved as PyTorch tensors:

```python
# Example shard
# path: data_pipeline/shards/train_0001.bin
# type: torch.Tensor
# dtype: int64
# shape: [num_sequences, seq_len]
```

The training loop and `verify_gpu_ready.py` expect shard tensors to contain integer token IDs with a stable sequence length.

## Tokenizer Contract

The active tokenizer contract used by the repository is:

- Vocabulary size: `151003`
- `<|user|>`: `151000`
- `<|assistant|>`: `151001`
- `<|system|>`: `151002`

Any dataset material prepared for chat-style supervised experiments must preserve these token IDs.

## Operational Notes

- Do not edit `pipeline_manifest.json` or `global_hash_log.json` by hand during normal operation.
- Do not rename shard files manually after creation.
- Validate new shards with `python3 verify_gpu_ready.py` before using `DATA_MODE=shard`.
- For first-pass debugging, prefer a small slice and `DATA_MODE=showcase` before escalating to shard-backed training.
