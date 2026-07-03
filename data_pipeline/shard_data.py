"""
shard_data.py — VantaLM Append-Only Sharder
=============================================
Loads only un-sharded tensors, appends them starting from last_shard_index.
"""

import os
import torch
import logging
import json
import sys
from collections import defaultdict

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data_pipeline.manifest_utils import (
    STATUS_TOKENIZED_PENDING_SHARD,
    mark_entry_sharded,
    normalize_manifest,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger("append_sharder")

TOKEN_DIR = "data_pipeline/tokenized"
SHARD_DIR = "data_pipeline/shards"
MANIFEST_PATH = "data_pipeline/pipeline_manifest.json"
SHARD_SIZE_MB = 256

def load_manifest() -> dict:
    with open(MANIFEST_PATH, 'r') as f:
        return normalize_manifest(json.load(f), token_dir=TOKEN_DIR)

def save_manifest(manifest: dict):
    with open(MANIFEST_PATH, 'w') as f:
        json.dump(manifest, f, indent=4)

def shard_incremental():
    os.makedirs(SHARD_DIR, exist_ok=True)
    manifest = load_manifest()
    
    # Find files that haven't been sharded yet
    unsharded_files = [
        f
        for f, entry in manifest.get("processed_tokenized_files", {}).items()
        if entry.get("status") == STATUS_TOKENIZED_PENDING_SHARD
        and os.path.exists(os.path.join(TOKEN_DIR, f))
    ]
    
    if not unsharded_files:
        log.info("No new tokens to shard. System up to date.")
        return

    log.info(f"Found {len(unsharded_files)} new .pt files to shard.")
    
    shard_idx = manifest.get("last_shard_index", 0)

    shard_buffer = []
    shard_sources = []
    buffer_count = 0
    samples_per_shard = None
    created_shards = []
    shards_by_source = defaultdict(list)

    def flush_shard():
        nonlocal shard_idx, shard_buffer, shard_sources, buffer_count
        if not shard_buffer:
            return

        shard_idx += 1
        shard_tensor = torch.cat(shard_buffer, dim=0)
        shard_filename = f"train_{shard_idx:04d}.bin"
        shard_path = os.path.join(SHARD_DIR, shard_filename)

        torch.save(shard_tensor.clone(), shard_path)
        size_mb = os.path.getsize(shard_path) / (1024 * 1024)
        log.info(f"  Saved NEW shard: {shard_filename} ({size_mb:.2f} MB, {shard_tensor.shape[0]:,} samples)")

        source_counts = defaultdict(int)
        for source_filename, seq_count in shard_sources:
            source_counts[source_filename] += seq_count
            shards_by_source[source_filename].append(shard_idx)

        manifest["shard_history"][shard_filename] = {
            "index": shard_idx,
            "path": shard_path,
            "num_sequences": int(shard_tensor.shape[0]),
            "sequence_length": int(shard_tensor.shape[1]),
            "dtype": str(shard_tensor.dtype),
            "source_tokenized_files": dict(source_counts),
        }
        created_shards.append(shard_filename)
        shard_buffer = []
        shard_sources = []
        buffer_count = 0

    for filename in unsharded_files:
        path = os.path.join(TOKEN_DIR, filename)
        log.info(f"  Loading delta tensor: {filename}")
        tensor = torch.load(path, map_location="cpu", weights_only=False)
        entry = manifest["processed_tokenized_files"][filename]
        entry["tokenized_exists"] = True
        entry["sequence_count"] = int(tensor.shape[0])
        entry["sequence_length"] = int(tensor.shape[1])
        entry["dtype"] = str(tensor.dtype)

        if samples_per_shard is None:
            seq_len = tensor.shape[1]
            item_size = max(1, tensor.element_size())
            samples_per_shard = max(1, int((SHARD_SIZE_MB * 1024 * 1024) / (seq_len * item_size)))

        cursor = 0
        while cursor < tensor.shape[0]:
            remaining_capacity = samples_per_shard - buffer_count
            take = min(remaining_capacity, tensor.shape[0] - cursor)
            shard_buffer.append(tensor[cursor:cursor + take])
            shard_sources.append((filename, int(take)))
            buffer_count += take
            cursor += take

            if buffer_count >= samples_per_shard:
                flush_shard()

    flush_shard()

    # Update manifest state
    for filename in unsharded_files:
        mark_entry_sharded(manifest, filename, shards_by_source.get(filename, []))
    manifest["last_shard_index"] = shard_idx
    
    save_manifest(manifest)
    log.info(f"[SUCCESS] Incremental sharding complete. Created {len(created_shards)} shard(s).")

if __name__ == "__main__":
    shard_incremental()
