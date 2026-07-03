import os
import json
import hashlib
import glob
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data_pipeline.manifest_utils import (
    STATUS_RECOVERED_MISSING_INTERMEDIATE,
    STATUS_TOKENIZED_PENDING_SHARD,
    normalize_manifest,
    upsert_tokenized_entry,
)

MANIFEST_PATH = "data_pipeline/pipeline_manifest.json"
SHARD_DIR = "data_pipeline/shards"
RAW_DIR = "data_pipeline/raw"
NORM_DIR = "data_pipeline/normalized"
TOKEN_DIR = "data_pipeline/tokenized"

def get_file_hash(filepath: str) -> str:
    md5 = hashlib.md5()
    try:
        with open(filepath, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                md5.update(chunk)
        return md5.hexdigest()
    except Exception:
        return ""

def recover_manifest():
    print("🚀 Starting Complete State Recovery...")
    existing_manifest = {}
    if os.path.exists(MANIFEST_PATH):
        with open(MANIFEST_PATH, "r") as f:
            existing_manifest = json.load(f)
    manifest = normalize_manifest(existing_manifest, token_dir=TOKEN_DIR)
    
    # 1. Safely detect highest shard index
    shard_files = glob.glob(os.path.join(SHARD_DIR, "train_*.bin"))
    max_index = 0
    for f in shard_files:
        try:
            idx = int(os.path.basename(f).replace('train_', '').replace('.bin', ''))
            max_index = max(max_index, idx)
        except ValueError: pass
    print(f"📊 Auto-detected highest shard index: {max_index}")

    # 2. Hash Raw Files (Protects Normalizer)
    processed_raw = {}
    if os.path.exists(RAW_DIR):
        for f in os.listdir(RAW_DIR):
            if f.endswith(".jsonl"):
                processed_raw[f] = get_file_hash(os.path.join(RAW_DIR, f))
    print(f"🔒 Hashed {len(processed_raw)} raw files to skip future normalization.")

    # 3. Hash Normalized Files (Protects Tokenizer)
    processed_norm = {}
    if os.path.exists(NORM_DIR):
        for f in os.listdir(NORM_DIR):
            if f.endswith(".jsonl"):
                processed_norm[f] = get_file_hash(os.path.join(NORM_DIR, f))
    print(f"🔒 Hashed {len(processed_norm)} normalized files to skip future tokenization.")

    # 4. Mark remaining tokenized files as safe
    processed_tokenized = manifest.get("processed_tokenized_files", {})
    if os.path.exists(TOKEN_DIR):
        for f in os.listdir(TOKEN_DIR):
            if f.endswith(".pt"):
                path = os.path.join(TOKEN_DIR, f)
                try:
                    tensor = None
                    try:
                        import torch
                        tensor = torch.load(path, map_location="cpu", weights_only=False)
                    except Exception:
                        tensor = None

                    upsert_tokenized_entry(
                        manifest,
                        f,
                        normalized_filename=f[:-3],
                        normalized_hash=processed_norm.get(f[:-3]),
                        tokenized_path=path,
                        tokenized_exists=True,
                        sequence_count=int(tensor.shape[0]) if tensor is not None else None,
                        sequence_length=int(tensor.shape[1]) if tensor is not None else None,
                        dtype=str(tensor.dtype) if tensor is not None else None,
                        status=processed_tokenized.get(f, {}).get("status", STATUS_TOKENIZED_PENDING_SHARD),
                        notes=processed_tokenized.get(f, {}).get("notes"),
                    )
                except Exception:
                    upsert_tokenized_entry(
                        manifest,
                        f,
                        normalized_filename=f[:-3],
                        normalized_hash=processed_norm.get(f[:-3]),
                        tokenized_path=path,
                        tokenized_exists=True,
                        status=processed_tokenized.get(f, {}).get("status", STATUS_TOKENIZED_PENDING_SHARD),
                        notes="Recovered tokenized artifact without tensor metadata.",
                    )

    # Preserve history even when .pt intermediates have been cleaned up.
    for normalized_filename, normalized_hash in processed_norm.items():
        tokenized_filename = f"{normalized_filename}.pt"
        if tokenized_filename not in manifest["processed_tokenized_files"]:
            upsert_tokenized_entry(
                manifest,
                tokenized_filename,
                normalized_filename=normalized_filename,
                normalized_hash=normalized_hash,
                tokenized_path=os.path.join(TOKEN_DIR, tokenized_filename),
                tokenized_exists=False,
                status=STATUS_RECOVERED_MISSING_INTERMEDIATE,
                notes="Recovered from normalized/shard state; intermediate tokenized artifact is not present on disk.",
            )

    manifest["processed_raw_files"] = processed_raw
    manifest["processed_normalized_files"] = processed_norm
    manifest["last_shard_index"] = max_index
    manifest["residual_tokens"] = manifest.get("residual_tokens", [])

    with open(MANIFEST_PATH, 'w') as f:
        json.dump(manifest, f, indent=4)
    
    print("✅ Successfully rebuilt pipeline_manifest.json with full hash protection!")

if __name__ == "__main__":
    recover_manifest()
