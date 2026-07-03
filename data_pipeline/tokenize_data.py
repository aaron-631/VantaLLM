"""
tokenize_data.py — VantaLLM Incremental Tokenizer
==================================================
Reads manifest, hashes normalized files, skips matches, tokenizes only new data.
"""

import os
import json
import logging
import hashlib
import torch
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from vantallm.tokenizer.tokenizer import load_tokenizer
from data_pipeline.manifest_utils import (
    STATUS_TOKENIZED_PENDING_SHARD,
    normalize_manifest,
    upsert_tokenized_entry,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger("smart_tokenizer")

NORM_DIR = "data_pipeline/normalized"
TOKEN_DIR = "data_pipeline/tokenized"
MANIFEST_PATH = "data_pipeline/pipeline_manifest.json"
SEQ_LEN = 2048

def get_file_hash(filepath: str) -> str:
    md5 = hashlib.md5()
    with open(filepath, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b""):
            md5.update(chunk)
    return md5.hexdigest()

def load_manifest() -> dict:
    if os.path.exists(MANIFEST_PATH):
        with open(MANIFEST_PATH, 'r') as f:
            return normalize_manifest(json.load(f), token_dir=TOKEN_DIR)
    return normalize_manifest({}, token_dir=TOKEN_DIR)

def save_manifest(manifest: dict):
    with open(MANIFEST_PATH, 'w') as f:
        json.dump(manifest, f, indent=4)

def tokenize_and_pack():
    os.makedirs(TOKEN_DIR, exist_ok=True)
    manifest = load_manifest()
    
    tokenizer = load_tokenizer()
    files = [f for f in os.listdir(NORM_DIR) if f.endswith(".jsonl")]
    
    token_buffer = manifest.get("residual_tokens", [])
    new_files_processed = 0
    
    for filename in files:
        input_path = os.path.join(NORM_DIR, filename)
        file_hash = get_file_hash(input_path)
        tokenized_filename = f"{filename}.pt"
        output_path = os.path.join(TOKEN_DIR, tokenized_filename)
        existing_entry = manifest["processed_tokenized_files"].get(tokenized_filename)
        
        # Check manifest memory
        if manifest["processed_normalized_files"].get(filename) == file_hash:
            if existing_entry is None and os.path.exists(output_path):
                tensor = torch.load(output_path, map_location="cpu", weights_only=False)
                upsert_tokenized_entry(
                    manifest,
                    tokenized_filename,
                    normalized_filename=filename,
                    normalized_hash=file_hash,
                    tokenized_path=output_path,
                    tokenized_exists=True,
                    sequence_count=tensor.shape[0],
                    sequence_length=tensor.shape[1],
                    dtype=str(tensor.dtype),
                    status=STATUS_TOKENIZED_PENDING_SHARD,
                    notes="Recovered tokenized intermediate from disk during skip pass.",
                )
                save_manifest(manifest)
            log.info(f"Skipping {filename} (Already tokenized)")
            continue
            
        log.info(f"Tokenizing new file: {filename}...")
        new_files_processed += 1
        packed_sequences = []
        
        with open(input_path, "r", encoding="utf-8") as f_in:
            for line in f_in:
                try:
                    data = json.loads(line)
                    text = data.get("text", "")
                    tokens = tokenizer.encode(text, return_tensors=None)
                    token_buffer.extend(tokens)
                    
                    while len(token_buffer) >= SEQ_LEN:
                        packed_sequences.append(token_buffer[:SEQ_LEN])
                        token_buffer = token_buffer[SEQ_LEN:]
                except Exception as e:
                    log.warning(f"Tokenization error on record in '{filename}': {e}")
        
        manifest["processed_normalized_files"][filename] = file_hash

        if packed_sequences:
            tensor_data = torch.tensor(packed_sequences, dtype=torch.long)
            torch.save(tensor_data, output_path)
            upsert_tokenized_entry(
                manifest,
                tokenized_filename,
                normalized_filename=filename,
                normalized_hash=file_hash,
                tokenized_path=output_path,
                tokenized_exists=True,
                sequence_count=tensor_data.shape[0],
                sequence_length=tensor_data.shape[1],
                dtype=str(tensor_data.dtype),
                status=STATUS_TOKENIZED_PENDING_SHARD,
                notes=None,
            )
            
            manifest["residual_tokens"] = token_buffer
            save_manifest(manifest)
            
            log.info(f"  Saved {len(packed_sequences)} sequences to {filename}.pt")
        else:
            manifest["residual_tokens"] = token_buffer
            save_manifest(manifest)
            log.info(f"  No full {SEQ_LEN}-token sequences emitted for {filename}; residual tokens carried forward.")

    manifest["residual_tokens"] = token_buffer
    save_manifest(manifest)
    
    if new_files_processed == 0:
        log.info("No new files to tokenize.")

if __name__ == "__main__":
    tokenize_and_pack()
