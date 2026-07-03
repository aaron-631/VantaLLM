"""
normalize_data.py — VantaLLM Incremental Data Normalizer
=========================================================
Standardizes records, trims whitespace, and ensures unicode safety.
Reads the manifest to skip unchanged raw files, preventing redundant I/O overhead.
"""

import os
import json
import logging
import re
import sys
import hashlib

# Add project root to sys.path to allow importing from 'utils'
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from vantallm.utils.prompting import build_prompt
from data_pipeline.manifest_utils import normalize_manifest

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger("smart_normalizer")

RAW_DIR = "data_pipeline/raw"
NORM_DIR = "data_pipeline/normalized"
MANIFEST_PATH = "data_pipeline/pipeline_manifest.json"

def get_file_hash(filepath: str) -> str:
    md5 = hashlib.md5()
    with open(filepath, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b""):
            md5.update(chunk)
    return md5.hexdigest()

def load_manifest() -> dict:
    if os.path.exists(MANIFEST_PATH):
        with open(MANIFEST_PATH, 'r') as f:
            return normalize_manifest(json.load(f))
    return normalize_manifest({})

def save_manifest(manifest: dict):
    with open(MANIFEST_PATH, 'w') as f:
        json.dump(manifest, f, indent=4)

def normalize_text(text: str) -> str:
    """Standardize text while preserving code and technical terms."""
    text = text.strip()
    text = text.replace("\t", "    ")
    text = "".join(c for c in text if c.isprintable() or c in "\n\r\t")
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text

def record_to_text(data: dict) -> str:
    if "text" in data:
        return data["text"]
    if "input" in data and "output" in data:
        inp = data["input"].strip()
        out = data["output"].strip()
        prompt = build_prompt(question=inp)
        return prompt + out
    parts = [str(v) for v in data.values() if isinstance(v, str)]
    return " ".join(parts)

def process():
    os.makedirs(NORM_DIR, exist_ok=True)
    manifest = load_manifest()

    # Ensure the raw file tracker exists in older manifests
    if "processed_raw_files" not in manifest:
        manifest["processed_raw_files"] = {}

    if not os.path.exists(RAW_DIR):
        log.error(f"Raw directory {RAW_DIR} not found.")
        return

    # Look ONLY at files directly inside raw/ (ignoring subdirectories like pretrain/)
    files = [f for f in os.listdir(RAW_DIR) if os.path.isfile(os.path.join(RAW_DIR, f)) and f.endswith(".jsonl")]

    if not files:
        log.warning(f"No .jsonl files found in {RAW_DIR}.")
        return

    processed_any = False

    for filename in files:
        input_path = os.path.join(RAW_DIR, filename)
        raw_hash = get_file_hash(input_path)

        # Check memory ledger: Skip if raw file hasn't changed
        if manifest["processed_raw_files"].get(filename) == raw_hash:
            log.info(f"Skipping {filename} (Already normalized)")
            continue

        log.info(f"Normalizing new/changed file: {filename}...")
        processed_any = True
        output_path = os.path.join(NORM_DIR, filename)

        count = 0
        skipped = 0
        with open(input_path, "r", encoding="utf-8") as f_in, \
             open(output_path, "w", encoding="utf-8") as f_out:

            for line in f_in:
                try:
                    data = json.loads(line)
                    text = record_to_text(data)
                    normalized = normalize_text(text)

                    if normalized:
                        f_out.write(json.dumps({"text": normalized}, ensure_ascii=False) + "\n")
                        count += 1
                    else:
                        skipped += 1
                except Exception as e:
                    skipped += 1

        log.info(f"  {filename}: {count} records normalized, {skipped} skipped.")
        
        # Log success to manifest
        manifest["processed_raw_files"][filename] = raw_hash

    if processed_any:
        save_manifest(manifest)
    else:
        log.info("No new raw files to normalize.")

if __name__ == "__main__":
    process()
