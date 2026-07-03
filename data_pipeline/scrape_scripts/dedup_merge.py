"""
dedup_merge.py — VantaLM Global Delta Extractor
=================================================
Scans raw/pretrain/, checks a global memory hash, and outputs ONLY 
brand-new unique documents as a timestamped delta batch.
"""

import os
import json
import logging
import hashlib
from datetime import datetime

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger("delta_extractor")

RAW_DIR = "data_pipeline/raw/pretrain"
NORM_DIR = "data_pipeline/normalized"
GLOBAL_HASH_LOG = "data_pipeline/global_hash_log.json"

def get_hash(content: str) -> str:
    return hashlib.md5(content.encode('utf-8')).hexdigest()

def extract_deltas():
    os.makedirs(NORM_DIR, exist_ok=True)
    
    # Load global memory
    seen_hashes = set()
    if os.path.exists(GLOBAL_HASH_LOG):
        with open(GLOBAL_HASH_LOG, 'r') as f:
            seen_hashes = set(json.load(f))
            
    log.info(f"Loaded {len(seen_hashes)} previously seen documents from memory.")

    files = [f for f in os.listdir(RAW_DIR) if f.endswith(".jsonl")]
    new_records = []
    duplicates_dropped = 0
    
    for filename in files:
        log.info(f"Scanning {filename}...")
        path = os.path.join(RAW_DIR, filename)
        
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    data = json.loads(line)
                    text = data.get("text", "").strip()
                    if not text: continue
                    
                    # Deduplicate based on exact text
                    doc_hash = get_hash(text)
                    if doc_hash in seen_hashes:
                        duplicates_dropped += 1
                        continue
                        
                    seen_hashes.add(doc_hash)
                    new_records.append(data)
                except Exception as e:
                    log.warning(f"Skipping malformed line in {filename}")

    # Output Delta Batch
    if new_records:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        delta_filename = f"delta_batch_{timestamp}.jsonl"
        delta_path = os.path.join(NORM_DIR, delta_filename)
        
        with open(delta_path, "w", encoding="utf-8") as f:
            for rec in new_records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                
        # Save updated memory
        with open(GLOBAL_HASH_LOG, "w") as f:
            json.dump(list(seen_hashes), f)
            
        log.info("========================================")
        log.info(f"[SUCCESS] Delta Batch Generated: {delta_filename}")
        log.info(f"New Unique Documents: {len(new_records)}")
        log.info(f"Duplicates Dropped: {duplicates_dropped}")
        log.info("========================================")
    else:
        log.info("No new unique documents found. System is up to date.")

if __name__ == "__main__":
    extract_deltas()