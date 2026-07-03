"""
scrape_nvd.py — VantaLLM NVD/CVE Data Ingestor
===============================================
Fetches CVE descriptions from a stable mirror (HuggingFace) to 
provide a foundational cybersecurity vocabulary layer.
"""

import os
import json
import logging
from datasets import load_dataset

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger("nvd_ingestor")

RAW_DIR = "data_pipeline/raw"

def fetch_nvd():
    os.makedirs(RAW_DIR, exist_ok=True)
    output_file = os.path.join(RAW_DIR, "nvd_cve.jsonl")
    
    log.info("Fetching CVE records from stable mirror (AlicanKiraz0/All-CVE-Records-Training-Dataset)...")
    
    try:
        # This dataset contains ~300k CVE records. 
        # Streaming=True for memory efficiency.
        ds = load_dataset("AlicanKiraz0/All-CVE-Records-Training-Dataset", split="train", streaming=True)
        
        count = 0
        with open(output_file, "w", encoding="utf-8") as f:
            for sample in ds:
                # Discovered Schema: ['System', 'User', 'Assistant']
                # The 'Assistant' field contains the detailed technical analysis and description.
                content = sample.get('Assistant', '')
                
                if content and len(content.strip()) > 50:
                    f.write(json.dumps({"text": content.strip()}, ensure_ascii=False) + "\n")
                    count += 1
                
                if count > 0 and count % 10000 == 0:
                    log.info(f"  {count} CVE records processed...")
                
                # Target: ~100k-200k records for the 100MB-500MB target layer
                if count >= 150000:
                    break
        
        log.info(f"Successfully saved {count} CVE records to {output_file}")
        
    except Exception as e:
        log.error(f"Failed to fetch NVD data: {e}")

if __name__ == "__main__":
    fetch_nvd()
