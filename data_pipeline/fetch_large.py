"""
fetch_large.py — VantaLM Large-Scale Data Fetcher
==================================================
Fetches from 4 HuggingFace datasets using streaming=True.
Writes to data_pipeline/raw/ in {"text": "..."} format.
"""

import os
import json
import logging
from datasets import load_dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("fetcher_large")

RAW_DIR = "data_pipeline/raw"

SOURCES = [
    {
        "name": "alpaca_code_full",
        "path": "iamtarun/code_instructions_120k_alpaca",
        "split": "train",
        "cap": 120000,
        "formatter": lambda x: f"Instruction: {x['instruction']}\nInput: {x['input']}\nResponse: {x['output']}" if x.get('input') else f"Instruction: {x['instruction']}\nResponse: {x['output']}"
    },
    {
        "name": "oig_large",
        "path": "0-hero/OIG-small-chip2",
        "split": "train",
        "cap": 500000,
        "formatter": lambda x: f"User: {x['user']}'\nAssistant: {x['chip2']}"
    },
    {
        "name": "cybersec_vanta-corpus_v2",
        "path": "vanta-llm/corpus",
        "split": "train",
        "cap": 200000,
        "formatter": lambda x: f"Content: {x.get('assistant', x.get('Assistant', ''))}"
    },
    {
        "name": "cve_records",
        "path": "AlicanKiraz0/All-CVE-Records-Training-Dataset",
        "split": "train",
        "cap": 300000,
        "formatter": lambda x: f"{x.get('text', x.get('description', str(x)))}"
    }
]

def fetch():
    os.makedirs(RAW_DIR, exist_ok=True)

    for src in SOURCES:
        name = src["name"]
        path = src["path"]
        split = src["split"]
        cap = src["cap"]
        formatter = src["formatter"]

        output_file = os.path.join(RAW_DIR, f"{name}.jsonl")
        log.info(f"Starting fetch for {name} from {path}...")

        count = 0
        try:
            # Using streaming=True to minimize memory/disk footprint during fetch
            ds = load_dataset(path, split=split, streaming=True)
            
            with open(output_file, "w", encoding="utf-8") as f:
                for record in ds:
                    text = formatter(record)
                    
                    if len(text.strip()) >= 30:
                        f.write(json.dumps({"text": text.strip()}, ensure_ascii=False) + "\n")
                        count += 1
                        
                        if count % 10000 == 0:
                            log.info(f"  {name}: {count} records saved...")
                        
                        if count >= cap:
                            break
            
            log.info(f"Completed {name}: {count} total records saved.")
            
        except Exception as e:
            log.error(f"Failed to fetch {name}: {e}")

if __name__ == "__main__":
    fetch()
