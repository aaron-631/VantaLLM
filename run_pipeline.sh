#!/bin/bash
set -euo pipefail

echo "Starting VantaLM incremental pipeline..."

# 1. Process standard open-source files in base raw/
python3 data_pipeline/normalize_data.py

# 2. Process custom threat intel in raw/pretrain/
python3 data_pipeline/scrape_scripts/dedup_merge.py

# 3. Tokenize only new/changed files
python3 data_pipeline/tokenize_data.py

# 4. Append to binary shards
python3 data_pipeline/shard_data.py

echo "Pipeline execution finished."
