#!/bin/bash
# Minimal DDP Launcher for VantaLM
# Usage: ./launch_ddp.sh <num_gpus>
#
# REQUIRED: Export these env vars in your shell BEFORE calling this script:
#   export MODEL_CONFIG=configs/vanta_567m.json   # mandatory unless TINY_MODEL=1
#   export TINY_MODEL=1                            # for CPU/tiny smoke tests only
#   export QUANT_MODE=bf16                         # recommended on modern GPUs
# This script inherits the caller's environment; it does NOT set them automatically.
# See the README for the full command sequence.

NUM_GPUS=${1:-1}

# MASTER_ADDR defaults to localhost for single-node.
# Override for multi-node: export MASTER_ADDR=<head_node_ip>
export MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
export MASTER_PORT=${MASTER_PORT:-29500}

echo "Launching distributed orchestration validation ($NUM_GPUS ranks)..."
echo "MASTER_ADDR=$MASTER_ADDR  MASTER_PORT=$MASTER_PORT"

TORCHRUN_BIN=${TORCHRUN_BIN:-torchrun}
if [ -x ".venv/bin/torchrun" ]; then
    TORCHRUN_BIN=".venv/bin/torchrun"
fi

"$TORCHRUN_BIN" \
    --nproc_per_node=$NUM_GPUS \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    training/train.py
