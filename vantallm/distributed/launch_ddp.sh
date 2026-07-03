#!/bin/bash
# Minimal DDP Launcher for VantaLLM
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

# MASTER_PORT: honor an explicitly-exported value (required for multi-node so all
# ranks rendezvous on the same port). If unset, pick a free ephemeral port instead
# of a fixed 29500 — a hardcoded default hits "address already in use" on back-to-
# back local runs or a stale socket. The rendezvous port does not affect training
# results, so autodetecting it is safe and smooths the out-of-box single-node run.
if [ -z "${MASTER_PORT}" ]; then
    _PYBIN=""
    if [ -x ".venv/bin/python" ]; then _PYBIN=".venv/bin/python"
    elif command -v python3 >/dev/null 2>&1; then _PYBIN="python3"
    elif command -v python  >/dev/null 2>&1; then _PYBIN="python"
    fi
    if [ -n "$_PYBIN" ]; then
        MASTER_PORT=$("$_PYBIN" -c 'import socket; s=socket.socket(); s.bind(("",0)); print(s.getsockname()[1]); s.close()' 2>/dev/null || echo 29500)
    else
        MASTER_PORT=29500
    fi
fi
export MASTER_PORT

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
    vantallm/training/train.py
