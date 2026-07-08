#!/bin/bash
# ============================================================================
# launch_vanta_4gpu.sh — locked, reproducible 32-step bring-up launcher
#   Target: 4× large-VRAM GPUs (e.g. RTX PRO 6000 Blackwell 95 GB) + high host RAM.
#   Model : full 125B (configs/vanta_125b_bringup.json — NO layer/width/expert cuts).
#
# ── RUN INSIDE tmux (SSH disconnect will kill the 30-90 min init otherwise) ──
#     tmux new -s train
#     ./scripts/launch_vanta_4gpu.sh 2>&1 | tee train.log
#     # detach: Ctrl-b d   reattach: tmux attach -t train
#
# ── MEMORY MONITOR (run in a second tmux pane) ──
#     watch -n5 'free -h; echo; nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv'
#
# First prove the pipeline on the SMALL config (minutes), THEN the 125B:
#     MODEL_CONFIG=configs/vanta_567m.json ./scripts/launch_vanta_4gpu.sh   # Stage B smoke
#     ./scripts/launch_vanta_4gpu.sh                                        # Stage C (125B)
#
# OPTIMIZER MEMORY NOTE: zero3.json offloads PARAMS ONLY. The optimizer is
# bitsandbytes PagedAdamW8bit (8-bit states) which does its own CPU paging.
# Do NOT add offload_optimizer to the DeepSpeed config: it would either hand bnb's
# CUDA step CPU tensors (crash) or force fp32 DeepSpeedCPUAdam (~1.5 TB at 125B → host OOM).
# ============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

# --- Tunables (override via env) ---
NUM_GPUS="${NUM_GPUS:-4}"
DS_CONFIG="${DS_CONFIG:-vantallm/distributed/deepspeed_configs/zero3.json}"
export MODEL_CONFIG="${MODEL_CONFIG:-configs/vanta_125b_bringup.json}"
export TRAIN_STEPS="${TRAIN_STEPS:-32}"
export BATCH_SIZE="${BATCH_SIZE:-2}"      # per-GPU micro-batch; must match train_micro_batch_size_per_gpu
export QUANT_MODE="${QUANT_MODE:-bf16}"
export DATA_MODE="${DATA_MODE:-shard}"
export SHARD_DIR="${SHARD_DIR:-data_pipeline/shards}"
export TINY_MODEL=0

# --- Disable validation loop for the smoke run ---
export VAL_PATH="${VAL_PATH:-}"
export VAL_EVERY="${VAL_EVERY:-999999}"

# --- Distributed networking (single-node loopback) ---
export MASTER_ADDR="${MASTER_ADDR:-localhost}"
export MASTER_PORT="${MASTER_PORT:-29500}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-lo}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export PYTHONUNBUFFERED=1

# --- Validate the JSON before spending 30-90 min on init ---
if command -v jq >/dev/null 2>&1; then
  echo "[launch] Validating $DS_CONFIG ..."; jq . "$DS_CONFIG" >/dev/null || { echo "[launch] ERROR: $DS_CONFIG invalid JSON"; exit 1; }
  echo "[launch] Validating $MODEL_CONFIG ..."; jq . "$MODEL_CONFIG" >/dev/null || { echo "[launch] ERROR: $MODEL_CONFIG invalid JSON"; exit 1; }
else
  echo "[launch] WARNING: jq not found — skipping JSON validation"
fi

cat <<EOF
[launch] ---------------------------------------------------------------
[launch]  GPUs           : $NUM_GPUS
[launch]  DeepSpeed cfg  : $DS_CONFIG
[launch]  Model config   : $MODEL_CONFIG
[launch]  Steps          : $TRAIN_STEPS
[launch]  Micro-batch/GPU: $BATCH_SIZE   (must == train_micro_batch_size_per_gpu in DS cfg)
[launch]  Data mode      : $DATA_MODE  ($SHARD_DIR)
[launch] ---------------------------------------------------------------
EOF

exec deepspeed --num_gpus="$NUM_GPUS" \
    vantallm/training/train.py \
    --deepspeed \
    --deepspeed_config="$DS_CONFIG"
