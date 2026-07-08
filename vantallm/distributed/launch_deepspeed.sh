#!/bin/bash
# Minimal DeepSpeed Launcher for VantaLLM
# Single-node usage: ./launch_deepspeed.sh <num_gpus> <config_path>
# Multi-node usage:  ./launch_deepspeed.sh <num_gpus> <config_path> <hostfile>
#
# Multi-node requires:
#   - Passwordless SSH from this (master) node to every node in <hostfile>,
#     including itself (ssh <ip> hostname must return with no prompt).
#   - Identical repo + Python env path on every node (no shared FS: sync by hand).
#   - pdsh installed on the master (with the ssh rcmd module; PDSH_RCMD_TYPE=ssh).
#   - Training env vars (MODEL_CONFIG, BATCH_SIZE, TRAIN_STEPS, ...) present in EACH
#     node's own environment — DeepSpeed only auto-forwards NCCL/master-address vars.
#
# REQUIRED: Export these env vars in your shell BEFORE calling this script:
#   export MODEL_CONFIG=configs/vanta_567m.json   # mandatory unless TINY_MODEL=1
#   export TINY_MODEL=1                            # for CPU/tiny smoke tests only
#   export QUANT_MODE=bf16                         # recommended on modern GPUs
# This script inherits the caller's environment; it does NOT set them automatically.
# See the README and docs/gpu_cluster_guide.md for the full command sequence.
#
# NOTE: Activation checkpointing is model-level (torch.utils.checkpoint), gated by the
# gradient_checkpointing flag in the model config. It is NOT wired through DeepSpeed's
# activation_checkpointing config block. See transformer.py for the flag.

# Default to ZeRO-2 for a smooth single-GPU out-of-box experience: the 567M
# reference fits an A100-class card under ZeRO-2 without CPU offload. ZeRO-3
# (zero3.json) additionally offloads params to CPU — needed only for the large
# analytical configs, and it can OOM/stall a small single-GPU bring-up.
# Pass an explicit second arg to override, e.g.:
#   ./launch_deepspeed.sh 4 vantallm/distributed/deepspeed_configs/zero3.json
NUM_GPUS=${1:-1}
CONFIG_PATH=${2:-"vantallm/distributed/deepspeed_configs/zero2.json"}
HOSTFILE=${3:-}

# Multi-node networking. Overridable from the environment; defaults suit a private
# 10.0.0.0/24 cluster. NCCL_SOCKET_IFNAME must be the NIC carrying the private
# network (check `ip -4 -o addr show`), NOT "lo".
export MASTER_ADDR=${MASTER_ADDR:-10.0.0.2}
export MASTER_PORT=${MASTER_PORT:-29500}
export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-eth0}
export NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-1}
export NCCL_DEBUG=${NCCL_DEBUG:-INFO}

echo "Using config: $CONFIG_PATH"

if [ -n "$HOSTFILE" ]; then
    echo "Launching DeepSpeed multi-node via hostfile: $HOSTFILE"
    echo "  MASTER_ADDR=$MASTER_ADDR MASTER_PORT=$MASTER_PORT NCCL_SOCKET_IFNAME=$NCCL_SOCKET_IFNAME"
    # In hostfile mode DeepSpeed derives the process count from slots= in the
    # hostfile; the NUM_GPUS argument is ignored.
    deepspeed \
        --hostfile="$HOSTFILE" \
        --master_addr="$MASTER_ADDR" \
        --master_port="$MASTER_PORT" \
        vantallm/training/train.py \
        --deepspeed \
        --deepspeed_config="$CONFIG_PATH"
else
    echo "Launching DeepSpeed single-node with $NUM_GPUS GPUs..."
    deepspeed \
        --num_gpus=$NUM_GPUS \
        vantallm/training/train.py \
        --deepspeed \
        --deepspeed_config="$CONFIG_PATH"
fi
