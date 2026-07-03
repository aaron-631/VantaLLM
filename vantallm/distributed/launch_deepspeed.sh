#!/bin/bash
# Minimal DeepSpeed Launcher for VantaLLM
# Usage: ./launch_deepspeed.sh <num_gpus> <config_path>
#
# REQUIRED: Export these env vars in your shell BEFORE calling this script:
#   export MODEL_CONFIG=configs/vanta_567m.json   # mandatory unless TINY_MODEL=1
#   export TINY_MODEL=1                            # for CPU/tiny smoke tests only
#   export QUANT_MODE=bf16                         # recommended on modern GPUs
# This script inherits the caller's environment; it does NOT set them automatically.
# See the README for the full command sequence.
#
# NOTE: Activation checkpointing is NOT wired in these configs. To enable it at
# larger scales, use deepspeed.checkpointing.configure() in the model and add
# the activation_checkpointing block back to the config. See transformer.py for the
# gradient_checkpointing flag (model-level, uses torch.utils.checkpoint).

NUM_GPUS=${1:-1}
CONFIG_PATH=${2:-"vantallm/distributed/deepspeed_configs/zero3.json"}

echo "Launching DeepSpeed with $NUM_GPUS GPUs..."
echo "Using config: $CONFIG_PATH"

# Pass --deepspeed flag to train.py when ready
deepspeed \
    --num_gpus=$NUM_GPUS \
    vantallm/training/train.py \
    --deepspeed \
    --deepspeed_config=$CONFIG_PATH
