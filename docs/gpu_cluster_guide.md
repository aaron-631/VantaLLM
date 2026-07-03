# GPU Cluster Guide

This guide details the procedures for hardware validation, distributed training execution, and multi-GPU cluster usage for VantaLLM.

## Hardware Validation

Before initiating large-scale training, validate the CUDA execution environment and memory allocation compatibility. 

### Single-GPU Validation
1. Verify CUDA availability and memory capacity using `nvidia-smi`.
2. Ensure Bfloat16 support is enabled for NVIDIA Ampere (or newer) architectures.
3. Run a minimal smoke test (e.g., using `TINY_MODEL=1`) to confirm the training loop runs successfully and checkpoint saving mechanisms are functional.

## Distributed Launch Instructions

VantaLLM supports PyTorch DDP and DeepSpeed ZeRO out of the box.

### PyTorch DDP
Launch multi-GPU training locally using `torchrun`:
```bash
torchrun --nproc_per_node=4 vantallm/training/train.py
```
This is ideal for single-node setups where model states fit into GPU memory without partitioning.

### DeepSpeed ZeRO
For memory-constrained setups or multi-node clusters, launch DeepSpeed:
```bash
deepspeed --num_gpus=4 vantallm/training/train.py --deepspeed deepspeed_config.json
```
DeepSpeed ZeRO-1 and ZeRO-2 will partition optimizer states and gradients across available ranks, freeing up significant VRAM for larger batch sizes.

## NCCL Configuration
When running on distributed clusters, ensure NCCL is configured correctly to prevent deadlocks:
- Set `NCCL_DEBUG=INFO` for verbosity during initial testing.
- Verify that InfiniBand (IB) or RoCE interfaces are selected properly via `NCCL_IB_DISABLE=0` and `NCCL_SOCKET_IFNAME`.

## Troubleshooting

- **OOM (Out Of Memory):** Reduce `BATCH_SIZE` or enable DeepSpeed ZeRO-offload to CPU.
- **Deadlocks during DDP:** Verify all processes are reading from the same data shards and that network topologies allow peer-to-peer communication between GPUs.
