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
deepspeed --num_gpus=4 vantallm/training/train.py --deepspeed vantallm/distributed/deepspeed_configs/zero2.json
```
DeepSpeed ZeRO-1 and ZeRO-2 will partition optimizer states and gradients across available ranks, freeing up significant VRAM for larger batch sizes.

## NCCL Configuration
When running on distributed clusters, ensure NCCL is configured correctly to prevent deadlocks:
- Set `NCCL_DEBUG=INFO` for verbosity during initial testing.
- Verify that InfiniBand (IB) or RoCE interfaces are selected properly via `NCCL_IB_DISABLE=0` and `NCCL_SOCKET_IFNAME`.

## Troubleshooting

- **OOM (Out Of Memory):** Reduce `BATCH_SIZE` or enable DeepSpeed ZeRO-offload to CPU.
- **Deadlocks during DDP:** Verify all processes are reading from the same data shards and that network topologies allow peer-to-peer communication between GPUs. If testing with very small datasets, ensure ranks do not exhaust data unevenly, as this causes a distributed deadlock.

## DeepSpeed ZeRO-3 Initialisation (CPU Offload)
When utilizing ZeRO-3 with CPU offloading for large models (e.g., 125B parameter profiles):
- **Initialization Silence:** `deepspeed.initialize()` must instantiate the full model on the CPU, partition it, allocate offloaded optimizer states, and move initial parameter shards to the GPUs. This CPU-bound phase can take **30 to 90 minutes of complete silence**.
- **Use Multiplexers:** You MUST use a terminal multiplexer (`tmux` or `screen`). If you run the command directly over SSH, the session will time out during the silent graph construction, killing the process (`SIGHUP`).
- **Dependencies:** Ensure `libaio-devel` is installed on your Linux system. It is strictly required for asynchronous I/O during ZeRO CPU offloading.
- **LR Scheduler Compatibility:** PyTorch LR schedulers fail the `isinstance` check against `DeepSpeedZeroOptimizer_Stage3`. The training script automatically extracts the base optimizer (`optimizer.optimizer`) to prevent crashes.

## A100 20GB Configuration Notes
The 567M model reference configuration (`vanta_567m.json`) is designed to easily fit within a 20GB VRAM footprint using the default micro-batch size and gradient accumulation settings. 

The validation and scaling advice provided in this guide applies equally to 20GB, 40GB, and 80GB A100 environments. For 20GB architectures on the AI Koch cluster, you can reliably use standard BF16 mixed-precision and DeepSpeed ZeRO-1 without encountering Out-Of-Memory (OOM) errors during the backward pass.

The same distributed topology assumptions also carry forward to Blackwell-class multi-node deployments. Revalidate kernel, NCCL, and micro-batch settings on the target cluster, but the package layout and launcher/config separation here are designed to remain forward-compatible.
