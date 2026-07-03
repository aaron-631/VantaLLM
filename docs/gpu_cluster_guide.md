# GPU Cluster Guide

This guide details the procedures for hardware validation, distributed training execution, and multi-GPU cluster usage for VantaLLM.

## Preflight Environment

Before launching any expensive run, make sure the shell session is explicit and reproducible:

```bash
source .venv/bin/activate
set -a
[ -f .env ] && source .env
set +a

export QUANT_MODE=bf16
export MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
export MASTER_PORT=${MASTER_PORT:-29500}
export NCCL_DEBUG=${NCCL_DEBUG:-INFO}
export PYTHONUNBUFFERED=1
```

If you are starting a long DDP or DeepSpeed run, use `tmux` or `screen` first.

For GPU-backed DeepSpeed runs, make sure the full NVIDIA CUDA Toolkit is installed so `nvcc` is available, and ensure both `$PATH` and `$LD_LIBRARY_PATH` expose the active CUDA install. DeepSpeed extension builds are sensitive to version skew, so keep the installed DeepSpeed package aligned with the PyTorch CUDA wheel and toolkit version on the host.

## Hardware Validation

Before initiating large-scale training, validate the CUDA execution environment and memory allocation compatibility. 

### Single-GPU Validation
1. Verify CUDA availability and memory capacity using `nvidia-smi`.
2. Ensure Bfloat16 support is enabled for NVIDIA Ampere (or newer) architectures.
3. Run a minimal smoke test (e.g., using `TINY_MODEL=1`) to confirm the training loop runs successfully and checkpoint saving mechanisms are functional.

## Recommended Bring-Up Order

The cleanest staged path is:

1. `python3 verify_distributed.py`
2. `torchrun --nproc_per_node=<gpus> verify_distributed.py`
3. `TRAIN_STEPS=10 TINY_MODEL=1 DATA_MODE=showcase DATA_PATH=data_pipeline/rag_synthetic_slice_v2_fixed.jsonl python3 vantallm/training/train.py`
4. `MODEL_CONFIG=configs/vanta_567m.json TRAIN_STEPS=1000 DATA_MODE=shard bash vantallm/distributed/launch_ddp.sh <gpus>`
5. `MODEL_CONFIG=configs/vanta_567m.json TRAIN_STEPS=1000 DATA_MODE=shard bash vantallm/distributed/launch_deepspeed.sh <gpus>`

The important rule is to validate topology first, then local training, then shard-backed training, and only then escalate to multi-rank orchestration.

## Bring-Up Discipline

Treat early GPU work as controlled validation, not as a throughput race:

1. Change one variable at a time. Do not alter model config, batch size, ZeRO stage, topology, and environment variables simultaneously.
2. Use bounded runs first. On a new GPU type, node image, or NCCL topology, start with `TRAIN_STEPS=2` so hangs, OOMs, and allocator regressions fail quickly and readably.
3. Record an environment snapshot for expensive runs. At minimum, save `nvidia-smi`, `python3 -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.nccl.version())"`, and the exact `MODEL_CONFIG` / DeepSpeed JSON used for the launch.

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

Repo-native launchers are also provided:

```bash
bash vantallm/distributed/launch_ddp.sh 4
bash vantallm/distributed/launch_deepspeed.sh 4
```

## NCCL Configuration
When running on distributed clusters, ensure NCCL is configured correctly to prevent deadlocks:
- Set `NCCL_DEBUG=INFO` for verbosity during initial testing.
- Verify that InfiniBand (IB) or RoCE interfaces are selected properly via `NCCL_IB_DISABLE=0` and `NCCL_SOCKET_IFNAME`.

## Troubleshooting

- **OOM (Out Of Memory):** Reduce `BATCH_SIZE` or enable DeepSpeed ZeRO-offload to CPU.
- **Deadlocks during DDP:** Verify all processes are reading from the same data shards and that network topologies allow peer-to-peer communication between GPUs. If testing with very small datasets, ensure ranks do not exhaust data unevenly, as this causes a distributed deadlock.
- **No shards found:** Run `./run_pipeline.sh` or `python3 data_pipeline/shard_data.py` before using `DATA_MODE=shard`.
- **Immediate trainer exit:** `TRAIN_STEPS` is mandatory. The training script intentionally refuses to infer a default target step count.

## DeepSpeed ZeRO-3 Initialisation (CPU Offload)
When utilizing ZeRO-3 with CPU offloading for large models (e.g., 125B parameter profiles):
- **Initialization Silence:** `deepspeed.initialize()` must instantiate the full model on the CPU, partition it, allocate offloaded optimizer states, and move initial parameter shards to the GPUs. This CPU-bound phase can take **30 to 90 minutes of complete silence**.
- **Use Multiplexers:** You MUST use a terminal multiplexer (`tmux` or `screen`). If you run the command directly over SSH, the session will time out during the silent graph construction, killing the process (`SIGHUP`).
- **Dependencies:** Ensure `python3-devel`, `gcc-c++`, `make`, `git`, and `libaio-devel` are installed on your Linux system. They are strictly required for compiling C++ DeepSpeed extensions and for asynchronous I/O during ZeRO CPU offloading.
- **LR Scheduler Compatibility:** PyTorch LR schedulers fail the `isinstance` check against `DeepSpeedZeroOptimizer_Stage3`. The training script automatically extracts the base optimizer (`optimizer.optimizer`) to prevent crashes.
- **125B Swapfile Requirements:** The 125B model combined with an 8-bit partitioned optimizer requires exactly 1,500 GB of virtual memory. For single-node runs on a standard 692GB node, an **850GB+ NVMe swapfile is absolutely mandatory** to prevent OOM kills during paging.
- **Optimizer Selection:** For the 125B run, you must use `bitsandbytes.optim.PagedAdamW8bit` to fit within memory constraints.

## A100 20GB Configuration Notes
The 567M model reference configuration (`vanta_567m.json`) is designed to easily fit within a 20GB VRAM footprint using the default micro-batch size and gradient accumulation settings. 

The validation and scaling advice provided in this guide applies equally to 20GB, 40GB, and 80GB A100 environments. For 20GB-class A100 setups, the validated reference path uses standard BF16 mixed precision and DeepSpeed ZeRO-1 without requiring branch-specific code changes.

### Hardware Tuning Knobs

When moving between A100, H100, and newer high-memory GPU environments, the intended operator knobs are:

- `MODEL_CONFIG` for architectural scale
- `BATCH_SIZE` for per-rank micro-batch size
- `GRAD_ACCUM_STEPS` for effective global batch size in non-DeepSpeed runs
- DeepSpeed JSON `gradient_accumulation_steps` for ZeRO-managed runs
- `QUANT_MODE` for BF16/FP16 selection
- The config's `gradient_checkpointing` and `max_seq_len` fields for memory pressure control

Recommended policy:

1. Validate the unchanged 567M reference path first.
2. Increase `BATCH_SIZE` or reduce accumulation on larger GPUs before changing architecture.
3. Escalate ZeRO stage or model config only after the smaller path is stable.
4. Revalidate NCCL/backend behavior any time node topology changes.

## RTX PRO 6000 Blackwell Notes

RTX PRO 6000 Blackwell class nodes use the same repo, the same launchers, and the same config system as the A100 reference path. The operator workflow is:

1. Validate the unchanged `vanta_567m.json` reference path first.
2. Confirm `verify_distributed.py`, tiny smoke training, and shard-backed single-node training all behave normally.
3. Only then raise throughput-oriented knobs such as `BATCH_SIZE`, `GRAD_ACCUM_STEPS`, and worker count.

### Practical Scaling Guidance

Compared with the 20GB A100 baseline path, a 96GB Blackwell-class GPU gives much more room for per-rank batch growth and for disabling memory-saving features only after a stable baseline exists.

| Setting | A100 20GB reference path | RTX PRO 6000 Blackwell direction |
|---|---|---|
| Model validation baseline | `configs/vanta_567m.json` | same baseline first |
| Precision | `QUANT_MODE=bf16` | keep `bf16` as the first-line path |
| `BATCH_SIZE` | conservative default | raise after baseline is stable |
| `GRAD_ACCUM_STEPS` | used to preserve effective batch | can often be reduced as VRAM headroom grows |
| Gradient checkpointing | useful pressure-relief knob | optional only after memory headroom is confirmed |
| ZeRO stage | start with ZeRO-1/ZeRO-2 as needed | escalate only if larger configs demand it |

Important implementation note: the current public repo exposes `bf16` and `fp16` as the documented real precision modes. Do not assume a production-ready FP4 path from the experimental notes; Blackwell hardware may support more aggressive precision strategies, but this repository's current operator contract is still BF16/FP16 plus ZeRO partitioning.

### Known Blackwell Bring-Up Observations

During earlier large-model bring-up work, kernel messages such as `kbifInitLtr_GB202: LTR is disabled` were treated as benign Blackwell driver quirks when they appeared without companion signs of actual failure such as:

- OOM-killer entries in `dmesg`
- NCCL initialisation failures
- rank exits during launcher startup
- monotonic host-memory growth ending in process death

Treat those messages as a signal to keep observing the run, not as standalone proof that the launch failed.

### Blackwell Operator Policy

- Keep the first Blackwell run bounded with a tiny or short-step validation before long jobs.
- Save `nvidia-smi`, `dmesg`, and the exact env/config used for the first stable run on any new node image.
- Increase one major variable at a time. Do not simultaneously change GPU type, batch size, ZeRO stage, and model config.
- If throughput is healthy and memory is stable, scale batch before scaling architecture.
- If you move beyond the 567M path onto the analytical larger configs, revalidate swap, host RAM, and DeepSpeed offload assumptions explicitly.

### Scaling to High-Capacity Compute (Big GPUs)
When scaling up to massive architectures on Blackwell, H100, or later high-memory environments:
- The distributed topology assumptions inherently scale forward. Ensure you revalidate your kernel versions, NCCL backend, and micro-batch settings for the target cluster architecture.
- For immense parameter expansions, rely on ZeRO-3 parameter/optimizer partitioning. The package layout and dynamic JSON configuration (e.g., decoupling dimensions and accumulation steps) are designed to remain forward-compatible for clusters handling these advanced workloads without requiring core logic modifications.
- The forward-compatibility point is architectural, not marketing language: larger GPUs should not require a separate codebase, only a revalidated config, topology, and launch plan.

## Stop Rules for Expensive Runs

Abort the run and fall back to a smaller validation step if any of the following occurs:

- `verify_distributed.py` hangs during initialization for more than a couple of minutes.
- GPU memory allocation rises monotonically step after step without stabilizing.
- Loss becomes `NaN` or `Inf`.
- Throughput collapses suddenly and stays depressed across multiple logging windows.
- Multi-rank jobs stall because ranks consume uneven data or one rank exits early.

Do not patch a live unstable training run. Kill the process group, return to the previous successful stage, and isolate the failure there.

## Known Runtime Unknowns

The repository is intentionally explicit about what still requires empirical hardware validation at larger scales.

- NCCL behavior can still vary across real cluster topologies, drivers, and interconnect layouts.
- ZeRO-3 behavior for the largest analytical model profiles is structurally prepared but still depends on the target node's RAM, swap, and offload characteristics.
- Sustained MoE routing skew under long real-throughput runs remains a runtime property, not something that can be fully proven from short local checks.
- Cross-node scaling efficiency depends on the actual network fabric and storage path, not just on the launcher scripts or JSON configs.

This is normal for systems work. The point of the staged validation path is to collapse these unknowns one layer at a time instead of hiding them.
