"""
verify_distributed.py — Standalone Infrastructure Validation Tool
===============================================================
Minimal, deterministic tool to verify CUDA, NCCL, and PyTorch DDP
initialization before loading heavy models or hitting OOMs.

Supports both GPU (NCCL) and CPU (Gloo) distributed validation.

Usage:
    torchrun --nproc_per_node=2 verify_distributed.py
"""

import os
import sys
import socket
import torch
import torch.distributed as dist

def verify_env():
    # 1. Gather Local Environment
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    hostname = socket.gethostname()

    print(f"[Init] Process starting on {hostname} | Rank: {rank}/{world_size-1} | Local Rank: {local_rank}")

    # 2. Verify Device Visibility
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        if local_rank >= device_count:
            print(f"[Rank {rank}] ❌ FATAL: LOCAL_RANK {local_rank} exceeds device count {device_count}!")
            sys.exit(1)

        torch.cuda.set_device(local_rank)
        device_name = torch.cuda.get_device_name(local_rank)
        vram_gb = torch.cuda.get_device_properties(local_rank).total_memory / (1024**3)
        device = torch.device(f"cuda:{local_rank}")
        print(f"[Rank {rank}] ✅ CUDA ok. Device: {device_name} | VRAM: {vram_gb:.1f}GB")
    else:
        print(f"[Rank {rank}] ⚠️ CUDA not available. Using CPU for validation.")
        device = torch.device("cpu")

    # 3. Verify Distributed Backend and DDP
    if world_size > 1:
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        
        if backend == "nccl" and not dist.is_nccl_available():
            print(f"[Rank {rank}] ❌ FATAL: NCCL backend is not available!")
            sys.exit(1)

        try:
            dist.init_process_group(backend=backend, init_method="env://")
            print(f"[Rank {rank}] ✅ PyTorch DDP Initialized ({backend.upper()}).")
        except Exception as e:
            print(f"[Rank {rank}] ❌ FATAL: Failed to init process group: {e}")
            sys.exit(1)

        # 4. Basic All-Reduce Sanity
        tensor = torch.ones(1, device=device)
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)

        if tensor.item() == world_size:
            print(f"[Rank {rank}] ✅ {backend.upper()} All-Reduce successful! (Got {tensor.item()}, expected {world_size})")
        else:
            print(f"[Rank {rank}] ❌ FATAL: All-Reduce failed! (Got {tensor.item()}, expected {world_size})")
            sys.exit(1)

        dist.barrier()
        if rank == 0:
            print(f"\n🚀 ALL CHECKS PASSED: Distributed environment ({backend.upper()}) is ready.")

        dist.destroy_process_group()
    else:
        print(f"[Rank {rank}] ⚠️ WORLD_SIZE=1. Skipping distributed validation.")
        print("\n🚀 SINGLE-NODE CHECKS PASSED.")

if __name__ == "__main__":
    verify_env()
