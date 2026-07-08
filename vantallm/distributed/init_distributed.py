import os
import torch
import torch.distributed as dist
from typing import Tuple

def setup_distributed() -> Tuple[bool, int, int, int]:
    """
    Initializes the distributed process group.
    Falls back gracefully to local CPU/GPU execution if not run via torchrun/deepspeed.
    """
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    is_distributed = world_size > 1

    if is_distributed and not dist.is_initialized():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
        try:
            import deepspeed
            # deepspeed.zero.Init() reads world size from DeepSpeed's OWN comm
            # backend, not raw torch.distributed. If we only call
            # torch.distributed.init_process_group() here, DeepSpeed's internal
            # DeepSpeedConfig still computes world_size=1 and the batch-size
            # assertion fails (e.g. "train_batch_size 128 != 1 * 1 * 16") on any
            # multi-GPU / multi-node run. Initializing through DeepSpeed sets up
            # BOTH its comm layer and the underlying torch process group.
            deepspeed.init_distributed(dist_backend=backend)
        except ImportError:
            # DeepSpeed not installed (plain DDP path): fall back to raw torch.
            dist.init_process_group(backend=backend)

    return is_distributed, world_size, rank, local_rank

def cleanup_distributed():
    """
    Destroys the distributed process group safely.
    """
    if dist.is_initialized():
        dist.destroy_process_group()
