import os
import logging
import torch
from typing import Tuple

log = logging.getLogger("distributed_utils")

def get_device(local_rank: int, is_distributed: bool) -> torch.device:
    """Returns the correct device assignment for the current rank or local fallback."""
    if is_distributed and torch.cuda.is_available():
        return torch.device(f"cuda:{local_rank}")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def set_deterministic_seed(base_seed: int, rank: int):
    """
    Sets deterministic seeds across the cluster.
    Rank is added to the base seed to ensure unique deterministic shuffling
    and dropout randomness per GPU rank.
    """
    seed = base_seed + rank
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def is_deepspeed_engine(model) -> bool:
    try:
        import deepspeed
        return isinstance(model, deepspeed.DeepSpeedEngine)
    except ImportError:
        return False

def get_underlying_model(model):
    """Safely extracts the raw model from potential DDP/DeepSpeed wrappers."""
    return model.module if hasattr(model, "module") else model

def is_main_process(rank: int) -> bool:
    """True only for the master rank (rank 0)."""
    return rank == 0

def prepare_distributed_model(model, optimizer, lr_scheduler, is_distributed: bool, local_rank: int) -> Tuple[torch.nn.Module, torch.optim.Optimizer, object]:
    """
    HOOK POINT FOR DDP OR DEEPSPEED (PRE-GPU SCAFFOLDING).
    """
    import sys
    use_deepspeed = "--deepspeed" in sys.argv
    if use_deepspeed:
        try:
            import deepspeed
        except ImportError:
            raise RuntimeError("DeepSpeed requested but not installed.")
        
        config_path = None
        for i, arg in enumerate(sys.argv):
            if arg.startswith("--deepspeed_config="):
                config_path = arg.split("=")[1]
            elif arg == "--deepspeed_config" and i + 1 < len(sys.argv):
                config_path = sys.argv[i + 1]
        
        if config_path and not os.path.exists(config_path):
            raise RuntimeError(f"DeepSpeed config not found: {config_path}")

        model, optimizer_ds, _, lr_scheduler = deepspeed.initialize(
            model=model, 
            model_parameters=model.parameters(),
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            config=config_path
        )
        optimizer = optimizer_ds
        return model, optimizer, lr_scheduler
    elif is_distributed:
        if torch.cuda.is_available():
            major, minor = torch.cuda.get_device_capability(local_rank)
            rank_safe_log(f"Rank {local_rank} GPU Capability: {major}.{minor}", local_rank)
            if major < 8:
                rank_safe_log(f"WARNING: Rank {local_rank} is pre-Ampere. BF16 performance may be suboptimal.", local_rank, "warning")
            
            model = torch.nn.parallel.DistributedDataParallel(
                model, 
                device_ids=[local_rank],
                find_unused_parameters=True
            )
        else:
            model = torch.nn.parallel.DistributedDataParallel(
                model,
                find_unused_parameters=True
            )
        return model, optimizer, lr_scheduler

    return model, optimizer, lr_scheduler

def rank_safe_log(msg: str, rank: int, level: str = "info"):
    """Logs only on Rank 0 to prevent cluster log duplication."""
    if is_main_process(rank):
        if level == "info":
            logging.getLogger("trainer").info(msg)
        elif level == "warning":
            logging.getLogger("trainer").warning(msg)
        elif level == "error":
            logging.getLogger("trainer").error(msg)

def rank_safe_save(model, save_path: str, rank: int, client_state: dict = None, optimizer=None):
    """Saves checkpoint only on Rank 0 to prevent simultaneous write corruption."""
    if is_deepspeed_engine(model):
        save_dir = os.path.dirname(save_path)
        tag = os.path.basename(save_path).replace('.pt', '')
        os.makedirs(save_dir, exist_ok=True)
        model.save_checkpoint(save_dir, tag=tag, client_state=client_state)
        rank_safe_log(f"Saved DeepSpeed checkpoint to {os.path.join(save_dir, tag)}", rank, "info")
    elif is_main_process(rank):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        # Handle potential DDP module wrapper safely
        state_dict = model.module.state_dict() if hasattr(model, "module") else model.state_dict()
        checkpoint = {
            "model": state_dict,
            "optimizer": optimizer.state_dict() if optimizer is not None else None,
            "client_state": client_state,
        }
        torch.save(checkpoint, save_path)
        rank_safe_log(f"Saved checkpoint to {save_path}", rank, "info")
