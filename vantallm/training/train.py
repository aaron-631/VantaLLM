"""
train.py — VantaLLM training loop.

Standard forward → loss → backward → optimizer step, with support for both
single-process and distributed (DDP / DeepSpeed) execution.

Key features:
  - LFS-stub detection before ``torch.load`` (actionable error on un-hydrated checkpoints)
  - VRAM peak stats reset per session for accurate ``max_memory_allocated`` reporting
  - ``DistributedSampler`` epoch tracking on dataloader restarts (shard mode)
  - Periodic validation on held-out data
  - Vocabulary consistency checks against the tokenizer (151,003)
  - Resumable loops with dataloader fast-forward and periodic interim checkpoints

Usage:
    env TINY_MODEL=1 python3 vantallm/training/train.py
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
import time
import logging
import json
import re
import math
from pathlib import Path

import torch
import torch.optim
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from typing import Dict

# Internal imports
from vantallm.model.transformer import load_model, LLMConfig
from vantallm.tokenizer.tokenizer import load_tokenizer
from vantallm.utils.prompting import PROMPT_VERSION, build_prompt
from vantallm.distributed.init_distributed import setup_distributed, cleanup_distributed
from vantallm.distributed.distributed_utils import get_device, set_deterministic_seed, is_main_process, prepare_distributed_model, rank_safe_log, rank_safe_save, is_deepspeed_engine, get_underlying_model

# A100 Ampere Tensor Core acceleration — free throughput, no precision loss for BF16 training
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32       = True

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
log = logging.getLogger("trainer")

# The repository's tokenizer pipeline packs fixed 2,048-token sequences.
# TRAIN_SEQ_LEN may be raised only when the input shards were repacked to match.
DEFAULT_TRAIN_SEQ_LEN = 2048

# --- Distributed Environment Setup via Helpers ---
is_distributed, world_size, rank, local_rank = setup_distributed()

def log_info(msg: str):
    rank_safe_log(msg, rank, "info")

def log_warning(msg: str):
    rank_safe_log(msg, rank, "warning")

def log_error(msg: str):
    rank_safe_log(msg, rank, "error")


# --- LFS Stub Detection Helper ---
def _is_lfs_stub(path: str) -> bool:
    """
    Returns True if the file at `path` is a Git-LFS pointer stub rather than
    a real PyTorch checkpoint. Stubs are small text files beginning with the
    literal string 'version https://git-lfs.github.com'.
    Reading only 50 bytes is enough to detect the header without loading the
    full file — safe to call before torch.load().
    """
    try:
        with open(path, "rb") as f:
            header = f.read(50)
        return b"git-lfs.github.com" in header
    except OSError:
        return False


# --- Dataset Definition ---
class ShowcaseDataset(Dataset):
    def __init__(self, samples, tokenizer, seq_len):
        self.samples = samples
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.pad_val = getattr(tokenizer, "pad_token_id", 0)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        sample_input = sample.get("input", sample.get("instruction", ""))
        sample_output = sample.get("output", sample.get("response", ""))

        context_str = None
        question_str = sample_input

        if "--- RETRIEVED CONTEXT ---" in sample_input:
            ctx_match = re.search(r"--- RETRIEVED CONTEXT ---\n(.*?)\n--- END CONTEXT ---", sample_input, re.DOTALL)
            if ctx_match:
                context_str = ctx_match.group(1).strip()

            q_match = re.search(r"<\|user\|>\n(.*?)\n<\|assistant\|>", sample_input, re.DOTALL)
            if q_match:
                question_str = q_match.group(1).strip()

        # 1. Format prompt using shared authoritative utility
        user_prompt = build_prompt(question=question_str, context=context_str)

        # 2. Tokenize independently without adding special tokens automatically
        input_ids = self.tokenizer.encode(user_prompt, add_special_tokens=False, return_tensors="list")
        output_ids = self.tokenizer.encode(sample_output.strip(), add_special_tokens=False, return_tensors="list")

        # Manually assemble sequence boundary layouts using raw token integer lists
        full_ids = [self.tokenizer.bos_token_id] + input_ids + output_ids + [self.tokenizer.eos_token_id]
        full_tokens = torch.tensor(full_ids, dtype=torch.long)

        # The prompt boundary length index maps cleanly and securely
        instruction_len = 1 + len(input_ids) 

        # 3. Create labels (-100 excludes instruction prompts from optimization tracking)
        labels = full_tokens.clone()
        labels[:instruction_len] = -100

        # 4. Truncate or pad to exactly seq_len
        if len(full_tokens) > self.seq_len:
            full_tokens = full_tokens[:self.seq_len]
            labels = labels[:self.seq_len]
        else:
            pad_len = self.seq_len - len(full_tokens)
            pad_tensor = torch.full((pad_len,), self.pad_val, dtype=torch.long)
            full_tokens = torch.cat([full_tokens, pad_tensor])

            label_pad_tensor = torch.full((pad_len,), -100, dtype=torch.long)
            labels = torch.cat([labels, label_pad_tensor])

        return full_tokens, labels


def clip_grad_norm_mixed_(parameters, max_norm, norm_type=2.0):
    """
    Robust gradient clipping that supports mixed parameter dtypes (e.g. float32 and bfloat16)
    without triggering type-promotion/concatenation errors on experimental PyTorch/CUDA nightlies.
    Computes norms in float32 for maximum accuracy and safety.
    """
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]
    max_norm = float(max_norm)
    norm_type = float(norm_type)
    if len(parameters) == 0:
        return 0.0

    # Compute the global norm on-device in a single reduction. Stacking the
    # per-parameter (float32) norms and reducing avoids the previous
    # per-parameter `.item()` call, which forced a host<->device sync for every
    # parameter (hundreds of stalls per step). One sync remains at the end for
    # the scalar clip coefficient / logging.
    device = parameters[0].grad.device
    per_param_norms = torch.stack(
        [p.grad.detach().float().norm(norm_type).to(device) for p in parameters]
    )
    total_norm = torch.norm(per_param_norms, norm_type)

    clip_coef = float(max_norm) / (total_norm.item() + 1e-6)
    if clip_coef < 1.0:
        for p in parameters:
            # Use .data.mul_() instead of .detach().mul_() — .data is guaranteed
            # to alias the same storage as p.grad in all PyTorch versions,
            # whereas .detach() alias guarantee does not hold on experimental nightlies.
            p.grad.data.mul_(clip_coef)

    return float(total_norm.item())


def sanity_check_loop():
    """
    Runs a minimal training loop with showcase data to stabilize
    assistant tone and simple code/RAG responses.
    """
    device = get_device(local_rank, is_distributed)

    log_info(f"Using device: {device} | Distributed: {is_distributed} | World Size: {world_size}")

    # Ensure deterministic but rank-unique seeding
    # Using 42 for Rank 0 ensures identical reproduction to earlier local logs.
    set_deterministic_seed(42, rank)

    default_checkpoint_path = ""
    if not os.environ.get("MODEL_CONFIG", "").strip():
        candidate_checkpoint = os.path.join(
            os.path.dirname(__file__), "../../checkpoints/baseline_verified_parity_hardened.pt"
        )
        if os.path.exists(candidate_checkpoint):
            default_checkpoint_path = candidate_checkpoint
    model_path_env = os.environ.get("MODEL_PATH")
    checkpoint_path = os.environ.get("MODEL_PATH", default_checkpoint_path).strip()
    explicit_resume_path = model_path_env is not None and model_path_env.strip() != ""
    save_path = os.path.join(os.path.dirname(__file__), "../../checkpoints/showcase_run.pt")
    
    # Initialize start_step globally
    start_step = 0

    # Resolve DeepSpeed config path from argv (needed for zero.Init)
    _ds_config_path = None
    for _i, _arg in enumerate(sys.argv):
        if _arg.startswith("--deepspeed_config="):
            _ds_config_path = _arg.split("=")[1]
        elif _arg == "--deepspeed_config" and _i + 1 < len(sys.argv):
            _ds_config_path = sys.argv[_i + 1]

    if "--deepspeed" in sys.argv and _ds_config_path:
        import deepspeed as _ds_init
        log_info("DeepSpeed ZeRO-3 detected: using zero.Init() for sharded model construction.")
        _prev_dtype = torch.get_default_dtype()
        torch.set_default_dtype(torch.bfloat16)
        with _ds_init.zero.Init(config_dict_or_path=_ds_config_path):
            model = load_model()
        torch.set_default_dtype(_prev_dtype)
    else:
        model = load_model().to(device)
        if getattr(get_underlying_model(model).config, 'quantization', '') == 'bf16' and torch.cuda.is_available():
            model = model.to(torch.bfloat16)
            log_info("BF16 cast applied: model parameters are now bfloat16.")

    # 1. Environment & Tokenizer Validation
    tokenizer = load_tokenizer()
    if tokenizer is None:
        log_error("CRITICAL: Tokenizer failed to load. Check TOKENIZER_PATH.")
        return False

    # 2. Runtime Observability (Rank 0 only)
    if rank == 0:
        log_info(f"--- GPU Runtime Initialization ---")
        log_info(f"Hostname      : {os.uname().nodename}")
        log_info(f"Python        : {sys.version.split()[0]}")
        log_info(f"Torch         : {torch.__version__}")
        log_info(f"Device        : {device}")
        log_info(f"Model dtype   : {next(get_underlying_model(model).parameters()).dtype}")
        log_info(f"Quantization  : {os.environ.get('QUANT_MODE', 'bf16')}")
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            vram_total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            log_info(f"GPU Name      : {gpu_name}")
            log_info(f"Total VRAM    : {vram_total:.2f} GB")
            log_info(f"MIG Partition : {'Detected' if 'MIG' in gpu_name else 'Standard'}")
        log_info(f"-----------------------------------")

    # Resolve runtime sequence length independently from architectural context.
    # The default matches data_pipeline/tokenize_data.py's fixed shard width.
    requested_seq_len = int(os.environ.get("TRAIN_SEQ_LEN", DEFAULT_TRAIN_SEQ_LEN))
    if requested_seq_len <= 0:
        log_error("TRAIN_SEQ_LEN must be a positive integer.")
        return False
    effective_seq_len = min(requested_seq_len, model.config.max_seq_len)
    if effective_seq_len < requested_seq_len:
        log_warning(
            f"Requested training seq_len={requested_seq_len} exceeds "
            f"model capability={model.config.max_seq_len}. "
            f"Using capped effective_seq_len={effective_seq_len} for validation."
        )

    # FIX [1]: Detect Git-LFS pointer stubs before attempting torch.load.
    _saved_optimizer_state = None
    if checkpoint_path and os.path.exists(checkpoint_path):
        if _is_lfs_stub(checkpoint_path):
            msg = (
                f"Checkpoint at '{checkpoint_path}' is a Git-LFS pointer stub, not real weights. "
                "Run 'git lfs install && git lfs pull' to hydrate the actual checkpoint."
            )
            if explicit_resume_path:
                log_error(f"{msg} Refusing to continue because MODEL_PATH was explicitly requested.")
                return False
            log_warning(f"{msg} Starting from random weights for this run.")
        else:
            if os.path.isdir(checkpoint_path):
                log_info(f"DeepSpeed checkpoint directory detected: {checkpoint_path}. Will load after DeepSpeed init.")
            else:
                try:
                    log_info(f"Resuming training from checkpoint: {checkpoint_path}")
                    # Manual fix for vocab size mismatch (151,000 vs 151,003)
                    checkpoint_data = torch.load(checkpoint_path, map_location=device, weights_only=False)
                    
                    # Backward-compatible metadata loading
                    if isinstance(checkpoint_data, dict) and "model" in checkpoint_data:
                        state_dict = checkpoint_data["model"]
                        if "client_state" in checkpoint_data and checkpoint_data["client_state"]:
                            start_step = checkpoint_data["client_state"].get("step", 0)
                            log_info(f"Restored start_step from checkpoint: {start_step}")
                    else:
                        state_dict = checkpoint_data

                    for key in ["token_embed.weight", "lm_head.weight"]:
                        if key in state_dict:
                            ckpt_weight = state_dict[key]
                            model_weight = model.state_dict()[key]
                            if ckpt_weight.shape != model_weight.shape:
                                log_warning(f"Shape mismatch for {key}: {ckpt_weight.shape} vs {model_weight.shape}. Patching...")
                                new_weight = model_weight.clone()
                                min_vocab = min(ckpt_weight.shape[0], model_weight.shape[0])
                                new_weight[:min_vocab] = ckpt_weight[:min_vocab]
                                state_dict[key] = new_weight
                    _load_result = model.load_state_dict(state_dict, strict=False)
                    if _load_result.missing_keys:
                        log_warning(
                            f"Checkpoint load: {len(_load_result.missing_keys)} missing key(s) "
                            f"(kept random init) — first few: {_load_result.missing_keys[:5]}"
                        )
                    if _load_result.unexpected_keys:
                        log_warning(
                            f"Checkpoint load: {len(_load_result.unexpected_keys)} unexpected key(s) "
                            f"(ignored) — first few: {_load_result.unexpected_keys[:5]}"
                        )
                    if not _load_result.missing_keys and not _load_result.unexpected_keys:
                        log_info("Checkpoint load: all keys matched exactly.")
                    if isinstance(checkpoint_data, dict):
                        _saved_optimizer_state = checkpoint_data.get("optimizer", None)
                except Exception as e:
                    if explicit_resume_path:
                        log_error(
                            f"Checkpoint load failed for explicit MODEL_PATH '{checkpoint_path}': {e}. "
                            "Refusing to continue with random weights."
                        )
                        return False
                    log_warning(
                        f"Checkpoint load failed for {checkpoint_path}: {e}. "
                        "Starting from random weights."
                    )
    elif checkpoint_path:
        if explicit_resume_path:
            log_error(
                f"Explicit MODEL_PATH '{checkpoint_path}' was not found. "
                "Refusing to continue with random weights."
            )
            return False
        log_warning(f"Checkpoint not found at {checkpoint_path}. Starting from random weights.")
    else:
        log_info("MODEL_PATH not set for this run. Starting from random weights.")

    model.train()

    # Dynamic LR assigned via Environment Variables
    peak_lr = float(os.environ.get("LR_PEAK", "3e-6"))
    try:
        import bitsandbytes as bnb
        optimizer = bnb.optim.PagedAdamW8bit(model.parameters(), lr=peak_lr, weight_decay=0.01)
        log_info("Using bitsandbytes PagedAdamW8bit optimizer.")
    except ImportError:
        log_warning("bitsandbytes not available. Falling back to torch.optim.AdamW.")
        optimizer = torch.optim.AdamW(model.parameters(), lr=peak_lr, weight_decay=0.01)

    if _saved_optimizer_state is not None:
        try:
            optimizer.load_state_dict(_saved_optimizer_state)
            log_info("Restored optimizer state (Adam m/v terms).")
        except Exception as opt_e:
            log_warning(f"Optimizer state restore failed: {opt_e}. Continuing with cold optimizer.")
        _saved_optimizer_state = None

    _train_steps_raw = os.environ.get("TRAIN_STEPS", "").strip()
    if not _train_steps_raw:
        log_error("CRITICAL: TRAIN_STEPS environment variable is NOT set.")
        log_error("For hardware safety, you must explicitly set TRAIN_STEPS (e.g. TRAIN_STEPS=100000).")
        return False
    num_steps = int(_train_steps_raw)

    # --- Pre-GPU Distributed Wrapper Hook ---
    # NOTE: lr_scheduler is created AFTER this call so it binds to the
    # correct optimizer (DeepSpeed's DeepSpeedCPUAdam, not the throwaway local AdamW).
    # We pass lr_scheduler=None so DeepSpeed does NOT manage the scheduler internally;
    # instead we step it ourselves in the training loop for full control.
    model, optimizer, _ = prepare_distributed_model(model, optimizer, None, is_distributed, local_rank)

    # DeepSpeed may wrap the optimizer; extract the inner torch optimizer for schedulers.
    if hasattr(optimizer, 'optimizer') and isinstance(optimizer.optimizer, torch.optim.Optimizer):
        base_optimizer = optimizer.optimizer
    else:
        base_optimizer = optimizer

    # If resuming a DeepSpeed checkpoint directory, load it now that DeepSpeed is initialized
    # NOTE: This MUST happen BEFORE LR scheduler creation so start_step is correct.
    if is_deepspeed_engine(model) and checkpoint_path and os.path.isdir(checkpoint_path):
        log_info(f"Resuming DeepSpeed checkpoint from: {checkpoint_path}")
        load_dir = os.path.dirname(checkpoint_path)
        tag = os.path.basename(checkpoint_path)
        _, client_state = model.load_checkpoint(load_dir, tag=tag)
        if client_state:
            start_step = client_state.get("step", 0)
            log_info(f"Restored DeepSpeed start_step: {start_step}")

    # LR warmup + cosine decay — bound to the FINAL optimizer returned above.
    # For DeepSpeed, num_steps counts micro-batches but scheduler steps only
    # fire at accumulation boundaries, so we divide by DS grad_accum_steps.
    from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
    _ds_accum = 1
    if is_deepspeed_engine(model):
        _ds_accum = model.gradient_accumulation_steps()
    else:
        # DDP path: the loop calls lr_scheduler.step() only at gradient-
        # accumulation boundaries, so the number of optimizer steps is
        # num_steps // grad_accum_steps. Sizing the cosine for num_steps
        # micro-batches (the old _ds_accum=1) meant it never reached eta_min.
        _ds_accum = max(1, int(os.environ.get("GRAD_ACCUM_STEPS", 16)))
    _optimizer_steps = max(1, num_steps // _ds_accum)
    _warmup_steps = int(os.environ.get("LR_WARMUP_STEPS", max(1, min(200, _optimizer_steps // 10))))
    _cosine_steps = max(1, _optimizer_steps - _warmup_steps)
    _warmup_sched  = LinearLR(base_optimizer, start_factor=0.1, end_factor=1.0, total_iters=_warmup_steps)
    _cosine_sched  = CosineAnnealingLR(base_optimizer, T_max=_cosine_steps, eta_min=3e-7)
    lr_scheduler   = SequentialLR(base_optimizer, schedulers=[_warmup_sched, _cosine_sched], milestones=[_warmup_steps])
    log_info(f"LR schedule: warmup {_warmup_steps} steps → cosine decay to 3e-7 over {_cosine_steps} optimizer steps (DS accum={_ds_accum})")

    # Fast-forward scheduler to match resumed checkpoint step.
    if start_step > 0:
        _resume_optimizer_steps = start_step // _ds_accum
        log_info(f"Fast-forwarding LR scheduler to optimizer step {_resume_optimizer_steps}...")
        for _ in range(_resume_optimizer_steps):
            lr_scheduler.step()
        log_info(f"LR scheduler aligned. Current LR: {lr_scheduler.get_last_lr()}")

    # Optimization & Hardware
    batch_size = int(os.environ.get("BATCH_SIZE", 2))
    grad_accum_steps = int(os.environ.get("GRAD_ACCUM_STEPS", 16))

    if is_deepspeed_engine(model) and grad_accum_steps > 1:
        log_warning(
            "GRAD_ACCUM_STEPS > 1 is set, but DeepSpeed is enabled. "
            "Python-level accumulation is ignored. "
            "Set `gradient_accumulation_steps` in the DeepSpeed config instead."
        )

    _effective_accum = model.gradient_accumulation_steps() if is_deepspeed_engine(model) else grad_accum_steps
    global_batch_size = batch_size * world_size * _effective_accum

    log_info(f"--- Distributed Scaling Info ---")
    log_info(f"Local Batch Size: {batch_size}")
    log_info(f"Gradient Accumulation Steps: {grad_accum_steps}")
    log_info(f"World Size: {world_size}")
    log_info(f"Effective Global Batch Size: {global_batch_size}")
    log_info(f"Learning Rate: {optimizer.param_groups[0]['lr']:.2e}")
    log_info(f"--------------------------------")

    raw_model = get_underlying_model(model)
    vocab_size = raw_model.config.vocab_size
    num_persona = raw_model.config.num_persona_types

    # Pretraining shards carry no persona labels, so every token is conditioned on
    # a single fixed "neutral" persona (id 0 by default). This is intentional:
    # persona conditioning is applied later at SFT/inference, not during base
    # pretraining. Exposed via env so a persona-labelled run can override it;
    # must stay within [0, num_persona_types).
    pretrain_persona_id = int(os.environ.get("PRETRAIN_PERSONA_ID", "0"))
    if not 0 <= pretrain_persona_id < num_persona:
        log_error(
            f"PRETRAIN_PERSONA_ID={pretrain_persona_id} out of range "
            f"[0, {num_persona})."
        )
        return False

    # --- Dataset Selection ---
    data_mode = os.environ.get("DATA_MODE", "showcase").lower()

    if data_mode == "showcase":
        data_path = os.environ.get(
            "DATA_PATH",
            os.path.join(os.path.dirname(__file__), "../../data_pipeline/rag_synthetic_slice_v2_fixed.jsonl"),
        )
        if not os.path.exists(data_path):
            log_error(f"Data file not found: {data_path}")
            return False

        log_info(f"Loading showcase data from {data_path}...")
        samples = []
        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                samples.append(json.loads(line))

        log_info(f"SHOWCASE TRAINING: Using {len(samples)} source samples.")

        total_samples_needed = num_steps * batch_size
        if is_distributed:
            global_samples_needed = total_samples_needed * world_size
            extended_samples = samples * ((global_samples_needed // len(samples)) + 1)
            extended_samples = extended_samples[:global_samples_needed]
        else:
            extended_samples = samples * ((total_samples_needed // len(samples)) + 1)
            extended_samples = extended_samples[:total_samples_needed]

        dataset = ShowcaseDataset(extended_samples, tokenizer, effective_seq_len)
    else:
        # Production Shard Mode
        from verify_gpu_ready import ShardedDataset
        shard_dir = os.path.join(os.path.dirname(__file__), "..", "..", os.environ.get("SHARD_DIR", "data_pipeline/shards"))
        shard_paths = sorted(str(p) for p in Path(shard_dir).glob("*.bin"))
        if not shard_paths:
            log_error(
                f"No .bin shards found in '{shard_dir}'. "
                "Run 'python3 data_pipeline/shard_data.py' (or './run_pipeline.sh') first."
            )
            return False
        dataset = ShardedDataset(shard_paths, seq_len=effective_seq_len)

    # --- Validation Dataset (Optional) ---
    val_dataset = None
    val_path = os.environ.get("VAL_PATH", os.path.join(os.path.dirname(__file__), "../../data_pipeline/phase1_val.jsonl"))
    if os.path.exists(val_path):
        log_info(f"Loading validation data from {val_path}...")
        val_samples = []
        with open(val_path, "r", encoding="utf-8") as f:
            for line in f:
                val_samples.append(json.loads(line))
        val_dataset = ShowcaseDataset(val_samples, tokenizer, effective_seq_len)
    
    log_info(f"Using prompt utility version: {PROMPT_VERSION}")
    log_info(f"Config: vocab={vocab_size}, seq={effective_seq_len}, batch={batch_size}, steps={num_steps}")

    sampler = None
    if is_distributed:
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=(data_mode == "shard"), drop_last=False)

    num_workers = int(os.environ.get("NUM_WORKERS", 0))
    pin_memory = torch.cuda.is_available() and num_workers > 0
    prefetch_factor = int(os.environ.get("PREFETCH_FACTOR", 2)) if num_workers > 0 else None

    # Final Dataloaders
    shuffle_data = (sampler is None)  # Shuffle when not using DistributedSampler
    dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, shuffle=shuffle_data, num_workers=num_workers, pin_memory=pin_memory, prefetch_factor=prefetch_factor, drop_last=False)
    
    val_loader = None
    if val_dataset:
        val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False) if is_distributed else None
        val_loader = DataLoader(val_dataset, batch_size=batch_size, sampler=val_sampler, num_workers=num_workers, pin_memory=pin_memory)

    log_info("Starting showcase training loop...")
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    current_epoch = 0
    if is_distributed and sampler is not None:
        sampler.set_epoch(current_epoch)

    data_iter = iter(dataloader)
    
    # Dataloader fast-forwarding
    if start_step > 0:
        log_info(f"Fast-forwarding dataloader: skipping {start_step} batches...")
        for _ in range(start_step):
            try:
                next(data_iter)
            except StopIteration:
                current_epoch += 1
                if is_distributed and sampler is not None:
                    sampler.set_epoch(current_epoch)
                data_iter = iter(dataloader)
                next(data_iter)

    optimizer.zero_grad()

    # Loop executes from start_step
    for i in range(start_step, num_steps):
        t0 = time.perf_counter()

        try:
            batch_ids, batch_labels = next(data_iter)
        except StopIteration:
            current_epoch += 1
            if is_distributed and sampler is not None:
                sampler.set_epoch(current_epoch)
            data_iter = iter(dataloader)
            batch_ids, batch_labels = next(data_iter)

        input_ids = batch_ids.to(device, dtype=torch.long, non_blocking=True)
        labels    = batch_labels.to(device, dtype=torch.long, non_blocking=True)
        p_ids     = torch.full((input_ids.size(0), num_persona), pretrain_persona_id, dtype=torch.long, device=device)

        # 1. Forward
        out = model(input_ids, persona_ids=p_ids, labels=labels)
        loss = out["loss"]

        # 2. Step
        if is_deepspeed_engine(model):
            is_accumulation_boundary = model.is_gradient_accumulation_boundary()
            model.backward(loss)
            model.step()
            if is_accumulation_boundary:
                lr_scheduler.step()
            # DeepSpeed handles its own grad clipping/norm calculation.
            # We extract it here for logging visibility.
            try:
                grad_norm = model.get_global_grad_norm()
                if grad_norm is None:
                    grad_norm = 0.0
            except Exception:
                grad_norm = 0.0
        else:
            is_accumulation_boundary = ((i + 1) % grad_accum_steps == 0) or (i == num_steps - 1)
            total_loss = loss / grad_accum_steps
            # Suppress redundant all-reduce on non-boundary accumulation steps (DDP only)
            if is_distributed and not is_accumulation_boundary and hasattr(model, "no_sync"):
                with model.no_sync():
                    total_loss.backward()
            else:
                total_loss.backward()
            if is_accumulation_boundary:
                grad_norm = clip_grad_norm_mixed_(model.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
            else:
                grad_norm = 0.0

        # Periodic interim checkpoint (Shared unconditionally for DDP/DeepSpeed)
        _ckpt_every = int(os.environ.get('CHECKPOINT_EVERY', '0'))
        if _ckpt_every > 0 and (i + 1) % _ckpt_every == 0:
            _interim_path = save_path.replace('.pt', f'_interim_step{i+1}.pt')
            rank_safe_save(model, _interim_path, rank, client_state={"step": i + 1})
            log_info(f"Interim checkpoint saved: {_interim_path}")

        dt = (time.perf_counter() - t0) * 1000

        if i % 10 == 0 or i == num_steps - 1:
            if rank == 0:
                tps = (input_ids.numel() * world_size) / (dt / 1000)
                gn_str = f"{grad_norm:.4f}" if is_accumulation_boundary else "accumulating"
                log_info(f"Step {i:4d} | Loss: {loss.item():.4f} | GradNorm: {gn_str} | {tps:.0f} tok/s")
                
                layers = get_underlying_model(model).layers
                moe_layer = next((l.ffn for l in layers if getattr(l, "use_moe", False)), None)
                if moe_layer and hasattr(moe_layer.router, "_telemetry"):
                    tel = moe_layer.router._telemetry
                    log_info(f"  -> MoE [First MoE] | Max Expert Load: {tel.get('max_load', 0)} | Skew: +{tel.get('skew_pct', 0):.1f}%")
                if torch.cuda.is_available():
                    log_info(
                        f"  -> VRAM | Alloc: {torch.cuda.memory_allocated()/1e6:.0f}MB"
                        f" | Res:  {torch.cuda.memory_reserved()/1e6:.0f}MB"
                        f" | Peak: {torch.cuda.max_memory_allocated()/1e6:.0f}MB"
                    )

        if torch.isnan(loss):
            log_error("CRITICAL: Loss is NaN! Training diverged.")
            return False

        # --- Validation Step ---
        val_every = int(os.environ.get("VAL_EVERY", 50))
        if val_loader and (i + 1) % val_every == 0:
            model.eval()
            val_loss, val_steps = 0.0, 0
            log_info(f"Running validation at step {i+1}...")
            with torch.no_grad():
                for v_ids, v_labels in val_loader:
                    v_ids, v_labels = v_ids.to(device, dtype=torch.long, non_blocking=True), v_labels.to(device, dtype=torch.long, non_blocking=True)
                    v_p_ids = torch.full((v_ids.size(0), num_persona), pretrain_persona_id, dtype=torch.long, device=device)
                    v_out = model(v_ids, persona_ids=v_p_ids, labels=v_labels)
                    val_loss += v_out["loss"].item()
                    val_steps += 1
                    if val_steps >= 20: break
            
            avg_val_loss = val_loss / max(val_steps, 1)
            if is_distributed:
                val_loss_t = torch.tensor(avg_val_loss, device=device)
                dist.all_reduce(val_loss_t, op=dist.ReduceOp.SUM)
                avg_val_loss = val_loss_t.item() / world_size
            
            ppl = math.exp(min(avg_val_loss, 88.0))
            log_info(f"Step {i+1:4d} | Validation Loss: {avg_val_loss:.4f} | PPL: {ppl:.2f}")
            model.train()

    log_info("Showcase Training Complete.")

    if rank == 0 and not is_deepspeed_engine(model):
        log_info("\n--- FINAL GENERATION CHECK ---")
        model.eval()
        test_prompt = build_prompt(question="Write a Python function to add two numbers.")
        test_ids = tokenizer.encode(test_prompt).to(device)[:, :-1]
        from vantallm.model.transformer import greedy_generate
        raw_model_for_gen = get_underlying_model(model)
        with torch.no_grad():
            gen_ids = greedy_generate(raw_model_for_gen, test_ids, max_new_tokens=40)
        log_info(f"OUTPUT:\n{tokenizer.decode(gen_ids[0, test_ids.shape[1]:])}")

    # Unconditional call to rank_safe_save (it natively handles rank logic inside)
    rank_safe_save(model, save_path, rank, client_state={"step": num_steps}, optimizer=optimizer if not is_deepspeed_engine(model) else None)
    
    return True

if __name__ == "__main__":
    if "TINY_MODEL" not in os.environ and not os.environ.get("MODEL_CONFIG", "").strip():
        os.environ["TINY_MODEL"] = "1"
    success = False
    try:
        success = sanity_check_loop()
    finally:
        cleanup_distributed()
    if rank == 0:
        if success:
            print("\n✅ Showcase training successful.")
        else:
            print("\n❌ Showcase training FAILED.")
            sys.exit(1)
