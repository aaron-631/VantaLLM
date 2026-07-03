"""
gpu_validation_5gb.py — Staged GPU Validation for 5GB A100 MIG
==============================================================
Single-GPU smoke validation for the VantaLLM training stack.
Validates: CUDA, BF16, Tiny-Model Training, Checkpoint Save/Load, Inference.
"""

import os
import sys
import torch
import logging
import time
import subprocess

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger("gpu_val")

def run_command(cmd, env_vars=None):
    log.info(f"Running: {cmd}")
    env = os.environ.copy()
    if env_vars:
        env.update(env_vars)
    
    result = subprocess.run(cmd, shell=True, env=env, capture_output=True, text=True)
    if result.returncode != 0:
        log.error(f"Command failed with exit code {result.returncode}")
        log.error(f"STDOUT: {result.stdout}")
        log.error(f"STDERR: {result.stderr}")
        return False, result.stdout
    return True, result.stdout

def log_gpu_stats(label=""):
    log.info(f"--- GPU Stats: {label} ---")
    run_command("nvidia-smi")
    if torch.cuda.is_available():
        vram_alloc = torch.cuda.memory_allocated() / (1024**2)
        vram_res = torch.cuda.memory_reserved() / (1024**2)
        log.info(f"Wrapper VRAM | Alloc: {vram_alloc:.2f}MB | Res: {vram_res:.2f}MB")

def validate_environment():
    log_gpu_stats("Baseline")
    log.info("--- Step 1: Environment Sanity ---")
    if not torch.cuda.is_available():
        log.error("CUDA NOT AVAILABLE. Check drivers and nvidia-smi.")
        return False
    
    gpu_name = torch.cuda.get_device_name(0)
    vram_total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    bf16_supported = torch.cuda.is_bf16_supported()
    
    log.info(f"GPU Name      : {gpu_name}")
    log.info(f"Total VRAM    : {vram_total:.2f} GB")
    log.info(f"BF16 Supported: {bf16_supported}")
    
    if vram_total < 4.5:
        log.warning(f"Very low VRAM detected ({vram_total:.2f} GB). Proceeding with extreme caution.")
    
    return True

def validate_model_demo():
    log.info("--- Step 2: Architecture Sanity (Model Demo) ---")
    env = {"QUANT_MODE": "bf16", "TINY_MODEL": "1"}
    success, output = run_command("python3 vantallm/model/transformer.py", env_vars=env)
    if success:
        log.info("Model demo passed on GPU/BF16.")
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return success

def validate_training():
    log.info("--- Step 3: Tiny-Model Training (10 steps) ---")
    # Reduced from 50 to 10 steps for initial contact safety
    env = {
        "TINY_MODEL": "1",
        "QUANT_MODE": "bf16",
        "TRAIN_STEPS": "10",
        "BATCH_SIZE": "2",  # Conservative for 5GB
        "MODEL_PATH": ""    # Start from random weights
    }
    log_gpu_stats("Pre-Training")
    success, output = run_command("python3 vantallm/training/train.py", env_vars=env)
    log_gpu_stats("Post-Training")
    
    if success:
        log.info("Tiny-model training loop executed successfully.")
        # Check if checkpoint was saved
        if os.path.exists("checkpoints/showcase_run.pt"):
            log.info("Checkpoint 'checkpoints/showcase_run.pt' verified.")
        else:
            log.error("Training succeeded but checkpoint was NOT found.")
            return False
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return success

def validate_inference():
    log.info("--- Step 4: Inference Sanity ---")
    # Verify we can load the newly saved checkpoint and generate
    env = {
        "TINY_MODEL": "1",
        "QUANT_MODE": "bf16",
        "MODEL_PATH": "checkpoints/showcase_run.pt"
    }
    # Simple generation test script inline
    gen_script = """
import torch
import os
from vantallm.model.transformer import load_model, greedy_generate
from vantallm.tokenizer.tokenizer import load_tokenizer
from vantallm.utils.prompting import build_prompt

device = torch.device("cuda")
tokenizer = load_tokenizer()
model = load_model(model_path="checkpoints/showcase_run.pt").to(device)
if torch.cuda.is_available():
    model = model.to(torch.bfloat16)

prompt = build_prompt(question="Hello GPU!")
ids = tokenizer.encode(prompt).to(device)
# remove EOS if present
if ids[0, -1] == tokenizer.eos_token_id:
    ids = ids[:, :-1]

output_ids = greedy_generate(model, ids, max_new_tokens=20)
output_text = tokenizer.decode(output_ids[0, ids.shape[1]:])
print(f"GEN_OUTPUT: {output_text}")

if torch.cuda.is_available():
    torch.cuda.empty_cache()
"""
    with open("temp_gen_test.py", "w") as f:
        f.write(gen_script)
    
    log_gpu_stats("Pre-Inference")
    success, output = run_command("python3 temp_gen_test.py", env_vars=env)
    log_gpu_stats("Post-Inference")
    
    if success:
        log.info("Inference sanity test passed.")
        for line in output.splitlines():
            if "GEN_OUTPUT:" in line:
                log.info(f"Generated text: {line.split('GEN_OUTPUT:')[1].strip()}")
    
    if os.path.exists("temp_gen_test.py"):
        os.remove("temp_gen_test.py")
        
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return success

def main():
    log.info("Starting single-GPU validation (5GB target)")
    log.info("="*50)
    
    steps = [
        validate_environment,
        validate_model_demo,
        validate_training,
        validate_inference
    ]
    
    for step in steps:
        if not step():
            log.error(f"Validation FAILED at {step.__name__}")
            sys.exit(1)
        print("\n")
            
    log.info("="*50)
    log.info("ALL GPU VALIDATION STEPS PASSED.")
    log.info("VantaLLM has successfully executed on real NVIDIA GPU hardware.")

if __name__ == "__main__":
    main()
