import os
import sys
import torch
import torch.nn as nn
import math

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from vantallm.model.transformer import VantaTransformer, LLMConfig
from vantallm.tokenizer.tokenizer import load_tokenizer

def run_final_audit():
    print("========================================")
    print(" FINAL STAGE 0.5 VALIDATION TESTS")
    print("========================================")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = load_tokenizer()
    
    cfg = LLMConfig(
        vocab_size=151003,
        d_model=64,
        max_seq_len=128,
        num_layers=4,
        num_q_heads=4,
        num_kv_heads=2,
        head_dim=16,
        d_ff_dense=128,
        num_experts=4,
        top_k_experts=2,
        d_ff_moe=64,
        moe_every_n=2,
        quantization="fp16",
        composite_persona_dim=32,
    )
    
    model = VantaTransformer(cfg).to(device)
    
    # ---------------------------------------------------------
    # TEST 1: BATCH SIZE = 4 STABILITY
    # ---------------------------------------------------------
    print("\n--- TEST 1: BATCH SIZE = 4 TRAINING STABILITY ---")
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-3)
    
    batch_size = 4
    seq_len = 16
    vocab_size = cfg.vocab_size
    
    # Structured predictable data: shifted sequences for each batch element
    base_seq = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)
    shift = torch.arange(batch_size).unsqueeze(1)
    batch_inputs = ((base_seq + shift) % vocab_size).to(device)
    batch_labels = batch_inputs.clone()
    
    losses = []
    grad_norms = []
    
    print("Training with batch_size=4 for 50 steps...")
    for step in range(50):
        optimizer.zero_grad()
        out = model(batch_inputs, labels=batch_labels)
        loss = out["loss"]
        loss.backward()
        
        # Calculate total gradient norm
        total_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        grad_norms.append(total_norm)
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        losses.append(loss.item())
        
        if step % 10 == 0 or step == 49:
            print(f"  Step {step:2d} | Loss: {loss.item():.4f} | Grad Norm: {total_norm:.4f}")
            
    if losses[-1] < losses[0] and losses[-1] < 1.0:
        print("PASS: Loss significantly decreased with batch_size=4.")
    else:
        print("FAIL: Loss did not decrease sufficiently.")
        
    stable_grads = all(not math.isnan(g) and not math.isinf(g) and g > 0 for g in grad_norms)
    if stable_grads:
        print("PASS: Gradients remained stable, non-zero, and finite throughout training.")
    else:
        print("FAIL: Gradients became unstable (zero, NaN, or Inf).")

    # ---------------------------------------------------------
    # TEST 2: SEQ_LEN = 16 TRAIN -> SEQ_LEN = 64 EVAL
    # ---------------------------------------------------------
    print("\n--- TEST 2: LENGTH GENERALIZATION (TRAIN 16 -> EVAL 64) ---")
    
    # We will train on a repeating sequence of length 16.
    # Then we will evaluate the loss on the SAME repeating sequence extended to length 64.
    # If RoPE and causal masking work correctly, the model should generalize well.
    
    model2 = VantaTransformer(cfg).to(device)
    model2.train()
    optimizer2 = torch.optim.AdamW(model2.parameters(), lr=1e-2)
    
    # We bypass the tokenizer to ensure exact lengths of 16 and 64.
    # Pattern: [0, 1, 2, 3, 0, 1, 2, 3, ...]
    train_ids = (torch.arange(16) % 4).unsqueeze(0).to(device)
    eval_ids = (torch.arange(64) % 4).unsqueeze(0).to(device)
    
    print(f"Training on sequence length = {train_ids.shape[1]} for 100 steps...")
    for step in range(100):
        optimizer2.zero_grad()
        out = model2(train_ids, labels=train_ids)
        loss = out["loss"]
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model2.parameters(), 1.0)
        optimizer2.step()
        if step % 20 == 0 or step == 99:
            print(f"  Train Step {step:2d} | Train Loss: {loss.item():.4f}")
            
    print(f"Evaluating on extended sequence length = {eval_ids.shape[1]}...")
    model2.eval()
    with torch.no_grad():
        out_eval = model2(eval_ids, labels=eval_ids)
        eval_loss = out_eval["loss"].item()
        
    print(f"  Eval Loss (seq_len={eval_ids.shape[1]}): {eval_loss:.4f}")
    
    # If the model learned the pattern "abc " on length 16, 
    # it should apply to length 64 without catastrophic failure.
    # An eval loss significantly lower than the initial ~11.92 entropy 
    # proves the causal masking and RoPE are partially generalizing.
    if eval_loss < 5.0:
        print("PASS: Model successfully extrapolated to sequence length 64 with non-random loss.")
    else:
        print(f"FAIL: Model failed to generalize to longer sequence. Eval loss too high: {eval_loss:.4f}")

    print("\n========================================")
    print(" FINAL VERDICT: SMOKE VALIDATION COMPLETE")
    print("========================================")

if __name__ == "__main__":
    run_final_audit()
