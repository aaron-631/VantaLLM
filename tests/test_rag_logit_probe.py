"""
tests/test_rag_logit_probe.py
=============================

Purpose:
  Scientific debugging of the attention mechanism. This test measures the 
  direct impact of retrieved context on the model's output distribution 
  (logits) before any sampling or decoding takes place.

Method:
  Probability Comparison:
  1. Baseline: Calculate the probability of the target token (e.g., "purple") 
     given only the question.
  2. RAG: Calculate the probability of the same target token when the 
     contradicting context is provided in the prompt.
  3. Ratio: Compute the ratio of (RAG Probability / Baseline Probability).

Goal:
  If the ratio is significantly greater than 1 (e.g., > 5x), it proves the 
  Grouped Query Attention (GQA) heads are successfully attending to the 
  context, even if the model is too underfitted to produce a coherent sentence.
"""
import os
import sys
import torch
import torch.nn.functional as F

TESTS_DIR = os.path.abspath(os.path.dirname(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(TESTS_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Ensure TINY_MODEL is set for CPU-based testing
os.environ["TINY_MODEL"] = "1"

from vantallm.model.transformer import load_model
from vantallm.tokenizer.tokenizer import load_tokenizer

from vantallm.utils.prompting import build_prompt, PROMPT_VERSION

def encode_prompt(tokenizer, question: str, context: str, device):
    prompt_str = build_prompt(question=question, context=context)
    input_ids = tokenizer.encode(prompt_str, return_tensors="pt").to(device)
    if input_ids[0, -1].item() == tokenizer.eos_token_id:
        input_ids = input_ids[:, :-1]
    return input_ids

@torch.no_grad()
def get_token_prob(model, tokenizer, question: str, context: str, target_token_id: int, device):
    input_ids = encode_prompt(tokenizer, question, context, device)
    
    # Forward pass
    logits, _, _ = model(input_ids) # (1, seq_len, vocab_size)
    
    # Get logits for the very last token
    last_token_logits = logits[0, -1, :]
    
    # Apply Softmax to get probabilities
    probs = F.softmax(last_token_logits, dim=-1)
    
    return probs[target_token_id].item()

def run_logit_probe():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Prompt Utility Version: {PROMPT_VERSION}")

    tokenizer = load_tokenizer()

    def single_token_id(text: str) -> int:
        token_ids = tokenizer.encode(
            text,
            return_tensors=None,
            add_special_tokens=False,
        )
        if len(token_ids) != 1:
            raise ValueError(
                f"Probe target {text!r} must encode to exactly one token; "
                f"found {len(token_ids)}."
            )
        return token_ids[0]

    checkpoint_path = os.environ.get(
        "MODEL_PATH",
        os.path.join(os.path.dirname(__file__), "..", "checkpoints", "baseline_verified_parity_hardened.pt"),
    ).strip()
    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"Loading checkpoint: {checkpoint_path}")
        model = load_model(model_path=checkpoint_path, tiny=True).to(device)
    else:
        print(
            f"Checkpoint not found at {checkpoint_path or '<unset>'}. "
            "Running the probe on random tiny weights; treat results as structural only."
        )
        model = load_model(tiny=True).to(device)

    model.eval()

    test_cases = [
        {
            "question": "What color is the sky on Planet Zephyria?",
            "context": "[FACT-001] The sky on Planet Zephyria is purple.",
            "target_token": "purple",
            "target_id": single_token_id("purple")
        },
        {
            "question": "What color is the grass on Planet Zephyria?",
            "context": "[FACT-002] Grass on Planet Zephyria is blue.",
            "target_token": "blue",
            "target_id": single_token_id("blue")
        }
    ]

    print("\n" + "="*60)
    print(f"{'TEST CASE':<40} | {'BASELINE':<10} | {'RAG':<10} | {'RATIO'}")
    print("-" * 75)

    for case in test_cases:
        # Run 1: Baseline (No context)
        prob_baseline = get_token_prob(
            model, tokenizer, case["question"], None, case["target_id"], device
        )

        # Run 2: RAG (With context)
        prob_rag = get_token_prob(
            model, tokenizer, case["question"], case["context"], case["target_id"], device
        )

        ratio = prob_rag / prob_baseline if prob_baseline > 0 else float('inf')

        print(f"{case['target_token']:<40} | {prob_baseline:.2e} | {prob_rag:.2e} | {ratio:.2f}x")

    print("="*60)
    print("\nINTERPRETATION:")
    print("Ratio > 1.0 : Attention is moving in the right direction.")
    print("Ratio > 5.0 : Strong signal that GQA heads are capturing the context.")
    print("Ratio < 1.0 : Model is ignoring context or being penalised by it.")

if __name__ == "__main__":
    run_logit_probe()
