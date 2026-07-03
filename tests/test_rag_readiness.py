"""
tests/test_rag_readiness.py
===========================

Purpose:
  A surgical, minimal test to validate that the model can prioritize
  information from supplied context over its own internal (and likely incorrect)
  statistical priors. This is a foundational test for Retrieval-Augmented
  Generation (RAG) readiness.

Method:
  We use a "contradiction" prompt. The model's internal knowledge, based
  on its training data (e.g., The sky is blue), is contradicted by
  explicit facts provided in the prompt's context.

  The test evaluates if the model's output repeats the *contextual* fact,
  not its pre-existing "knowledge".

Goal:
  Verify: "Instruction → Attention to Context → Answer"
  Not:   "Instruction → Best guess from internal stats"
"""
import os
import sys
import torch

TESTS_DIR = os.path.abspath(os.path.dirname(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(TESTS_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Ensure TINY_MODEL is set for CPU-based testing, consistent with project conventions
os.environ["TINY_MODEL"] = "1"

# Reuse existing project components for loading model and tokenizer
from vantallm.model.transformer import load_model
from vantallm.tokenizer.tokenizer import load_tokenizer


from vantallm.utils.prompting import build_prompt, PROMPT_VERSION

def encode_rag_prompt(tokenizer, retrieved_context: str, question: str, device):
    prompt_str = build_prompt(question=question, context=retrieved_context)
    input_ids = tokenizer.encode(prompt_str, return_tensors="pt").to(device)
    if input_ids[0, -1].item() == tokenizer.eos_token_id:
        input_ids = input_ids[:, :-1]
    return input_ids


@torch.no_grad()
def run_rag_test():
    """
    Loads the model and tokenizer, then runs the contradiction test.
    """
    # 1. Setup Environment
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. Load Tokenizer
    # Assuming this script is run from the root of the VantaLLM repository
    tokenizer_dir = os.path.join(os.path.dirname(__file__), '..', 'vantallm', 'tokenizer', 'cybersec_tokenizer')
    os.environ['TOKENIZER_PATH'] = tokenizer_dir
    tokenizer = load_tokenizer()
    print(f"Tokenizer loaded. Vocab size: {tokenizer.vocab_size}")

    print(f"Prompt Utility Version: {PROMPT_VERSION}")

    # 3. Load Model
    # The `load_model` function respects the TINY_MODEL=1 env var
    # Load the latest checkpoint
    checkpoint_path = os.path.join(os.path.dirname(__file__), "..", "checkpoints", "vanta_phase1.pt")
    if not os.path.exists(checkpoint_path):
        print(f"ERROR: Checkpoint not found at {checkpoint_path}")
        print("Please ensure you have a trained checkpoint file at that location.")
        return

    print(f"Loading checkpoint from: {checkpoint_path}")
    model = load_model(model_path=checkpoint_path, tiny=True).to(device)
    model.eval()
    print("Model and checkpoint loaded successfully.")

    # 4. Define Test Cases
    retrieved_context = """[FACT-001] The sky on Planet Zephyria is purple.
[FACT-002] Grass on Planet Zephyria is blue."""

    questions = [
        "What color is the sky on Planet Zephyria?",
        "What color is the grass on Planet Zephyria?",
    ]

    # 5. Run Generation
    for question in questions:
        print()
        print("="*50)
        print(f"TESTING QUESTION: {question}")
        print("="*50)

        prompt = build_prompt(question=question, context=retrieved_context)
        input_ids = encode_rag_prompt(tokenizer, retrieved_context, question, device)

        print("PROMPT:")
        print(prompt)

        # Generation loop (greedy decoding)
        generated_ids = input_ids
        eos_token_id = getattr(tokenizer, "eos_token_id", 1)
        max_new_tokens = 30

        for _ in range(max_new_tokens):
            logits, _, _ = model(generated_ids)
            next_token_id = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            generated_ids = torch.cat([generated_ids, next_token_id], dim=1)

            if next_token_id.item() == eos_token_id:
                break
        
        output_ids = generated_ids[:, input_ids.shape[1]:]

        print()
        print(f"GENERATED TOKEN IDs: {output_ids.tolist()}")
        
        decoded_output = tokenizer.decode(output_ids[0])
        print(f"DECODED OUTPUT: {decoded_output}")
        print("="*50)
        print()


if __name__ == "__main__":
    run_rag_test()
