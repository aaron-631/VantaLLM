import os
import sys
import torch
import logging

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from vantallm.model.transformer import load_model, greedy_generate
from vantallm.utils.prompting import build_prompt, PROMPT_VERSION
from vantallm.tokenizer.tokenizer import load_tokenizer

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
log = logging.getLogger("evaluator")

def evaluate_model():
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = load_tokenizer()
    
    # 1. Load the trained Showcase Tiny Model if available, otherwise fall back
    # to a random tiny model so the evaluation script still runs on a clean clone.
    os.environ["TINY_MODEL"] = "1"
    checkpoint_path = os.environ.get(
        "MODEL_PATH",
        os.path.join(os.path.dirname(__file__), "../../checkpoints/showcase_run.pt"),
    ).strip()

    if checkpoint_path and os.path.exists(checkpoint_path):
        log.info(f"Loading checkpoint from {checkpoint_path}")
        model = load_model(model_path=checkpoint_path, tiny=True).to(device)
    else:
        log.warning(
            f"Checkpoint not found at {checkpoint_path or '<unset>'}. "
            "Running manual evaluation with random tiny weights."
        )
        model = load_model(tiny=True).to(device)
    model.eval()
    
    # 2. Define Manual Test Prompts
    instructions = [
        "Write a Python function for factorial.",
        "Reverse a string in Python.",
        "Explain binary search."
    ]
    
    log.info("\n========================================")
    log.info(f" PHASE 2 MANUAL EVALUATION ({PROMPT_VERSION})")
    log.info("========================================\n")
    
    for inst in instructions:
        prompt_text = build_prompt(question=inst)
        log.info(f"PROMPT:\n{prompt_text}")
        
        # Strip prompt-end EOS to avoid premature model stop
        prompt_ids = tokenizer.encode(prompt_text).to(device)
        if prompt_ids[0, -1].item() == tokenizer.eos_token_id:
            prompt_ids = prompt_ids[:, :-1]
        
        with torch.no_grad():
            generated_ids = greedy_generate(model, prompt_ids, max_new_tokens=40)
            
        # Slicing to get only the NEW generated tokens
        output_ids = generated_ids[:, prompt_ids.shape[1]:]
        output_text = tokenizer.decode(output_ids[0])
        
        log.info(f"\nRAW TOKENS GENERATED: {output_ids[0].tolist()}")
        log.info(f"MODEL OUTPUT:\n{output_text}")
        log.info("-" * 40 + "\n")

if __name__ == "__main__":
    evaluate_model()
