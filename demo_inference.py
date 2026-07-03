import os
import argparse
import torch
from vantallm.model.transformer import load_model, greedy_generate
from vantallm.tokenizer.tokenizer import load_tokenizer
from vantallm.utils.prompting import build_prompt

def main():
    parser = argparse.ArgumentParser(description="VantaLLM Minimal Inference Demo")
    parser.add_argument("--prompt", type=str, default="What are the fundamental principles of language modeling?", help="Input prompt text")
    parser.add_argument("--model-path", type=str, default=None, help="Path to checkpoint .pt file (if None, uses untrained weights)")
    parser.add_argument("--max-new-tokens", type=int, default=50, help="Maximum number of tokens to generate")
    args = parser.parse_args()

    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Initializing inference on device: {device}")

    # Load tokenizer
    print("[*] Loading tokenizer...")
    tokenizer = load_tokenizer()

    # Load model
    print(f"[*] Loading model (checkpoint: {args.model_path if args.model_path else 'Untrained/Random'})...")
    model = load_model(model_path=args.model_path).to(device)
    
    # Use bfloat16 for GPU inference if available
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        model = model.to(torch.bfloat16)
        
    model.eval()

    # Format the prompt
    formatted_prompt = build_prompt(question=args.prompt)
    print(f"\n[*] Prompt:\n{formatted_prompt}")

    # Tokenize and move to device
    ids = tokenizer.encode(formatted_prompt).to(device)
    
    # Optional: remove EOS if present depending on tokenizer behavior
    if ids.dim() == 2 and ids[0, -1] == tokenizer.eos_token_id:
        ids = ids[:, :-1]

    # Generate
    print("[*] Generating...")
    with torch.no_grad():
        output_ids = greedy_generate(model, ids, max_new_tokens=args.max_new_tokens)
    
    # Decode only the newly generated tokens
    input_length = ids.shape[1]
    output_text = tokenizer.decode(output_ids[0, input_length:])
    
    print("\n--- Output ---")
    print(output_text)
    print("--------------\n")

if __name__ == "__main__":
    main()
