import re
from typing import Optional

PROMPT_VERSION = "v1_parity_hardened"

# Exact neutral system prompt requested by the user
DEFAULT_SYSTEM_PROMPT = "You are a precise technical assistant. Follow instructions carefully and respond clearly."

def strip_raw_tags(text: str) -> str:
    """
    Safely clean old formatting and prevent double-wrapping bugs.
    Removes <|user|> and <|assistant|> tags and associated surrounding whitespace.
    """
    if not text:
        return ""
        
    cleaned = text
    # Remove <|user|> and <|assistant|> tags
    cleaned = re.sub(r"<\|user\|>\s*", "", cleaned)
    cleaned = re.sub(r"\s*<\|assistant\|>", "", cleaned)
    
    return cleaned.strip()

def build_prompt(
    question: str, 
    context: Optional[str] = None, 
    system_prompt: Optional[str] = DEFAULT_SYSTEM_PROMPT
) -> str:
    """
    Single authoritative prompt builder.
    Ensures deterministic spacing, newline placement, and section ordering.
    """
    # 1. Clean inputs to prevent double-wrapping
    clean_question = strip_raw_tags(question)
    
    parts = []
    
    # 2. System Prompt
    if system_prompt:
        parts.append(system_prompt.strip())
        
    # 3. RAG Context Block
    if context:
        # Note: If context has baked-in tags, they aren't stripped here by default 
        # unless explicitly requested. However, RAG context usually doesn't have tags.
        # But we ensure consistent formatting.
        clean_context = context.replace("--- RETRIEVED CONTEXT ---", "").replace("--- END CONTEXT ---", "").strip()
        parts.append("--- RETRIEVED CONTEXT ---")
        parts.append(clean_context)
        parts.append("--- END CONTEXT ---")
        
    # 4. User Question Block
    parts.append("<|user|>")
    parts.append(clean_question)
    parts.append("<|assistant|>")
    parts.append("") # Ensures the final string ends with \n
    
    # Join with explicit single newlines
    return "\n".join(parts)
