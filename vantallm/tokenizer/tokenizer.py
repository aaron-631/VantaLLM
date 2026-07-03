"""
tokenizer.py — Token & Output Engineer
=================================================
Production-grade Cybersecurity BPE Tokenizer for the VantaLM architecture.
"""

import os
import logging
import torch

log = logging.getLogger(__name__)

# ── Constants (must match LLMConfig in transformer.py) ───────────────────────────
VOCAB_SIZE    = 151_003
EOS_TOKEN_ID  = 1        # Matches "<eos>": 1 in vocab.json
BOS_TOKEN_ID  = 2        # Matches "<bos>": 2 in vocab.json
PAD_TOKEN_ID  = 0
MAX_INPUT_LEN = 131_072  # 128k context window

DEFAULT_TOKENIZER_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "cybersec_tokenizer"
)

class CybersecBPETokenizer:
    def __init__(self, tokenizer_dir: str = DEFAULT_TOKENIZER_DIR):
        from tokenizers import ByteLevelBPETokenizer
        
        vocab_path  = os.path.join(tokenizer_dir, "vocab.json")
        merges_path = os.path.join(tokenizer_dir, "merges.txt")

        if not os.path.exists(vocab_path):
            raise FileNotFoundError(f"vocab.json not found in {tokenizer_dir}")
        if not os.path.exists(merges_path):
            raise FileNotFoundError(f"merges.txt not found in {tokenizer_dir}")

        # Proper ByteLevel configuration
        self._tokenizer = ByteLevelBPETokenizer(
            vocab_path,
            merges_path,
            add_prefix_space=True
        )

        # Register special architectural markers
        self.special_tokens = [
            "<pad>", "<eos>", "<bos>", "<unk>", "<mask>",
            "<|user|>", "<|assistant|>", "<|system|>"
        ]
        self._tokenizer.add_special_tokens(self.special_tokens)

        self.vocab_size   = self._tokenizer.get_vocab_size()
        self.eos_token_id = EOS_TOKEN_ID
        self.bos_token_id = BOS_TOKEN_ID
        self.pad_token_id = PAD_TOKEN_ID

        # Map out special token IDs for absolute lookup speeds during filtering
        self.special_token_ids = {self.pad_token_id, self.eos_token_id, self.bos_token_id}
        for token in self.special_tokens:
            token_id = self._tokenizer.token_to_id(token)
            if token_id is not None:
                self.special_token_ids.add(token_id)

        assert self._tokenizer.token_to_id("<pad>") == PAD_TOKEN_ID, "PAD token ID mismatch"
        assert self._tokenizer.token_to_id("<eos>") == EOS_TOKEN_ID, "EOS token ID mismatch"
        assert self._tokenizer.token_to_id("<bos>") == BOS_TOKEN_ID, "BOS token ID mismatch"

        log.info(f"CybersecBPETokenizer loaded safely | vocab_size={self.vocab_size:,}")

    def encode(
        self,
        text: str,
        return_tensors: str = "pt",
        add_special_tokens: bool = True,
        **kwargs
    ):
        encoded = self._tokenizer.encode(text)
        ids = encoded.ids

        # Enforce max length context matrix boundaries safely
        if len(ids) > MAX_INPUT_LEN - 2:
            ids = ids[:MAX_INPUT_LEN - 2]

        if add_special_tokens:
            ids = [self.bos_token_id] + ids + [self.eos_token_id]

        if return_tensors == "pt":
            return torch.tensor([ids], dtype=torch.long)
        return ids

    def __call__(self, text: str, return_tensors: str = "pt", **kwargs):
        return self.encode(text, return_tensors=return_tensors, **kwargs)

    def decode(
        self, 
        token_ids, 
        skip_special_tokens: bool = False, 
        **kwargs
    ):
        # Flatten tensors automatically if passed directly from inference slices
        if isinstance(token_ids, torch.Tensor):
            ids = token_ids.flatten().tolist()
        elif isinstance(token_ids, list):
            ids = token_ids
        else:
            ids = [int(token_ids)]

        # Dynamically drop structural markers based on UI needs
        if skip_special_tokens:
            ids = [i for i in ids if i not in self.special_token_ids]
        else:
            # ORIGINAL BEHAVIOR: Always strip base structural tokens to clean UI
            ids = [i for i in ids if i not in (self.pad_token_id, self.eos_token_id, self.bos_token_id)]

        return self._tokenizer.decode(ids)

    def batch_decode(self, batch_token_ids, skip_special_tokens: bool = False, **kwargs):
        return [self.decode(ids, skip_special_tokens=skip_special_tokens, **kwargs) for ids in batch_token_ids]

    def __len__(self):
        return self.vocab_size

    def __repr__(self):
        return f"CybersecBPETokenizer(vocab_size={self.vocab_size:,})"


class MockTokenizer:
    vocab_size   = VOCAB_SIZE
    eos_token_id = EOS_TOKEN_ID
    pad_token_id = PAD_TOKEN_ID
    bos_token_id = BOS_TOKEN_ID

    def encode(self, text: str, return_tensors: str = "pt", add_special_tokens: bool = True, **kwargs):
        ids = [ord(c) % self.vocab_size for c in text[:MAX_INPUT_LEN - 2]]
        if add_special_tokens:
            ids = [self.bos_token_id] + ids + [self.eos_token_id]
        if return_tensors == "pt":
            return torch.tensor([ids], dtype=torch.long)
        return ids

    def __call__(self, text: str, return_tensors: str = "pt", **kwargs):
        return self.encode(text, return_tensors=return_tensors, **kwargs)

    def decode(self, token_ids, skip_special_tokens: bool = False, **kwargs):
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.flatten().tolist()
        return f"[MOCK OUTPUT — {len(token_ids)} tokens]"

    def batch_decode(self, batch_token_ids, skip_special_tokens: bool = False, **kwargs):
        return [self.decode(ids, skip_special_tokens=skip_special_tokens, **kwargs) for ids in batch_token_ids]

    def __len__(self):
        return self.vocab_size


def load_tokenizer():
    tokenizer_dir = os.environ.get("TOKENIZER_PATH", DEFAULT_TOKENIZER_DIR).strip()
    vocab_path  = os.path.join(tokenizer_dir, "vocab.json")
    merges_path = os.path.join(tokenizer_dir, "merges.txt")

    if os.path.exists(vocab_path) and os.path.exists(merges_path):
        try:
            return CybersecBPETokenizer(tokenizer_dir)
        except Exception as e:
            log.warning(f"Failed to load tokenizer: {e}")

    log.warning("Using MockTokenizer (NOT for production)")
    return MockTokenizer()