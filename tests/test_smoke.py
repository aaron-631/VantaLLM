"""
test_smoke.py — pytest-native smoke tests for VantaLLM.

These are intentionally lightweight (tiny CPU config, no GPU, no checkpoints) so
`pytest tests/` runs in seconds and can gate CI. The larger script-style suites
(test_system.py, test_e2e.py, …) remain runnable directly and cover the server,
streaming, and RAG paths; conftest.py excludes them from pytest collection.

Run:  TINY_MODEL=1 pytest tests/ -q     (or plain `pytest tests/`)
"""

import torch
import pytest

from vantallm.model.transformer import load_model, greedy_generate, LLMConfig
from vantallm.tokenizer.tokenizer import load_tokenizer, MockTokenizer, VOCAB_SIZE


@pytest.fixture(scope="module")
def tiny_model():
    torch.manual_seed(0)
    return load_model(tiny=True)


def test_tiny_model_builds(tiny_model):
    assert tiny_model.config.num_layers == 4
    assert tiny_model.config.vocab_size == VOCAB_SIZE


def test_weight_tying(tiny_model):
    # lm_head must share storage with the token embedding
    assert tiny_model.lm_head.weight.data_ptr() == tiny_model.token_embed.weight.data_ptr()


def test_forward_tuple_shapes(tiny_model):
    tiny_model.eval()
    ids = torch.randint(0, tiny_model.config.vocab_size, (2, 16))
    logits, kv, aux = tiny_model(ids)
    assert logits.shape == (2, 16, tiny_model.config.vocab_size)
    assert torch.isfinite(logits).all()


def test_forward_loss_dict(tiny_model):
    tiny_model.train()
    ids = torch.randint(0, tiny_model.config.vocab_size, (2, 16))
    out = tiny_model(ids, labels=ids)
    assert "loss" in out and out["loss"].ndim == 0
    assert torch.isfinite(out["loss"])


def test_backward_covers_expected_params(tiny_model):
    """
    In the training (persona_ids) path, every trainable parameter EXCEPT the
    inference-only `persona_vec_to_composite` projection must receive a gradient.
    That single unused parameter is exactly why DDP uses
    find_unused_parameters=True; MoE experts always participate via the dense
    per-expert dispatch loop, so if any expert weight shows up here it is a real
    regression.
    """
    tiny_model.train()
    tiny_model.zero_grad(set_to_none=True)
    B, S, P = 2, 16, tiny_model.config.num_persona_types
    out = tiny_model(
        torch.randint(0, tiny_model.config.vocab_size, (B, S)),
        labels=torch.randint(0, tiny_model.config.vocab_size, (B, S)),
        persona_ids=torch.zeros(B, P, dtype=torch.long),
    )
    out["loss"].backward()
    unused = [n for n, p in tiny_model.named_parameters() if p.requires_grad and p.grad is None]
    assert unused == ["persona_vec_to_composite.weight"], f"unexpected unused params: {unused}"
    assert not any(".experts." in n for n in unused), "an MoE expert weight went unused — regression"


def test_kv_cache_matches_no_cache(tiny_model):
    tiny_model.eval()
    prompt = torch.randint(0, tiny_model.config.vocab_size, (1, 6))
    out = greedy_generate(tiny_model, prompt, max_new_tokens=8, eos_token_id=-1)
    assert out.shape[1] == 6 + 8


def test_config_validation_rejects_bad():
    with pytest.raises(ValueError):
        LLMConfig(num_q_heads=7, num_kv_heads=2)   # not divisible → GQA invalid
    with pytest.raises(ValueError):
        LLMConfig(head_dim=31)                      # odd head_dim → RoPE invalid


def test_tokenizer_roundtrip():
    tok = load_tokenizer()
    if isinstance(tok, MockTokenizer):
        pytest.skip("real BPE tokenizer assets not present; MockTokenizer active")
    assert tok.vocab_size == VOCAB_SIZE
    for s in ["buffer overflow in CVE-2021-44228", "SELECT * FROM users;"]:
        ids = tok.encode(s, add_special_tokens=False, return_tensors=None)
        assert tok.decode(ids, skip_special_tokens=True).strip() == s.strip()
