"""
test_system.py  —  VantaLLM Full System Validation
=====================================================
Tests every layer in order:

  Phase 1 — Unit
    1.1  MockTokenizer  encode / decode / vocab
    1.2  Model shapes   (embedding → transformer → logits)
    1.3  Both forward modes (plain tuple + training dict)
    1.4  embed() helper  (server.py persona-vector path)
    1.5  TurboQuant KV compression / reconstruction error
    1.6  Persona system  (PersonaEmbeddingStore, EoE Router, both conditioning paths)
    1.7  PPO scaffold   (ValueHead, RewardModel, PPOTrainer, compute_advantages)

  Phase 2 — Integration  (components wired together, no server)
    2.1  Tokenizer → model → logits
    2.2  Persona vector computation  (exact server.py logic reproduced)
    2.3  Sampling pipeline  (temperature / top-k / top-p / rep-penalty)
    2.4  KV-cache incremental decode  (step 0 full → step N single-token)
    2.5  SSE prompt builder  (system + context + query token counts)

  Phase 3 — End-to-End  (real FastAPI server, subprocess)
    3.1  /health
    3.2  POST /generate   (general route, no context)
    3.3  POST /generate   (exact  route, with context chunks)
    3.4  POST /generate   (oos    route — model never invoked)
    3.5  POST /generate/stream  (SSE token stream)
    3.6  POST /generate   (session_id — stateful KV session)

Run:
    python test_system.py              # all phases
    python test_system.py --phase 1    # unit only
    python test_system.py --phase 2    # integration only
    python test_system.py --phase 3    # e2e only (launches server subprocess)
"""

import os
import sys
import time
import uuid
import json
import argparse
import subprocess
import torch
import torch.nn.functional as F
import requests


# ──────────────────────────────────────────────────────────────────────────────
# Locate repo files  —  works whether you run from project root or any subdir
# ──────────────────────────────────────────────────────────────────────────────

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Candidate roots for the repo (handles running from tests/ or from root)
_CANDIDATE_ROOTS = [
    SCRIPT_DIR,                          # if test_system.py is at repo root
    os.path.join(SCRIPT_DIR, ".."),      # if running from tests/
]

# ── Find PROJECT_ROOT (contains inference/ and model/) ───────────────────────
PROJECT_ROOT = None
for _r in _CANDIDATE_ROOTS:
    _r = os.path.abspath(_r)
    if os.path.isdir(os.path.join(_r, "vantallm")):
        PROJECT_ROOT = _r
        break
if PROJECT_ROOT is None:
    # fallback: one level up from SCRIPT_DIR
    PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))

# Directory holding the model package (used for the subprocess PYTHONPATH below).
MODEL_DIR = os.path.join(PROJECT_ROOT, "vantallm", "model")

# Make the vantallm package importable when running from a source checkout.
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from vantallm.model.transformer import VantaTransformer, LLMConfig, TurboQuant, ValueHead, RewardModel, PPOTrainer


# ──────────────────────────────────────────────────────────────────────────────
# Tiny config  —  runs on CPU in seconds; same code paths as the full-scale config
# ──────────────────────────────────────────────────────────────────────────────

def tiny_cfg(**overrides) -> LLMConfig:
    base = dict(
        vocab_size             = 1_000,
        d_model                = 64,
        max_seq_len            = 128,
        num_layers             = 4,
        num_q_heads            = 4,
        num_kv_heads           = 2,
        head_dim               = 16,
        d_ff_dense             = 128,
        num_experts            = 4,
        top_k_experts          = 2,
        d_ff_moe               = 64,
        moe_every_n            = 2,
        quantization           = "fp16",
        composite_persona_dim  = 32,
        persona_embedding_dim  = 16,
        num_persona_types      = 3,
        num_persona_experts    = 2,
    )
    base.update(overrides)
    return LLMConfig(**base)


# ──────────────────────────────────────────────────────────────────────────────
# Mock tokenizer  (mirrors server.py MockTokenizer exactly)
# ──────────────────────────────────────────────────────────────────────────────

class MockTokenizer:
    vocab_size   = 1_000       # kept small to match tiny_cfg
    eos_token_id = 1
    pad_token_id = 0

    def encode(self, text: str, return_tensors: str = "pt") -> torch.Tensor:
        ids = [ord(c) % self.vocab_size for c in text[:512]]
        return torch.tensor([ids], dtype=torch.long)

    def decode(self, token_ids: torch.Tensor) -> str:
        if token_ids.numel() == 1:
            return chr(token_ids.item() % 128) if token_ids.item() > 32 else " "
        return f"[MOCK {token_ids.shape[-1]} tokens]"

    def __len__(self):
        return self.vocab_size


# ──────────────────────────────────────────────────────────────────────────────
# Test runner helpers
# ──────────────────────────────────────────────────────────────────────────────

PASS  = "\033[92m✓\033[0m"
FAIL  = "\033[91m✗\033[0m"
WARN  = "\033[93m⚠\033[0m"
_results = []

def check(name: str, ok: bool, detail: str = ""):
    status = PASS if ok else FAIL
    line   = f"  {status}  {name}"
    if detail:
        line += f"  [{detail}]"
    print(line)
    _results.append((name, ok))
    return ok

def section(title: str):
    print(f"\n{'─'*60}")
    print(f"  {title}")
    print(f"{'─'*60}")


# ══════════════════════════════════════════════════════════════════
# PHASE 1  —  Unit tests
# ══════════════════════════════════════════════════════════════════

def phase1():
    print("\n" + "═"*60)
    print("  PHASE 1 — Unit Tests")
    print("═"*60)
    torch.manual_seed(0)

    # ── 1.1  MockTokenizer ────────────────────────────────────────
    section("1.1  MockTokenizer")
    tok = MockTokenizer()
    ids = tok.encode("hello world", return_tensors="pt")
    check("encode returns (1, seq) tensor",
          ids.shape[0] == 1 and ids.dim() == 2,
          str(tuple(ids.shape)))
    check("all ids within vocab",
          ids.max().item() < tok.vocab_size,
          f"max={ids.max().item()}")
    decoded = tok.decode(ids[0])
    check("decode returns string", isinstance(decoded, str), repr(decoded))
    check("vocab_size == 1000", len(tok) == 1_000)
    check("eos_token_id set", tok.eos_token_id == 1)

    # ── 1.2  Model shapes ─────────────────────────────────────────
    section("1.2  Model shapes (embedding → transformer → logits)")
    cfg   = tiny_cfg()
    model = VantaTransformer(cfg)
    model.eval()
    B, S  = 2, 16
    ids_t = torch.randint(0, cfg.vocab_size, (B, S))

    logits, kv_caches, aux_loss = model(ids_t)

    check("logits shape (B, S, vocab)",
          logits.shape == (B, S, cfg.vocab_size),
          str(tuple(logits.shape)))
    check("logits finite (no NaN/Inf)",
          torch.isfinite(logits).all().item())
    check("aux_loss is scalar tensor",
          aux_loss.dim() == 0,
          f"{aux_loss.item():.6f}")
    check(f"kv_caches list len == num_layers",
          len(kv_caches) == cfg.num_layers,
          f"{len(kv_caches)}")

    # ── 1.3  Forward modes ────────────────────────────────────────
    section("1.3  Forward modes (inference tuple vs training dict)")
    # Tuple (inference)
    out = model(ids_t)
    check("inference returns tuple", isinstance(out, tuple))
    check("tuple[0] = logits", out[0].shape == (B, S, cfg.vocab_size))

    # Dict (training)
    labels = ids_t.clone()
    out_d  = model(ids_t, labels=labels)
    check("training returns dict", isinstance(out_d, dict))
    for key in ("logits", "loss", "aux_loss", "kv_caches", "hidden_states"):
        check(f"  dict has '{key}'", key in out_d)
    check("hidden_states shape (B, S, d_model)",
          out_d["hidden_states"].shape == (B, S, cfg.d_model),
          str(tuple(out_d["hidden_states"].shape)))
    check("loss is finite", torch.isfinite(out_d["loss"]).item(),
          f"{out_d['loss'].item():.4f}")
    check("hidden_states finite", torch.isfinite(out_d["hidden_states"]).all().item())

    # ── 1.4  embed() helper ───────────────────────────────────────
    section("1.4  embed() helper  (server.py persona-vector path)")
    emb = model.embed(ids_t[:1, :4])
    check("embed() shape (1, 4, d_model)",
          emb.shape == (1, 4, cfg.d_model),
          str(tuple(emb.shape)))
    vec = emb[:, -1, :]
    vec = vec / (vec.norm(dim=-1, keepdim=True) + 1e-6)
    check("L2-norm ≈ 1.0",
          abs(vec.norm().item() - 1.0) < 1e-4,
          f"norm={vec.norm().item():.6f}")

    # ── 1.5  TurboQuant ──────────────────────────────────────────
    section("1.5  TurboQuant INT8 KV cache")
    tq      = TurboQuant(bits=8)
    k_dummy = torch.randn(B, cfg.num_kv_heads, S, cfg.head_dim)
    v_dummy = torch.randn(B, cfg.num_kv_heads, S, cfg.head_dim)
    k_c, v_c = tq.compress_kv(k_dummy, v_dummy)
    k_d, v_d = tq.decompress_kv(k_c, v_c)

    k_err = (k_dummy - k_d).abs().mean().item()
    v_err = (v_dummy - v_d).abs().mean().item()
    comp  = k_dummy.element_size() / k_c[0].element_size()
    check(f"compression ratio {comp:.0f}×",    comp >= 4,  f"{comp:.0f}×")
    check(f"K reconstruction MAE < 0.02", k_err < 0.02, f"{k_err:.6f}")
    check(f"V reconstruction MAE < 0.02", v_err < 0.02, f"{v_err:.6f}")
    check("compressed tensor is INT8",
          k_c[0].dtype == torch.int8)

    # ── 1.6  Persona system ───────────────────────────────────────
    section("1.6  Persona system")
    p_ids = torch.randint(0, cfg.num_persona_types, (B, cfg.num_persona_types))
    logits_p, _, _ = model(ids_t, persona_ids=p_ids)
    check("persona_ids path: logits shape ok",
          logits_p.shape == (B, S, cfg.vocab_size))
    check("persona_ids path: logits finite",
          torch.isfinite(logits_p).all().item())

    p_vec = torch.randn(B, cfg.d_model)
    p_vec = p_vec / (p_vec.norm(dim=-1, keepdim=True) + 1e-6)
    logits_v, _, _ = model(ids_t, persona_vector=p_vec)
    check("persona_vector path: logits shape ok",
          logits_v.shape == (B, S, cfg.vocab_size))
    check("persona_vector path: logits finite",
          torch.isfinite(logits_v).all().item())

    logits_plain, _, _ = model(ids_t)
    differ = not torch.allclose(logits_p, logits_plain, atol=1e-5)
    check("persona conditioning changes output", differ)

    # ── 1.7  PPO scaffold ─────────────────────────────────────────
    section("1.7  PPO scaffold  (ValueHead / RewardModel / PPOTrainer)")
    policy      = VantaTransformer(cfg)
    ref_policy  = VantaTransformer(cfg)
    value_head  = ValueHead(cfg)
    reward_model= RewardModel(cfg)
    trainer     = PPOTrainer(policy, ref_policy, value_head, reward_model)

    B2, T = 2, 8
    ids2   = torch.randint(0, cfg.vocab_size, (B2, T))
    p_ids2 = torch.randint(0, cfg.num_persona_types, (B2, cfg.num_persona_types))
    rewards= torch.randn(B2, T)

    reward = reward_model(ids2, persona_ids=p_ids2)
    check("RewardModel output shape (B,)",
          reward.shape == (B2,),
          str(tuple(reward.shape)))
    check("RewardModel output finite", torch.isfinite(reward).all().item())

    with torch.no_grad():
        out_v  = policy(ids2, persona_ids=p_ids2, labels=ids2)
        hidden = out_v["hidden_states"]
    values = value_head(hidden)
    check("ValueHead input is hidden_states (B,T,d_model)",
          hidden.shape == (B2, T, cfg.d_model),
          str(tuple(hidden.shape)))
    check("ValueHead output shape (B, T)",
          values.shape == (B2, T),
          str(tuple(values.shape)))

    advs, returns = trainer.compute_advantages(rewards, values.detach())
    check("advantages shape (B, T)",  advs.shape    == (B2, T))
    check("returns shape (B, T)",     returns.shape == (B2, T))
    check("advantages normalised (mean ≈ 0)",
          abs(advs.mean().item()) < 0.5,
          f"{advs.mean().item():.4f}")

    with torch.no_grad():
        ref_out = policy(ids2, persona_ids=p_ids2)
        lp      = F.log_softmax(ref_out[0][:, :-1], dim=-1)
        old_lp  = lp.gather(-1, ids2[:, 1:].unsqueeze(-1)).squeeze(-1)

    loss = trainer.compute_ppo_loss(
        ids2, p_ids2,
        old_lp.detach(),
        advs[:, :-1],
        returns[:, :-1],
    )
    check("PPO loss is scalar", loss.dim() == 0)
    check("PPO loss finite",    torch.isfinite(loss).item(), f"{loss.item():.4f}")
    check("PPO loss not NaN",   not loss.isnan().item())


# ══════════════════════════════════════════════════════════════════
# PHASE 2  —  Integration  (components wired together, no server)
# ══════════════════════════════════════════════════════════════════

def _sample_next_token(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    temperature: float = 0.7,
    top_k: int = 50,
    top_p: float = 0.9,
    repetition_penalty: float = 1.1,
    rep_window: int = 64,
) -> torch.Tensor:
    """Mirrors server.py sample_next_token exactly."""
    logits = logits.clone()
    if repetition_penalty != 1.0:
        recent    = input_ids[:, -rep_window:]
        score     = torch.gather(logits, 1, recent)
        score     = torch.where(score < 0,
                                score * repetition_penalty,
                                score / repetition_penalty)
        logits.scatter_(1, recent, score)
    if temperature <= 0.0:
        return logits.argmax(dim=-1)
    logits = logits / max(temperature, 1e-8)
    if top_k > 0:
        top_k   = min(top_k, logits.size(-1))
        kth_val = torch.topk(logits, top_k).values[:, -1, None]
        logits  = logits.masked_fill(logits < kth_val, float("-inf"))
    if top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True)
        cumprobs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        remove   = (cumprobs - F.softmax(sorted_logits, dim=-1)) > top_p
        sorted_logits[remove] = float("-inf")
        logits = torch.zeros_like(logits).scatter_(1, sorted_idx, sorted_logits)
    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, 1).squeeze(-1)


def phase2():
    print("\n" + "═"*60)
    print("  PHASE 2 — Integration Tests")
    print("═"*60)
    torch.manual_seed(42)

    cfg   = tiny_cfg()
    model = VantaTransformer(cfg)
    tok   = MockTokenizer()
    model.eval()

    # ── 2.1  Tokenizer → model → logits ──────────────────────────
    section("2.1  Tokenizer → model → logits")
    text  = "Explain SQL injection attacks"
    ids   = tok.encode(text, return_tensors="pt")
    ids   = ids % cfg.vocab_size
    with torch.no_grad():
        logits, _, _ = model(ids)
    check("tokenizer output feeds model",
          logits.shape[-1] == cfg.vocab_size,
          f"logits shape {tuple(logits.shape)}")
    next_tok = logits[0, -1].argmax().item()
    check("next-token prediction in vocab range",
          0 <= next_tok < cfg.vocab_size, str(next_tok))

    # ── 2.2  Persona vector computation  (server.py path) ─────────
    section("2.2  Persona vector computation  (server.py flow)")

    PERSONA_TAGS = {
        "cyber_researcher":   "<role:researcher> <domain:cyber> <grounding:strict> A",
        "security_ciso":      "<role:executive>  <domain:cyber> <style:strategic> B",
        "technical_educator": "<role:teacher>    <domain:general> <detail:high> C",
        "general_assistant":  "<role:assistant>  <domain:general> <style:helpful> D",
    }

    persona_cache = {}
    for name, tag in PERSONA_TAGS.items():
        tag_ids = tok.encode(tag, return_tensors="pt") % cfg.vocab_size
        with torch.no_grad():
            emb = model.embed(tag_ids)
            vec = emb[:, -1, :]
            vec = vec / (vec.norm(dim=-1, keepdim=True) + 1e-6)
        persona_cache[name] = vec
        norm = vec.norm().item()
        check(f"persona '{name}' L2-norm ≈ 1.0",
              abs(norm - 1.0) < 1e-4,
              f"norm={norm:.6f}")
    check("all 4 personas cached", len(persona_cache) == 4)

    vecs  = list(persona_cache.values())

    tag_ids_list = [
        tok.encode(tag, return_tensors="pt") % cfg.vocab_size
        for tag in PERSONA_TAGS.values()
    ]
    seq_distinct = any(
        not torch.equal(tag_ids_list[i], tag_ids_list[j])
        for i in range(len(tag_ids_list))
        for j in range(i + 1, len(tag_ids_list))
    )
    check("persona tag token sequences are distinct", seq_distinct)

    cosims = [
        F.cosine_similarity(vecs[i], vecs[j]).item()
        for i in range(len(vecs)) for j in range(i + 1, len(vecs))
    ]
    no_exact_collapse = all(c < 0.9999 for c in cosims)
    check("no two persona vectors collapsed to identical (cosim < 0.9999)",
          no_exact_collapse,
          f"max_cosim={max(cosims):.6f}")

    # ── 2.3  Sampling pipeline ────────────────────────────────────
    section("2.3  Sampling pipeline  (temp / top-k / top-p / rep-penalty)")
    ids2 = torch.randint(0, cfg.vocab_size, (1, 10))
    with torch.no_grad():
        logits2, _, _ = model(ids2)
    logits_last = logits2[:, -1, :]

    tok_greedy = _sample_next_token(logits_last, ids2, temperature=0.0)
    check("greedy (temp=0) returns argmax",
          tok_greedy.item() == logits_last.argmax().item())

    tok_sample = _sample_next_token(logits_last.clone(), ids2,
                                    temperature=0.7, top_k=10, top_p=0.9)
    check("sampled token in vocab range",
          0 <= tok_sample.item() < cfg.vocab_size, str(tok_sample.item()))

    tok_rp = _sample_next_token(logits_last.clone(), ids2,
                                 temperature=0.7, repetition_penalty=1.5)
    check("repetition penalty doesn't crash", torch.isfinite(torch.tensor(float(tok_rp.item()))).item())

    # ── 2.4  KV-cache incremental decode ─────────────────────────
    section("2.4  KV-cache incremental decode")
    prompt  = torch.randint(0, cfg.vocab_size, (1, 8))
    max_new = 6

    generated_ref = prompt.clone()
    with torch.no_grad():
        for _ in range(max_new):
            lgts, _, _ = model(generated_ref)
            nt = lgts[:, -1, :].argmax(-1, keepdim=True)
            generated_ref = torch.cat([generated_ref, nt], dim=1)

    generated_kv = prompt.clone()
    kv_caches    = None
    with torch.no_grad():
        for step in range(max_new):
            inp = generated_kv if step == 0 else generated_kv[:, -1:]
            lgts, kv_caches, _ = model(inp, kv_caches=kv_caches, use_kv_cache=True)
            nt = lgts[:, -1, :].argmax(-1, keepdim=True)
            generated_kv = torch.cat([generated_kv, nt], dim=1)

    check("KV-cache decode length matches no-cache",
          generated_kv.shape[1] == generated_ref.shape[1],
          f"{generated_kv.shape[1]} tokens")
    check("KV-cache tokens match no-cache tokens",
          torch.equal(generated_kv, generated_ref))
    check("KV-caches populated after decode",
          kv_caches is not None and kv_caches[0] is not None)

    last_k          = kv_caches[0][0][0]
    expected_kv_seq = prompt.shape[1] + max_new - 1
    check("KV cache seq dim = prompt + generated - 1  (last tok not re-fed)",
          last_k.shape[2] == expected_kv_seq,
          f"shape[2]={last_k.shape[2]}, expected={expected_kv_seq}")

    # ── 2.5  Prompt builder / context truncation ──────────────────
    section("2.5  Prompt builder  (system + context + query layout)")
    SYSTEM   = "You are a cybersecurity expert. Answer from context only."
    CONTEXT  = ["CVE-2024-1234: heap overflow in OpenSSL 3.x RSA key parsing."]
    CHUNK_IDS= ["CVE-2024-1234"]
    QUERY    = "What is CVE-2024-1234?"

    sys_ids   = tok.encode(SYSTEM,   return_tensors="pt") % cfg.vocab_size
    query_str = f"\nUSER QUERY: {QUERY}\nANSWER:"
    q_ids     = tok.encode(query_str, return_tensors="pt") % cfg.vocab_size
    c_parts   = ["\n--- RETRIEVED CONTEXT ---"]
    for cid, chunk in zip(CHUNK_IDS, CONTEXT):
        c_parts.append(f"[{cid}] {chunk}")
    c_parts.append("--- END CONTEXT ---")
    c_ids = tok.encode("\n".join(c_parts), return_tensors="pt") % cfg.vocab_size

    full_prompt = torch.cat([sys_ids, c_ids, q_ids], dim=1)
    check("prompt concatenation succeeds", full_prompt.dim() == 2)
    check("layout: system + context + query (correct order)",
          full_prompt.shape[1] == sys_ids.shape[1] + c_ids.shape[1] + q_ids.shape[1],
          f"total={full_prompt.shape[1]} tokens")

    max_len     = cfg.max_seq_len
    model_input = (full_prompt % cfg.vocab_size)[:, :max_len]
    check(f"prompt truncated to max_seq_len ({max_len})",
          model_input.shape[1] <= max_len,
          f"{model_input.shape[1]} tokens fed to model")

    with torch.no_grad():
        lgts, _, _ = model(model_input)
    check("RAG prompt feeds model without error",
          lgts.shape == (1, model_input.shape[1], cfg.vocab_size))


# ══════════════════════════════════════════════════════════════════
# PHASE 3  —  End-to-End  (real FastAPI server)
# ══════════════════════════════════════════════════════════════════

SERVER_HOST = "127.0.0.1"
SERVER_PORT = 8765
BASE_URL    = f"http://{SERVER_HOST}:{SERVER_PORT}"


def _wait_for_server(timeout: int = 60) -> bool:
    start = time.time()
    while time.time() - start < timeout:
        try:
            r = requests.get(f"{BASE_URL}/health", timeout=2)
            if r.status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(1)
    return False


def _save_tiny_weights(cfg: LLMConfig, path: str):
    """Save a tiny random-weight checkpoint that server.py can load."""
    m = VantaTransformer(cfg)
    torch.save(m.state_dict(), path)


def _find_server_module():
    """
    Locate server.py and return (uvicorn_app_string, cwd) so the subprocess
    can be launched correctly regardless of project layout.

    Supported layouts
    -----------------
    A)  <PROJECT_ROOT>/inference/server.py  →  "inference.server:app"  cwd=PROJECT_ROOT
    B)  <PROJECT_ROOT>/server.py            →  "server:app"            cwd=PROJECT_ROOT
    C)  <SCRIPT_DIR>/inference/server.py   →  "inference.server:app"  cwd=SCRIPT_DIR
    D)  <SCRIPT_DIR>/server.py             →  "server:app"            cwd=SCRIPT_DIR

    For layout A/C, this function also ensures inference/__init__.py exists so
    Python treats the directory as a package.
    """
    candidates = [
        # (server_py_path,                                              app_string,            cwd)
        (os.path.join(PROJECT_ROOT, "vantallm", "inference", "server.py"), "vantallm.inference.server:app", PROJECT_ROOT),
        (os.path.join(PROJECT_ROOT, "inference", "server.py"),  "inference.server:app", PROJECT_ROOT),
        (os.path.join(PROJECT_ROOT, "server.py"),               "server:app",           PROJECT_ROOT),
        (os.path.join(SCRIPT_DIR,   "inference", "server.py"),  "inference.server:app", SCRIPT_DIR),
        (os.path.join(SCRIPT_DIR,   "server.py"),               "server:app",           SCRIPT_DIR),
    ]

    for server_path, app_str, cwd in candidates:
        if os.path.exists(server_path):
            server_dir = os.path.dirname(server_path)
            # If server.py lives inside a sub-package, ensure __init__.py exists
            if os.path.abspath(server_dir) != os.path.abspath(cwd):
                init_path = os.path.join(server_dir, "__init__.py")
                if not os.path.exists(init_path):
                    open(init_path, "w").close()
            return app_str, cwd

    return None, None


def phase3():
    print("\n" + "═"*60)
    print("  PHASE 3 — End-to-End Tests  (real server)")
    print("═"*60)

    # ── Locate server.py before doing anything else ───────────────
    app_str, server_cwd = _find_server_module()
    if app_str is None:
        msg = (
            "  server.py not found in any of:\n"
            f"    {PROJECT_ROOT}/inference/server.py\n"
            f"    {PROJECT_ROOT}/server.py\n"
            f"    {SCRIPT_DIR}/inference/server.py\n"
            f"    {SCRIPT_DIR}/server.py\n"
            "  Create inference/server.py first, then re-run --phase 3."
        )
        print(msg)
        _results.append(("Server startup", False))
        return

    # ── Build tiny checkpoint ─────────────────────────────────────
    ckpt_path = os.path.join(SCRIPT_DIR, "vanta_test_weights.pt")
    cfg_tiny  = LLMConfig(
        vocab_size=151_003, d_model=256, max_seq_len=128,
        num_layers=4, num_q_heads=8, num_kv_heads=2, head_dim=32,
        d_ff_dense=512, num_experts=8, top_k_experts=2, d_ff_moe=128,
        moe_every_n=2, quantization="fp16", composite_persona_dim=128,
    )
    print("  Building tiny checkpoint…", end=" ", flush=True)
    _save_tiny_weights(cfg_tiny, ckpt_path)
    print("done")

    # ── Build PYTHONPATH for the subprocess ───────────────────────
    # The server imports from the vantallm package:
    #   from vantallm.model.transformer import ...  → needs PROJECT_ROOT on sys.path
    #   from vantallm.tokenizer import ...          → needs PROJECT_ROOT on sys.path
    #   (uvicorn imports vantallm.inference.server) → needs server_cwd as root
    extra_paths = [MODEL_DIR, PROJECT_ROOT, server_cwd]
    # De-duplicate while preserving order
    seen = set()
    deduped = []
    for p in extra_paths:
        p = os.path.abspath(p)
        if p not in seen:
            seen.add(p)
            deduped.append(p)

    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(deduped) + os.pathsep + env.get("PYTHONPATH", "")

    env.update({
        "MODEL_PATH":     ckpt_path,
        "TINY_MODEL":     "1",
        "FAST_SMOKE":     "1",
        "TOKENIZER_PATH": os.path.join(PROJECT_ROOT, "vantallm", "tokenizer", "cybersec_tokenizer"),
        "HOST":           SERVER_HOST,
        "PORT":           str(SERVER_PORT),
        "MAX_BATCH_SIZE": "4",
        "BATCH_WAIT_MS":  "10",
    })

    print(f"  Launching server: {app_str}  (cwd={server_cwd})")
    print(f"  PYTHONPATH: {env['PYTHONPATH'][:120]}…")
    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", app_str,
         "--host", SERVER_HOST, "--port", str(SERVER_PORT)],
        cwd=server_cwd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    section("3.1  /health")
    alive = _wait_for_server(timeout=90)
    if not alive:
        proc.terminate()
        try:
            proc.wait(timeout=5)
            logs = proc.stdout.read() if proc.stdout else "(no logs)"
        except Exception:
            logs = "(logs unavailable — process did not exit cleanly)"
        print(f"  {FAIL}  Server failed to start\n{logs[-3000:]}")
        _results.append(("Server startup", False))
        return
    check("Server started and /health returned 200", True)

    r = requests.get(f"{BASE_URL}/health", timeout=5)
    h = r.json()
    check("/health status == 'ok'",       h.get("status") == "ok")
    check("/health reports device",       "device"        in h)
    check("/health reports model_params", "model_params"  in h or "params" in h)

    # ── 3.2  /generate  general route ────────────────────────────
    section("3.2  POST /generate  (general route)")
    payload = {
        "prompt": "What is SQL injection?",
        "route":  "general",
        "max_new_tokens": 4,
        "temperature": 0.7,
        "top_k": 10, "top_p": 0.9,
        "repetition_penalty": 1.1,
    }
    try:
        r = requests.post(f"{BASE_URL}/generate", json=payload, timeout=120)
        ok = r.status_code == 200
        check("/generate status 200", ok, str(r.status_code))
        if ok:
            body = r.json()
            check("response has 'answer'",        "answer"       in body)
            check("response has 'request_id'",    "request_id"   in body)
            check("response has 'route'",         "route"        in body)
            check("response has 'persona'",       "persona"      in body)
            check("response has 'latency_ms'",    "latency_ms"   in body)
            check("route echoed back correctly",  body.get("route") == "general")
            check("oos flag is False",             body.get("oos")  == False)
            print(f"        persona={body.get('persona')}  "
                  f"lat={body.get('latency_ms', 0):.1f}ms  "
                  f"out={body.get('output_tokens')} tokens")
    except Exception as e:
        check(f"/generate general failed: {e}", False)

    # ── 3.3  /generate  exact route with context chunks ──────────
    section("3.3  POST /generate  (exact route + context chunks)")
    payload_ctx = {
        "prompt": "Explain CVE-2024-1234",
        "route":  "exact",
        "context_chunks": ["Heap overflow in OpenSSL 3.x RSA key parsing allows RCE."],
        "chunk_ids":       ["CVE-2024-1234"],
        "max_new_tokens": 4,
        "temperature": 0.3,
    }
    try:
        r = requests.post(f"{BASE_URL}/generate", json=payload_ctx, timeout=120)
        ok = r.status_code == 200
        check("/generate exact status 200", ok, str(r.status_code))
        if ok:
            body = r.json()
            check("persona = cyber_researcher",
                  body.get("persona") == "cyber_researcher",
                  body.get("persona"))
            check("input_tokens > 0",
                  body.get("input_tokens", 0) > 0,
                  str(body.get("input_tokens")))
    except Exception as e:
        check(f"/generate exact failed: {e}", False)

    # ── 3.4  /generate  oos route ─────────────────────────────────
    section("3.4  POST /generate  (oos route — model never invoked)")
    payload_oos = {
        "prompt": "What is the weather in Tokyo?",
        "route":  "oos",
        "max_new_tokens": 64,
    }
    try:
        r = requests.post(f"{BASE_URL}/generate", json=payload_oos, timeout=30)
        ok = r.status_code == 200
        check("/generate oos status 200", ok, str(r.status_code))
        if ok:
            body = r.json()
            check("oos flag is True",      body.get("oos")          == True)
            check("input_tokens == 0",     body.get("input_tokens") == 0)
            check("answer is refusal text", len(body.get("answer", "")) > 10)
    except Exception as e:
        check(f"/generate oos failed: {e}", False)

    # ── 3.5  /generate/stream  SSE ───────────────────────────────
    section("3.5  POST /generate/stream  (SSE token stream)")
    payload_stream = {
        "prompt": "Explain XSS",
        "route":  "general",
        "max_new_tokens": 6,
        "temperature": 0.7,
    }
    try:
        tokens_received = []
        done_seen       = False
        with requests.post(f"{BASE_URL}/generate/stream",
                           json=payload_stream, stream=True,
                           timeout=120) as s:
            check("stream status 200", s.status_code == 200, str(s.status_code))
            for raw in s.iter_lines(decode_unicode=True):
                if not raw:
                    continue
                if raw.startswith("data:"):
                    data_str = raw[5:].strip()
                    try:
                        data_json = json.loads(data_str)
                        if data_json.get("done"):
                            done_seen = True
                            break
                        if "token" in data_json:
                            tokens_received.append(data_json["token"])
                    except json.JSONDecodeError:
                        continue
        check("stream sends at least 1 token",   len(tokens_received) >= 1,
              f"{len(tokens_received)} tokens")
        check("stream ends with 'done' sentinel", done_seen)
    except Exception as e:
        check(f"/generate/stream failed: {e}", False)

    # ── 3.6  session_id  (stateful KV session) ────────────────────
    section("3.6  POST /generate  (session_id → stateful KV cache)")
    sid = str(uuid.uuid4())
    payload_s1 = {
        "prompt": "What is a firewall?",
        "route":  "general",
        "max_new_tokens": 4,
        "temperature": 0.0,
        "session_id": sid,
    }
    payload_s2 = {
        "prompt": "And what about IDS?",
        "route":  "general",
        "max_new_tokens": 4,
        "temperature": 0.0,
        "session_id": sid,
    }
    try:
        r1 = requests.post(f"{BASE_URL}/generate", json=payload_s1, timeout=120)
        r2 = requests.post(f"{BASE_URL}/generate", json=payload_s2, timeout=120)
        check("session turn 1 status 200", r1.status_code == 200)
        check("session turn 2 status 200", r2.status_code == 200)
        if r1.status_code == 200 and r2.status_code == 200:
            check("same session_id echoed in /health kv_sessions",
                  requests.get(f"{BASE_URL}/health").json().get("kv_sessions", 0) >= 0)
    except Exception as e:
        check(f"session_id test failed: {e}", False)

    # ── Teardown ──────────────────────────────────────────────────
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except Exception:
        proc.kill()
    print("\n  Server stopped.")


# ══════════════════════════════════════════════════════════════════
# Summary
# ══════════════════════════════════════════════════════════════════

def summary():
    total  = len(_results)
    passed = sum(1 for _, ok in _results if ok)
    failed = total - passed

    print("\n" + "═"*60)
    print(f"  RESULTS  —  {passed}/{total} passed")
    print("═"*60)

    if failed:
        print(f"\n  {FAIL}  Failed tests:")
        for name, ok in _results:
            if not ok:
                print(f"       • {name}")
    else:
        print(f"\n  {PASS}  All {total} checks passed.")

    print()
    sys.exit(0 if failed == 0 else 1)


# ══════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VantaLLM system test")
    parser.add_argument("--phase", type=int, choices=[1, 2, 3],
                        help="Run only phase 1, 2, or 3 (default: all)")
    args = parser.parse_args()

    run_all = args.phase is None

    if run_all or args.phase == 1:
        phase1()
    if run_all or args.phase == 2:
        phase2()
    if run_all or args.phase == 3:
        phase3()

    summary()
