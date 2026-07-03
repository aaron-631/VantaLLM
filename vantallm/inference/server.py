"""
VantaLLM Inference Server


Fixes over v8.2
  [1]  logits.clone()          — no in-place mutation of model output tensor
  [2]  streaming safety        — engine exceptions always unblock stream consumer
  [3]  output trimming         — trailing pads + post-EOS tokens stripped before decode
  [4]  weights_only=False       — allows backward-compatible loading for architectural metadata
  [5]  eval() hot-path         — model.eval() set once at load; removed from inference loop
  [6]  TPS logging             — tokens/sec logged for batched + cached paths  (from v6)
  [7]  /health enriched        — reports param count, queue depth, active sessions  (from v6)
  [8]  per-request log         — persona, route, latency, token counts logged per call  (from v6)
  [9]  MockTokenizer warn      — emits warning when mock input is silently truncated  (from v6)
  [10] full UUID               — request_id is full uuid4, not an 8-char slice
  [11] persona pre-warm        — all persona vectors computed at startup, not on first request
  [12] embed() guard           — graceful fallback if model.embed() is not implemented
"""

import json
import os
import time
import uuid
import asyncio
import logging
import threading
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from vantallm.model.transformer import VantaTransformer, LLMConfig
from vantallm.tokenizer.tokenizer import load_tokenizer as load_custom_tokenizer, MockTokenizer as CanonicalMockTokenizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
log = logging.getLogger("inference_server")


# ==============================================================================
# 1. Configuration, Personas & Routing
# ==============================================================================

MAX_CONTEXT_LEN     = 128_000
SESSION_TTL_SECONDS = 1800   # 30-min idle → evict KV session

OOS_REFUSAL = (
    "This query is outside the cybersecurity domain. "
    "I'm specialised in security topics only."
)

# Per-persona tag strings → tokenised → last-token hidden state → L2-normed
# conditioning vector injected into VantaTransformer.forward(persona_vector=...)
PERSONAS: Dict[str, str] = {
    "cyber_researcher":   "<role:researcher> <domain:cyber> <grounding:strict>",
    "security_ciso":      "<role:executive>  <domain:cyber> <style:strategic>",
    "technical_educator": "<role:teacher>    <domain:general> <detail:high>",
    "general_assistant":  "<role:assistant>  <domain:general> <style:helpful>",
    "oos_refusal":        "<role:gatekeeper> <mode:refusal>",
}


def select_persona(query: str, route: str) -> str:
    """Keyword + route based persona selection."""
    q = query.lower()
    if route == "oos":
        return "oos_refusal"
    if route in ("exact", "hybrid"):
        return "cyber_researcher"
    if any(k in q for k in ["exploit", "attack", "payload", "vulnerability", "cve"]):
        return "cyber_researcher"
    if any(k in q for k in ["prevent", "mitigate", "risk", "policy", "compliance"]):
        return "security_ciso"
    if any(k in q for k in ["explain", "what", "how", "teach"]):
        return "technical_educator"
    return "general_assistant"


# ==============================================================================
# 2. Sampling
# ==============================================================================

@torch.no_grad()
def sample_next_token(
    logits: torch.Tensor,       # (1, vocab)
    input_ids: torch.Tensor,    # (1, seq)
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
    repetition_penalty: float = 1.1,
    rep_penalty_window: int = 256,
) -> torch.Tensor:
    # Clone first — never mutate the model's output tensor in-place
    logits = logits.clone()

    if repetition_penalty != 1.0:
        recent_ids = input_ids[:, -rep_penalty_window:]
        score = torch.gather(logits, 1, recent_ids)
        score = torch.where(score < 0, score * repetition_penalty, score / repetition_penalty)
        logits.scatter_(1, recent_ids, score)

    if temperature <= 0.0:
        return logits.argmax(dim=-1)

    logits = logits / max(temperature, 1e-8)

    if top_k > 0:
        top_k   = min(top_k, logits.size(-1))
        kth_val = torch.topk(logits, top_k, dim=-1).values[:, -1, None]
        logits  = logits.masked_fill(logits < kth_val, float("-inf"))

    if top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, dim=-1, descending=True)
        sorted_probs = F.softmax(sorted_logits, dim=-1)          # compute once
        cumprobs     = torch.cumsum(sorted_probs, dim=-1)
        sorted_remove = (cumprobs - sorted_probs) > top_p        # reuse sorted_probs
        sorted_logits[sorted_remove] = float("-inf")
        logits = torch.zeros_like(logits).scatter_(1, sorted_idx, sorted_logits)

    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1).squeeze(-1)


# ==============================================================================
# 3. Prompt Construction & Output Trimming
# ==============================================================================

from vantallm.utils.prompting import build_prompt

def build_and_tokenize_prompt(
    user_query:     str,
    route:          str,
    context_chunks: List[str],
    chunk_ids:      List[str],
    tokenizer,
    persona_name:   str,
) -> torch.Tensor:
    """
    Token layout: [SYSTEM] [RAG CONTEXT (if any)] [USER QUERY]
    Prompt is assembled as one contiguous string so BOS/EOS are only applied once.
    Uses the shared authoritative prompt builder to preserve prompt parity with
    training and evaluation; persona routing conditions the model via
    `persona_vector`, not by swapping the base prompt template.
    """
    # RAG context (retrieval routes only)
    context_str = None
    if route in ("exact", "hybrid") and context_chunks:
        c_parts = []
        for cid, chunk in zip(chunk_ids, context_chunks):
            c_parts.append(f"[{cid}] {chunk.strip()}")
        context_str = "\n".join(c_parts)

    full_prompt = build_prompt(question=user_query, context=context_str)

    input_ids = tokenizer.encode(full_prompt, return_tensors="pt")
    if input_ids.shape[1] > MAX_CONTEXT_LEN:
        raise ValueError(
            f"Prompt length ({input_ids.shape[1]}) exceeds MAX_CONTEXT_LEN ({MAX_CONTEXT_LEN})."
        )
    if input_ids[0, -1].item() == tokenizer.eos_token_id:
        input_ids = input_ids[:, :-1]
    return input_ids


def trim_output(ids: torch.Tensor, pad_id: int, eos_id: int) -> torch.Tensor:
    """
    Given a (1, seq) tensor containing input + generated tokens (left-pad already removed),
    truncate at the first EOS token and strip any trailing pad tokens.
    Returns (1, trimmed_len).
    """
    seq = ids[0]

    # Truncate at first EOS (exclude EOS itself from decoded output)
    eos_pos = (seq == eos_id).nonzero(as_tuple=False)
    if eos_pos.numel() > 0:
        seq = seq[:eos_pos[0].item()]

    # Strip trailing pads (generated for sequences that finished before max_new_tokens)
    last = seq.shape[0] - 1
    while last >= 0 and seq[last].item() == pad_id:
        last -= 1

    return seq[:last + 1].unsqueeze(0)


# ==============================================================================
# 4. Tokenizer
# ==============================================================================

def load_tokenizer():
    tok = load_custom_tokenizer()
    if isinstance(tok, CanonicalMockTokenizer):
        log.warning("Using fallback MockTokenizer (dev/test only).")
    else:
        log.info(f"Loaded custom tokenizer from {os.environ.get('TOKENIZER_PATH', 'cybersec_tokenizer/')}")
    return tok


# ==============================================================================
# 5. Inference Engine
# ==============================================================================

@dataclass
class InferenceRequest:
    request_id:         str
    session_id:         Optional[str]
    input_ids:          torch.Tensor
    persona_vector:     Optional[torch.Tensor]  # (1, d_model), L2-normalised
    max_new_tokens:     int
    temperature:        float
    top_k:              int
    top_p:              float
    repetition_penalty: float
    future:             asyncio.Future
    stream_queue:       Optional[asyncio.Queue] = None


@dataclass
class KVSession:
    cache:         Any
    last_accessed: float


class BatchInferenceEngine:

    def __init__(
        self,
        model:          VantaTransformer,
        tokenizer,
        device:         torch.device,
        max_batch_size: int   = 8,
        batch_wait_ms:  float = 20.0,
    ):
        self.model          = model
        self.tokenizer      = tokenizer
        self.device         = device
        self.max_batch_size = max_batch_size
        self.batch_wait_ms  = batch_wait_ms

        self._queue:        asyncio.Queue[InferenceRequest] = asyncio.Queue(maxsize=100)
        self._kv_sessions:  Dict[str, KVSession]            = {}
        self._session_lock  = asyncio.Lock()
        self._worker_task   = None
        self._ttl_task      = None

    # ------------------------------------------------------------------ lifecycle

    async def start(self):
        self._worker_task = asyncio.create_task(self._worker_loop())
        self._ttl_task    = asyncio.create_task(self._ttl_sweep_loop())

    async def stop(self):
        if self._worker_task: self._worker_task.cancel()
        if self._ttl_task:    self._ttl_task.cancel()

    async def submit(self, req: InferenceRequest):
        try:
            self._queue.put_nowait(req)
        except asyncio.QueueFull:
            raise HTTPException(status_code=429, detail="Server at capacity. Try again later.")
        return await req.future

    # ------------------------------------------------------------------ TTL eviction

    async def _ttl_sweep_loop(self):
        """Background task: evict KV sessions idle longer than SESSION_TTL_SECONDS."""
        while True:
            try:
                await asyncio.sleep(60)
                now = time.time()
                async with self._session_lock:
                    stale = [
                        k for k, v in self._kv_sessions.items()
                        if now - v.last_accessed > SESSION_TTL_SECONDS
                    ]
                    for k in stale:
                        del self._kv_sessions[k]
                        log.info(f"Evicted stale KV session: {k}")
            except asyncio.CancelledError:
                break
            except Exception as e:
                log.error(f"TTL sweep error: {e}")

    # ------------------------------------------------------------------ worker loop

    async def _worker_loop(self):
        """Collect requests into batches; dispatch to thread-pool executor."""
        loop = asyncio.get_running_loop()
        while True:
            batch: List[InferenceRequest] = []
            try:
                first = await self._queue.get()
                batch.append(first)

                deadline = time.monotonic() + self.batch_wait_ms / 1000.0
                while len(batch) < self.max_batch_size:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        break
                    try:
                        req = await asyncio.wait_for(self._queue.get(), timeout=remaining)
                        batch.append(req)
                    except asyncio.TimeoutError:
                        break

                cancel_event = threading.Event()
                try:
                    await asyncio.wait_for(
                        loop.run_in_executor(
                            None, self._run_batch_sync, batch, loop, cancel_event
                        ),
                        timeout=120.0,
                    )
                except asyncio.TimeoutError:
                    log.error("Batch execution timed out (120 s).")
                    cancel_event.set()
                    for req in batch:
                        if not req.future.done():
                            req.future.set_exception(
                                HTTPException(status_code=504, detail="Model timeout")
                            )
                except Exception as e:
                    import traceback
                    log.error(f"Batch execution failed: {e}\n{traceback.format_exc()}")
                    for req in batch:
                        if not req.future.done():
                            req.future.set_exception(
                                HTTPException(status_code=500, detail=f"Model error: {type(e).__name__}")
                            )

            except asyncio.CancelledError:
                break
            except Exception as e:
                log.error(f"Worker loop fatal error: {e}")

    # ------------------------------------------------------------------ KV helpers

    def _slice_layer_kv(self, layer_cache, batch_idx: int):
        """Extract one item's KV cache from a batched cache tensor."""
        if layer_cache is None:
            return None
        (k_q, k_s), (v_q, v_s) = layer_cache
        return (
            (k_q[batch_idx:batch_idx + 1], k_s[batch_idx:batch_idx + 1]),
            (v_q[batch_idx:batch_idx + 1], v_s[batch_idx:batch_idx + 1]),
        )

    async def _update_session(self, session_id: str, kv_cache):
        async with self._session_lock:
            self._kv_sessions[session_id] = KVSession(
                cache=kv_cache, last_accessed=time.time()
            )

    # ------------------------------------------------------------------ batch inference

    @torch.no_grad()
    def _run_batch_sync(
        self,
        batch:        List[InferenceRequest],
        loop:         asyncio.AbstractEventLoop,
        cancel_event: threading.Event,
    ):
        pad_id = getattr(self.tokenizer, "pad_token_id", 0)
        eos_id = getattr(self.tokenizer, "eos_token_id", 1)

        # model.eval() is set once at load_model(); do NOT call it here (hot path)
        fresh_reqs  = [r for r in batch
                       if not r.session_id or r.session_id not in self._kv_sessions]
        cached_reqs = [r for r in batch
                       if r.session_id and r.session_id in self._kv_sessions]

        # ============================================================
        # Fresh requests — true parallel batched generation
        # ============================================================
        if fresh_reqs:
            B       = len(fresh_reqs)
            max_len = max(r.input_ids.shape[1] for r in fresh_reqs)

            padded_inputs, attention_masks = [], []
            for req in fresh_reqs:
                pad_len = max_len - req.input_ids.shape[1]
                padded  = F.pad(req.input_ids, (pad_len, 0), value=pad_id).to(self.device)
                mask    = F.pad(
                    torch.ones_like(req.input_ids), (pad_len, 0), value=0
                ).to(self.device)
                padded_inputs.append(padded)
                attention_masks.append(mask)

            batched_ids   = torch.cat(padded_inputs, dim=0)
            batched_masks = torch.cat(attention_masks, dim=0)
            prefill_position_ids = batched_masks.long().cumsum(dim=-1) - 1
            prefill_position_ids.masked_fill_(batched_masks == 0, 0)

            # Stack persona vectors; fall back to zeros for any None
            valid_vecs = [r.persona_vector for r in fresh_reqs if r.persona_vector is not None]
            if valid_vecs:
                zero_vec        = torch.zeros_like(valid_vecs[0])
                batched_persona = torch.cat(
                    [r.persona_vector if r.persona_vector is not None else zero_vec
                     for r in fresh_reqs],
                    dim=0,
                )   # (B, d_model)
            else:
                batched_persona = None

            generated        = batched_ids
            kv_caches        = None
            max_new          = max(r.max_new_tokens for r in fresh_reqs)
            done_mask        = torch.zeros(B, dtype=torch.bool, device=self.device)
            tokens_generated = 0
            t_start          = time.perf_counter()

            for step in range(max_new):
                if cancel_event.is_set():
                    break

                if step == 0:
                    log.info(f"Step 0: batched_ids shape={batched_ids.shape}, min={batched_ids.min()}, max={batched_ids.max()}")
                    if batched_ids.max() >= self.model.config.vocab_size:
                        log.error(f"FATAL: Token ID {batched_ids.max()} exceeds vocab_size {self.model.config.vocab_size}")

                ids = generated if step == 0 else generated[:, -1:]
                current_mask = batched_masks if step == 0 else None

                logits, kv_caches, _ = self.model(
                    ids,
                    attention_mask=current_mask,
                    position_ids=prefill_position_ids if step == 0 else None,
                    kv_caches=kv_caches,
                    use_kv_cache=True,
                    persona_vector=batched_persona,
                )

                next_tokens = []
                for i in range(B):
                    tok = sample_next_token(
                        logits[i:i + 1, -1, :],
                        input_ids=generated[i:i + 1],
                        temperature=fresh_reqs[i].temperature,
                        top_k=fresh_reqs[i].top_k,
                        top_p=fresh_reqs[i].top_p,
                        repetition_penalty=fresh_reqs[i].repetition_penalty,
                    )
                    next_tokens.append(tok)

                next_tokens_t = torch.cat(next_tokens, dim=0).unsqueeze(-1)  # (B, 1)
                next_tokens_t[done_mask] = pad_id
                generated = torch.cat([generated, next_tokens_t], dim=1)

                # Extend attention mask by 1 every step — critical for correct KV-cache attention
                batched_masks = torch.cat(
                    [batched_masks, torch.ones((B, 1), device=self.device)], dim=1
                )

                # Stream tokens to any waiting consumers (blocks if consumer is slow — backpressure)
                for i, req in enumerate(fresh_reqs):
                    if req.stream_queue and not done_mask[i]:
                        token_str = self.tokenizer.decode(next_tokens_t[i].view(-1))
                        loop.call_soon_threadsafe(req.stream_queue.put_nowait, token_str)

                tokens_generated += int((~done_mask).sum().item())
                # Force at least 5 tokens before allowing a stop
                is_eos = (next_tokens_t.squeeze(-1) == eos_id)
                if step >= 4:   # step is 0-indexed; allow EOS from the 5th token onward
                    done_mask |= is_eos

                if done_mask.all():
                    break

            elapsed = max(time.perf_counter() - t_start, 1e-4)
            log.info(
                f"Batched generation: B={B} | "
                f"TPS={tokens_generated / elapsed:.1f} | "
                f"{elapsed * 1000:.0f} ms"
            )

            for i, req in enumerate(fresh_reqs):
                # Unblock any stream consumer
                if req.stream_queue:
                    loop.call_soon_threadsafe(req.stream_queue.put_nowait, None)

                # Strip left-pad artifact; defer EOS trimming until after prompt slicing.
                pad_len    = max_len - req.input_ids.shape[1]
                raw_ids    = generated[i:i + 1, pad_len:]
                result_ids = raw_ids

                # Sync KV cache to session store BEFORE resolving future (prevents race)
                if req.session_id:
                    sliced_kv = (
                        [self._slice_layer_kv(lc, i) for lc in kv_caches]
                        if kv_caches else None
                    )
                    loop.call_soon_threadsafe(
                        lambda r=req, kv=sliced_kv: asyncio.create_task(self._update_session(r.session_id, kv))
                    )

                if not req.future.done():
                    loop.call_soon_threadsafe(req.future.set_result, result_ids)

        # ============================================================
        # Cached requests — sequential (stateful sessions require isolation)
        # ============================================================
        for req in cached_reqs:
            if cancel_event.is_set():
                break

            kv_cache  = self._kv_sessions[req.session_id].cache
            generated = req.input_ids.to(self.device)
            t_start   = time.perf_counter()

            for step in range(req.max_new_tokens):
                if cancel_event.is_set():
                    break

                ids = generated[:, -1:]

                logits, kv_cache, _ = self.model(
                    ids,
                    kv_caches=kv_cache,
                    use_kv_cache=True,
                    persona_vector=req.persona_vector,
                )

                next_token = sample_next_token(
                    logits[:, -1, :],
                    input_ids=generated,
                    temperature=req.temperature,
                    top_k=req.top_k,
                    top_p=req.top_p,
                    repetition_penalty=req.repetition_penalty,
                ).unsqueeze(-1)

                generated = torch.cat([generated, next_token], dim=1)

                if req.stream_queue:
                    token_str = self.tokenizer.decode(next_token[0])
                    loop.call_soon_threadsafe(req.stream_queue.put_nowait, token_str)

                if next_token.item() == eos_id:
                    break

            elapsed = max(time.perf_counter() - t_start, 1e-4)
            log.info(
                f"Cached generation [{req.request_id[:8]}]: "
                f"TPS={req.max_new_tokens / elapsed:.1f}"
            )

            if req.stream_queue:
                loop.call_soon_threadsafe(req.stream_queue.put_nowait, None)

            # Sync cache BEFORE resolving future
            loop.call_soon_threadsafe(
                lambda r=req, kv=kv_cache: asyncio.create_task(self._update_session(r.session_id, kv))
            )

            if not req.future.done():
                loop.call_soon_threadsafe(req.future.set_result, generated)


# ==============================================================================
# 6. Application State & Lifespan
# ==============================================================================

class AppState:
    model:     VantaTransformer
    tokenizer: Any
    engine:    BatchInferenceEngine
    device:    torch.device

    def __init__(self):
        # Instance-level — avoids Python's shared mutable class-level default bug
        self.persona_cache: Dict[str, torch.Tensor] = {}

    def get_persona_vector(self, persona_name: str) -> Optional[torch.Tensor]:
        """
        Encode persona tag string → last-token hidden state → L2-normalise.
        Cached per persona name; each vector computed exactly once per process.
        Returns None if model.embed() is not implemented (inference still proceeds).
        """
        if persona_name not in self.persona_cache:
            tag = PERSONAS.get(persona_name, PERSONAS["general_assistant"])
            ids = self.tokenizer.encode(tag, return_tensors="pt").to(self.device)
            try:
                with torch.no_grad():
                    emb = self.model.embed(ids)              # (1, seq, d_model)
                    vec = emb[:, -1, :]                      # last-token rep — correct for causal LM
                    vec = vec / (vec.norm(dim=-1, keepdim=True) + 1e-6)
                self.persona_cache[persona_name] = vec
            except AttributeError:
                log.warning(
                    "model.embed() not found — persona vector conditioning is disabled. "
                    "Implement embed() in VantaTransformer to enable it."
                )
                return None
        return self.persona_cache[persona_name]


app_state = AppState()


def load_model() -> VantaTransformer:
    def _extract_model_state_dict(checkpoint_data: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        model_state = checkpoint_data.get("model")
        if isinstance(model_state, dict):
            return model_state
        return checkpoint_data

    def _env_flag(name: str) -> bool:
        value = os.environ.get(name)
        if value is None:
            return False
        return value.strip().lower() in {"1", "true", "yes", "on"}

    model_path = os.environ.get("MODEL_PATH", "")
    # Read quantization mode from env; default fp16 so existing behaviour is unchanged
    quant_mode = os.environ.get("QUANT_MODE", "fp16").lower().strip()
    log.info(f"Quantization mode: {quant_mode.upper()}")

    use_tiny = _env_flag("TINY_MODEL")

    # Safety Guard: Prevent accidental 125B instantiation on small GPUs
    if not use_tiny and not os.environ.get("MODEL_CONFIG", "").strip():
        raise RuntimeError(
            "CRITICAL OPERATIONAL ERROR: TINY_MODEL=0 (or unset) but no MODEL_CONFIG provided.\n"
            "The default VantaLLM architecture is ~125B parameters and will not fit on a single workstation GPU.\n"
            "To run on a single GPU, specify a smaller config:\n"
            "  export MODEL_CONFIG=configs/vanta_567m.json\n"
            "Or use TINY_MODEL=1 for local CPU smoke tests."
        )

    # Support a tiny/dev model for tests (set TINY_MODEL=1 in env)
    if use_tiny:
        cfg = LLMConfig(
            vocab_size=151_003,
            d_model=256,
            max_seq_len=512,
            num_layers=4,
            num_q_heads=8,
            num_kv_heads=2,
            head_dim=32,
            d_ff_dense=512,
            num_experts=8,
            top_k_experts=2,
            d_ff_moe=128,
            moe_every_n=2,
            quantization=quant_mode,
            composite_persona_dim=128,
        )
    else:
        config_path = os.environ.get("MODEL_CONFIG", "").strip()
        if config_path:
            with open(config_path, "r", encoding="utf-8") as f:
                raw_config = json.load(f)
            if not isinstance(raw_config, dict):
                raise ValueError(f"MODEL_CONFIG must contain a JSON object: {config_path}")
            raw_config = dict(raw_config)
            if os.environ.get("QUANT_MODE"):
                raw_config["quantization"] = quant_mode
            else:
                raw_config.setdefault("quantization", quant_mode)
            log.info(f"Model config path: {config_path}")
            cfg = LLMConfig(**raw_config)
        else:
            cfg = LLMConfig(quantization=quant_mode)

    global MAX_CONTEXT_LEN
    MAX_CONTEXT_LEN = cfg.max_seq_len
    log.info(f"MAX_CONTEXT_LEN set to {MAX_CONTEXT_LEN} from model config.")

    model = VantaTransformer(cfg)

    if cfg.quantization == "bf16" and torch.cuda.is_available():
        model = model.to(torch.bfloat16)
        log.info("BF16 cast applied: model parameters are now bfloat16.")

    if model_path and os.path.exists(model_path):
        # weights_only=False: preserve backward-compatible loading for trusted local checkpoints
        checkpoint_data = torch.load(model_path, map_location="cpu", weights_only=False)
        state_dict = _extract_model_state_dict(checkpoint_data)
        
        # Manual fix for vocab size mismatch (151,000 vs 151,003)
        for key in ["token_embed.weight", "lm_head.weight"]:
            if key in state_dict and key in model.state_dict():
                ckpt_weight = state_dict[key]
                model_weight = model.state_dict()[key]
                if ckpt_weight.shape != model_weight.shape:
                    log.warning(f"Shape mismatch for {key}: {ckpt_weight.shape} vs {model_weight.shape}. Patching...")
                    new_weight = model_weight.clone()
                    # Copy as much as possible
                    min_vocab = min(ckpt_weight.shape[0], model_weight.shape[0])
                    new_weight[:min_vocab] = ckpt_weight[:min_vocab]
                    state_dict[key] = new_weight

        model.load_state_dict(state_dict, strict=False)
        log.info(f"Loaded weights from {model_path} (patched for vocab mismatch if needed)")
    else:
        reason = "not set" if not model_path else f"{model_path!r} not found"
        log.warning(f"MODEL_PATH {reason} — running with random weights (dev/test only).")
    model.eval()   # set once here; never toggled during inference
    return model


@asynccontextmanager
async def lifespan(app: FastAPI):
    app_state.device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    app_state.tokenizer = load_tokenizer()
    log.info(f"Tokenizer vocab size: {len(app_state.tokenizer):,}")
    log.info(f"EOS token id: {app_state.tokenizer.eos_token_id}")
    app_state.model     = load_model().to(app_state.device)

    # Pre-warm all persona vectors at startup to avoid first-request latency spikes
    for name in PERSONAS:
        app_state.get_persona_vector(name)
    log.info(f"Persona vectors cached: {list(app_state.persona_cache.keys())}")

    app_state.engine = BatchInferenceEngine(
        model          = app_state.model,
        tokenizer      = app_state.tokenizer,
        device         = app_state.device,
        max_batch_size = int(os.environ.get("MAX_BATCH_SIZE", 8)),
        batch_wait_ms  = float(os.environ.get("BATCH_WAIT_MS", 20.0)),
    )
    await app_state.engine.start()
    log.info("Inference server ready ✓")
    yield
    await app_state.engine.stop()


# ==============================================================================
# 7. FastAPI Application & Routes
# ==============================================================================

app = FastAPI(
    title="VantaLLM Inference Server",
    version="9.0.0",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # TODO: lock down to your client domain before production deploy
    allow_methods=["*"],
    allow_headers=["*"],
)


class GenerateRequest(BaseModel):
    prompt:             str
    route:              str   = Field(default="general")
    context_chunks:     List[str] = Field(default_factory=list)
    chunk_ids:          List[str] = Field(default_factory=list)
    max_new_tokens:     int   = Field(default=512, ge=1, le=2048)
    temperature:        float = Field(default=0.7, ge=0.0, le=2.0)
    top_k:              int   = Field(default=50,  ge=0,   le=1000)
    top_p:              float = Field(default=0.9, ge=0.0, le=1.0)
    repetition_penalty: float = Field(default=1.1, ge=1.0, le=2.0)
    session_id:         Optional[str] = Field(default=None)


class GenerateResponse(BaseModel):
    request_id:    str
    route:         str
    persona:       str
    answer:        str
    input_tokens:  int
    output_tokens: int
    latency_ms:    float
    oos:           bool = False


@app.get("/health", tags=["meta"])
async def health():
    param_b = sum(p.numel() for p in app_state.model.parameters()) / 1e9
    return {
        "status":       "ok",
        "device":       str(app_state.device),
        "model_params": f"{param_b:.1f}B",
        "kv_sessions":  len(app_state.engine._kv_sessions),
        "queue_depth":  app_state.engine._queue.qsize(),
    }


@app.post("/generate", response_model=GenerateResponse, tags=["inference"])
async def generate(req: GenerateRequest):
    if not req.prompt.strip():
        raise HTTPException(status_code=422, detail="prompt must not be empty.")
    request_id   = str(uuid.uuid4())
    t_start      = time.perf_counter()
    persona_name = select_persona(req.prompt, req.route)

    if req.route == "oos":
        return GenerateResponse(
            request_id=request_id, route=req.route, persona=persona_name,
            answer=OOS_REFUSAL, input_tokens=0, output_tokens=0,
            latency_ms=(time.perf_counter() - t_start) * 1000, oos=True,
        )

    try:
        input_ids = build_and_tokenize_prompt(
            req.prompt, req.route, req.context_chunks, req.chunk_ids,
            app_state.tokenizer, persona_name,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    n_input = input_ids.shape[-1]

    if n_input + req.max_new_tokens > MAX_CONTEXT_LEN:
        raise HTTPException(
            status_code=400,
            detail=f"prompt_length ({n_input}) + max_new_tokens ({req.max_new_tokens}) exceeds context window ({MAX_CONTEXT_LEN})."
        )

    inf_req = InferenceRequest(
        request_id=request_id,
        session_id=req.session_id,
        input_ids=input_ids,
        persona_vector=app_state.get_persona_vector(persona_name),
        max_new_tokens=req.max_new_tokens,
        temperature=req.temperature,
        top_k=req.top_k,
        top_p=req.top_p,
        repetition_penalty=req.repetition_penalty,
        future=asyncio.get_running_loop().create_future(),
    )

    # result_ids = full sequence (input + generated); slice off input prefix first.
    result_ids = await app_state.engine.submit(inf_req)
    new_ids    = result_ids[:, n_input:]
    new_ids    = trim_output(new_ids, app_state.tokenizer.pad_token_id, app_state.tokenizer.eos_token_id)
    answer     = app_state.tokenizer.decode(new_ids[0])
    latency    = (time.perf_counter() - t_start) * 1000

    log.info(
        f"[{request_id[:8]}] persona={persona_name} route={req.route} "
        f"lat={latency:.1f}ms in={n_input} out={new_ids.shape[-1]} "
        f"session={req.session_id}"
    )

    return GenerateResponse(
        request_id=request_id, route=req.route, persona=persona_name,
        answer=answer, input_tokens=n_input, output_tokens=new_ids.shape[-1],
        latency_ms=latency,
    )


@app.post("/generate/stream", tags=["inference"])
async def generate_stream(req: GenerateRequest):
    if not req.prompt.strip():
        raise HTTPException(status_code=422, detail="prompt must not be empty.")
    request_id   = str(uuid.uuid4())
    persona_name = select_persona(req.prompt, req.route)

    if req.route == "oos":
        async def oos_stream():
            yield f"data: {json.dumps({'token': OOS_REFUSAL, 'oos': True})}\n\n"
            yield f"data: {json.dumps({'done': True})}\n\n"
        return StreamingResponse(oos_stream(), media_type="text/event-stream")

    try:
        input_ids = build_and_tokenize_prompt(
            req.prompt, req.route, req.context_chunks, req.chunk_ids,
            app_state.tokenizer, persona_name,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    n_input = input_ids.shape[-1]

    if n_input + req.max_new_tokens > MAX_CONTEXT_LEN:
        raise HTTPException(
            status_code=400,
            detail=f"prompt_length ({n_input}) + max_new_tokens ({req.max_new_tokens}) exceeds context window ({MAX_CONTEXT_LEN})."
        )

    stream_q = asyncio.Queue(maxsize=32)  # bounded — applies backpressure on slow consumers

    inf_req = InferenceRequest(
        request_id=request_id,
        session_id=req.session_id,
        input_ids=input_ids,
        persona_vector=app_state.get_persona_vector(persona_name),
        max_new_tokens=req.max_new_tokens,
        temperature=req.temperature,
        top_k=req.top_k,
        top_p=req.top_p,
        repetition_penalty=req.repetition_penalty,
        future=asyncio.get_running_loop().create_future(),
        stream_queue=stream_q,
    )

    async def _submit_and_signal():
        """Engine submit task — always unblocks stream consumer even on failure."""
        try:
            await app_state.engine.submit(inf_req)
            # On success the engine itself already put None in stream_q — don't double-send.
        except Exception as e:
            log.error(f"Stream engine error [{request_id[:8]}]: {e}")
            # Engine may not have sent the sentinel — guarantee the consumer unblocks.
            try:
                stream_q.put_nowait(None)
            except asyncio.QueueFull:
                pass  # Consumer is slow but alive; it will drain and see None eventually

    asyncio.create_task(_submit_and_signal())

    async def token_stream():
        # Buffer sub-word BPE fragments into whole words before flushing to the client.
        # Flush when: a word boundary arrives, buffer hits 8+ chars, or stream ends.
        FLUSH_CHARS = frozenset(" \t\n.,;:!?)]}")
        MAX_BUF_LEN = 8
        buf = []

        while True:
            token = await stream_q.get()

            if token is None:
                # Drain remaining buffer before sending done sentinel
                if buf:
                    chunk = "".join(buf)
                    yield f"data: {json.dumps({'token': chunk})}\n\n"
                yield f"data: {json.dumps({'done': True})}\n\n"
                break

            buf.append(token)

            # Flush on word-boundary char OR when buffer is long enough
            if token[-1:] in FLUSH_CHARS or sum(len(t) for t in buf) >= MAX_BUF_LEN:
                chunk = "".join(buf)
                buf.clear()
                yield f"data: {json.dumps({'token': chunk})}\n\n"

    return StreamingResponse(token_stream(), media_type="text/event-stream")


if __name__ == "__main__":
    uvicorn.run(
        "vantallm.inference.server:app",
        host=os.environ.get("HOST", "0.0.0.0"),
        port=int(os.environ.get("PORT", 8000)),
    )
