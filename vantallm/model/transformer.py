"""
transformer.py — VantaLLM decoder-only transformer.

A single, self-contained model definition that combines the base transformer,
optional persona-aware conditioning, and a lightweight PPO/RL training scaffold.
The same code paths run from a tiny CPU config up to the largest scaling-reference
config; model size is controlled entirely by ``LLMConfig``.

Architecture
────────────
Token Embedding  (vocab = V, dim = d_model)
  + 0.1 × persona_signal  ← injected here when persona conditioning is enabled
  ↓
N × TransformerBlock  (dense and MoE feed-forwards alternate every ``moe_every_n`` blocks)
  ┌─────────────────────────────────────────────────────────────────┐
  │  RMSNorm → grouped-query attention (RoPE + quantized KV) → residual │
  │  RMSNorm → dense SwiGLU  OR  MoE feed-forward            → residual │
  └─────────────────────────────────────────────────────────────────┘
  ↓
Final RMSNorm → Linear(d_model → V)  [weight-tied to the embedding]

Persona conditioning (optional)
───────────────────────────────
Training  (persona_ids given):
  persona_ids → PersonaEmbeddingStore → mixture-of-experts router
              → composite_ctx → conditions attention K/V and the MoE gate
              → persona_to_hidden → added to the hidden state at the embedding layer

Inference (persona_vector given):
  persona_vector (L2-normed) → added to the hidden state at the embedding layer
                             → projected to composite_ctx → conditions attention / MoE

Both None → no persona conditioning (plain forward pass, fully backwards compatible).

Quantization modes (``QUANT_MODE`` env var)
───────────────────────────────────────────
  fp16  →  nn.Linear  (default)
  bf16  →  nn.Linear  (apply ``model.to(torch.bfloat16)`` at runtime)
  int8  →  Int8Linear (experimental scaffold; no current VRAM savings)
  int4  →  Int4Linear (experimental scaffold; no current VRAM savings)
"""

import os
import json
import math
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

log = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════
# 1.  Configuration
# ═══════════════════════════════════════════════════════════════════

@dataclass
class LLMConfig:
    # ── Embedding ──────────────────────────────────────────────────
    vocab_size: int        = 151_003
    d_model: int           = 4_096
    max_seq_len: int       = 131_072        # 128k context window

    # ── Transformer depth ──────────────────────────────────────────
    num_layers: int        = 94
    moe_every_n: int       = 2              # MoE on odd-indexed layers (1,3,5,…)

    # ── Attention (GQA) ────────────────────────────────────────────
    num_q_heads: int       = 32
    num_kv_heads: int      = 8              # 4× KV compression
    head_dim: int          = 128

    # ── Dense FFN (SwiGLU) ─────────────────────────────────────────
    d_ff_dense: int        = 12_288

    # ── MoE FFN ────────────────────────────────────────────────────
    num_experts: int       = 128
    top_k_experts: int     = 2
    d_ff_moe: int          = 1_536          # per-expert hidden dim
    moe_aux_loss_coeff: float = 1e-2
    router_jitter: float   = 0.0

    # ── Misc ───────────────────────────────────────────────────────
    rms_norm_eps: float    = 1e-6
    rope_theta: float      = 500_000.0      # extended base for 128k context
    dropout: float         = 0.0

    # ── TurboQuant KV cache ────────────────────────────────────────
    kv_quant_bits: int     = 8

    # ── Quantization ───────────────────────────────────────────────
    quantization: str      = "fp16"         # "fp16" | "bf16" | "int8" (mock) | "int4" (mock)

    # ── Activation Checkpointing ───────────────────────────────────
    gradient_checkpointing: bool = False

    # ── MoE Safety ─────────────────────────────────────────────────
    capacity_factor: float = 0.0            # 0.0 means unbounded capacity

    # ── Persona System ─────────────────────────────────────────────
    # Persona conditioning components
    persona_embedding_dim: int  = 256       # per-axis embedding size
    num_persona_types: int      = 3         # Pn (narrative), Po (objective), Ps (style)
    num_persona_experts: int    = 4         # EoE composition strategies
    composite_persona_dim: int  = 512       # composite vector dim → GQA + MoE router

    def __post_init__(self):
        if self.vocab_size <= 0:
            raise ValueError("vocab_size must be positive.")
        if self.d_model <= 0:
            raise ValueError("d_model must be positive.")
        if self.max_seq_len <= 0:
            raise ValueError("max_seq_len must be positive.")
        if self.num_layers <= 0:
            raise ValueError("num_layers must be positive.")
        if self.moe_every_n <= 0:
            raise ValueError("moe_every_n must be positive.")
        if self.num_q_heads <= 0 or self.num_kv_heads <= 0:
            raise ValueError("num_q_heads and num_kv_heads must be positive.")
        if self.num_q_heads % self.num_kv_heads != 0:
            raise ValueError(
                "num_q_heads must be divisible by num_kv_heads for grouped-query attention."
            )
        if self.head_dim <= 0 or self.head_dim % 2 != 0:
            raise ValueError("head_dim must be positive and even for RoPE.")
        if self.num_experts <= 0:
            raise ValueError("num_experts must be positive.")
        if not 1 <= self.top_k_experts <= self.num_experts:
            raise ValueError("top_k_experts must be in [1, num_experts].")
        if self.d_ff_dense <= 0 or self.d_ff_moe <= 0:
            raise ValueError("d_ff_dense and d_ff_moe must be positive.")
        if self.num_persona_types <= 0 or self.num_persona_experts <= 0:
            raise ValueError("Persona dimensions must be positive.")
        if self.persona_embedding_dim <= 0 or self.composite_persona_dim <= 0:
            raise ValueError("Persona embedding dimensions must be positive.")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0.0, 1.0).")
        if self.kv_quant_bits <= 0:
            raise ValueError("kv_quant_bits must be positive.")


# ═══════════════════════════════════════════════════════════════════
# 2.  Quantized Linear Layers
# ═══════════════════════════════════════════════════════════════════

class Int8Linear(nn.Module):
    """
    Experimental mock scaffold for INT8 quantization.
    NOTE: Currently converts weights back to float32 during forward pass.
    This does NOT yet provide VRAM reduction or quantized acceleration.
    Replace forward() with bitsandbytes kernel for production.
    """
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.register_buffer(
            "weight",
            torch.randint(-128, 127, (out_features, in_features), dtype=torch.int8),
        )
        self.register_buffer("scale", torch.ones(out_features, dtype=torch.float32))
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None
        self.in_features  = in_features
        self.out_features = out_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w   = self.weight.float() * self.scale.unsqueeze(1)
        out = x @ w.T
        if self.bias is not None:
            out = out + self.bias
        return out.to(x.dtype)

    def extra_repr(self) -> str:
        return f"in={self.in_features}, out={self.out_features}, quant=INT8"


class Int4Linear(nn.Module):
    """
    Experimental mock scaffold for INT4 quantization.
    NOTE: Currently converts weights back to float32 during forward pass.
    This does NOT yet provide VRAM reduction or quantized acceleration.
    Replace forward() with GPTQ / AWQ kernel for production.
    """
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.register_buffer(
            "weight",
            torch.randint(-8, 7, (out_features, in_features), dtype=torch.int8),
        )
        self.register_buffer("scale", torch.ones(out_features, dtype=torch.float32))
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None
        self.in_features  = in_features
        self.out_features = out_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w   = self.weight.float() * self.scale.unsqueeze(1)
        out = x @ w.T
        if self.bias is not None:
            out = out + self.bias
        return out.to(x.dtype)

    def extra_repr(self) -> str:
        return f"in={self.in_features}, out={self.out_features}, quant=INT4"


def build_linear(
    in_features: int,
    out_features: int,
    config: LLMConfig,
    bias: bool = True,
) -> nn.Module:
    """Return the correct linear layer for the current quantization mode."""
    mode = getattr(config, "quantization", "fp16")
    if mode in {"fp16", "bf16"}:
        return nn.Linear(in_features, out_features, bias=bias)
    elif mode == "int8":
        return Int8Linear(in_features, out_features, bias=bias)
    elif mode == "int4":
        return Int4Linear(in_features, out_features, bias=bias)
    else:
        raise ValueError(
            f"Unknown quantization mode '{mode}'.  Valid: 'fp16', 'bf16', 'int8', 'int4'."
        )


# ═══════════════════════════════════════════════════════════════════
# 3.  RMSNorm
# ═══════════════════════════════════════════════════════════════════

class RMSNorm(nn.Module):
    """Root-Mean-Square Layer Normalisation (no mean subtraction)."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps    = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * rms).to(x.dtype) * self.weight


# ═══════════════════════════════════════════════════════════════════
# 4.  Rotary Positional Embedding (RoPE)
# ═══════════════════════════════════════════════════════════════════

class RotaryEmbedding(nn.Module):
    """
    RoPE applied to Q and K.
    θ = 500,000 for 128k context generalisation (LLaMA-3 / Qwen2 style).
    position_ids support enables correct KV-cache decode offsets.
    """

    def __init__(self, head_dim: int, max_seq_len: int = 131_072,
                 theta: float = 500_000.0):
        super().__init__()
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        inv_freq = 1.0 / (
            theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._cached_len = 0
        # Lazy cache: build only an initial window and grow on demand, rather than
        # materializing the full max_seq_len (e.g. 128k) cos/sin cache up front. This
        # RoPE instance is SHARED across all transformer blocks, so the saving applies
        # once for the whole model. For a capped bring-up (max_seq_len<=4096) the initial
        # build already covers it, so no growth happens during training (checkpoint-safe).
        self._build_cache(min(max_seq_len, 4096))

    def _build_cache(self, seq_len: int):
        seq_len = min(int(seq_len), self.max_seq_len)
        t     = torch.arange(seq_len, device=self.inv_freq.device, dtype=torch.float32)
        freqs = torch.outer(t, self.inv_freq)
        emb   = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer("cos_cache", emb.cos(), persistent=False)
        self.register_buffer("sin_cache", emb.sin(), persistent=False)
        self._cached_len = seq_len

    def _ensure_cache(self, needed_len: int):
        """Grow the cache (doubling, capped at max_seq_len) only if a longer sequence appears."""
        if needed_len > self._cached_len:
            self._build_cache(min(self.max_seq_len, max(int(needed_len), self._cached_len * 2)))

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        half = x.shape[-1] // 2
        x1, x2 = x[..., :half], x[..., half:]
        return torch.cat([-x2, x1], dim=-1)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """q, k : (B, num_heads, S, head_dim)"""
        seq = q.shape[2]
        if position_ids is None:
            self._ensure_cache(seq)
            cos = self.cos_cache[:seq].unsqueeze(0).unsqueeze(0)   # (1,1,S,D)
            sin = self.sin_cache[:seq].unsqueeze(0).unsqueeze(0)
        else:
            self._ensure_cache(int(position_ids.max().item()) + 1)
            cos = self.cos_cache[position_ids].unsqueeze(1)        # (B,1,S,D)
            sin = self.sin_cache[position_ids].unsqueeze(1)

        # Cast cos/sin to query dtype here to avoid implicit FP32 upcasting
        # when the model runs in BF16. The buffers are stored as FP32 for
        # maximum precision during cache construction, but we downcast to the
        # working dtype at use-time.
        cos = cos.to(q.dtype)
        sin = sin.to(q.dtype)
        q_rot = q * cos + self._rotate_half(q) * sin
        k_rot = k * cos + self._rotate_half(k) * sin
        return q_rot, k_rot


# ═══════════════════════════════════════════════════════════════════
# 5.  TurboQuant — INT8 KV Cache
# ═══════════════════════════════════════════════════════════════════

class TurboQuant:
    """
    Per-token, per-head INT8 quantisation of KV cache tensors.
    ~47% memory reduction vs FP16; reconstruction MAE < 0.006.
    """

    def __init__(self, bits: int = 8):
        self.bits  = bits
        self.q_max = 2 ** (bits - 1) - 1    # 127 for INT8
        self.q_min = -(2 ** (bits - 1))     # -128

    def compress(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x      : (B, H, S, D)  float
        returns: (quantised int8, scale float (B, H, S, 1))
        """
        scale = x.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
        x_q   = (x / scale * self.q_max).round().clamp(self.q_min, self.q_max)
        return x_q.to(torch.int8), scale

    def decompress(self, x_q: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        return x_q.float() / self.q_max * scale

    def compress_kv(
        self, key: torch.Tensor, value: torch.Tensor,
    ) -> Tuple[Tuple, Tuple]:
        k_q, k_s = self.compress(key)
        v_q, v_s = self.compress(value)
        return (k_q, k_s), (v_q, v_s)

    def decompress_kv(
        self, k_cache: Tuple, v_cache: Tuple,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.decompress(*k_cache), self.decompress(*v_cache)


# ═══════════════════════════════════════════════════════════════════
# 6.  Persona System 
# ═══════════════════════════════════════════════════════════════════

class PersonaEmbeddingStore(nn.Module):
    """
    Learnable embeddings for each persona axis: Pn (narrative), Po (objective),
    Ps (style).  Each axis has a residual adapter for fine-grained conditioning.

    Used in training mode when persona_ids are passed to forward().
    """

    def __init__(self, config: LLMConfig):
        super().__init__()
        self.embeddings = nn.Embedding(
            config.num_persona_types, config.persona_embedding_dim
        )
        self.adapters = nn.ModuleList([
            nn.Linear(
                config.persona_embedding_dim, config.persona_embedding_dim, bias=False
            )
            for _ in range(config.num_persona_types)
        ])

    def forward(self, persona_ids: torch.Tensor) -> torch.Tensor:
        """
        persona_ids : (B, num_persona_types)  integer type indices
        returns     : (B, num_persona_types, persona_embedding_dim)
        """
        embs    = self.embeddings(persona_ids)         # (B, P, D)
        adapted = torch.zeros_like(embs)
        for i, adapter in enumerate(self.adapters):
            mask    = (persona_ids == i).unsqueeze(-1).to(embs.dtype)   # (B, P, 1)
            adapted = adapted + adapter(embs) * mask
        return embs + adapted                          # residual adaptation


class ExpertOfExpertsCompositionRouter(nn.Module):
    """
    Pn, Po, Ps embeddings → multiple composition-strategy experts →
    soft-weighted Composite Persona Context Vector.

    Output: (B, composite_persona_dim)
    """

    def __init__(self, config: LLMConfig):
        super().__init__()
        in_dim = config.num_persona_types * config.persona_embedding_dim
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(in_dim, config.composite_persona_dim),
                nn.SiLU(),
                nn.Linear(config.composite_persona_dim, config.composite_persona_dim),
            )
            for _ in range(config.num_persona_experts)
        ])
        self.router = nn.Linear(in_dim, config.num_persona_experts, bias=False)

    def forward(self, persona_embs: torch.Tensor) -> torch.Tensor:
        """
        persona_embs : (B, num_persona_types, persona_embedding_dim)
        returns      : (B, composite_persona_dim)
        """
        B    = persona_embs.shape[0]
        flat = persona_embs.view(B, -1)                          # (B, P*D)
        gates       = F.softmax(self.router(flat), dim=-1)       # (B, E)
        expert_outs = torch.stack(
            [e(flat) for e in self.experts], dim=1
        )                                                         # (B, E, composite_dim)
        return (gates.unsqueeze(-1) * expert_outs).sum(dim=1)    # (B, composite_dim)


# ═══════════════════════════════════════════════════════════════════
# 7.  Grouped Query Attention  (Persona-Aware + RoPE + TurboQuant)
# ═══════════════════════════════════════════════════════════════════

class GroupedQueryAttention(nn.Module):
    """
    GQA with:
      • 32 Q heads / 8 KV heads  (4× KV compression)
      • RoPE on Q and K  (θ = 500k for 128k context)
      • TurboQuant INT8 KV cache
      • Persona K/V additive shifts  (from composite_ctx)
      • Flash-Attention-compatible via F.scaled_dot_product_attention
    """

    def __init__(self, config: LLMConfig, rope: Optional["RotaryEmbedding"] = None):
        super().__init__()
        self.num_q    = config.num_q_heads
        self.num_kv   = config.num_kv_heads
        self.head_dim = config.head_dim
        self.groups   = self.num_q // self.num_kv

        # Core projections (quantization-aware)
        self.q_proj = build_linear(config.d_model, self.num_q  * self.head_dim, config, bias=False)
        self.k_proj = build_linear(config.d_model, self.num_kv * self.head_dim, config, bias=False)
        self.v_proj = build_linear(config.d_model, self.num_kv * self.head_dim, config, bias=False)
        self.o_proj = build_linear(self.num_q * self.head_dim, config.d_model,  config, bias=False)

        # Persona conditioning: additive shift on K and V
        self.persona_k_shift = nn.Linear(
            config.composite_persona_dim, self.num_kv * self.head_dim, bias=False
        )
        self.persona_v_shift = nn.Linear(
            config.composite_persona_dim, self.num_kv * self.head_dim, bias=False
        )

        # RoPE cos/sin caches depend only on (head_dim, max_seq_len, theta), which
        # are identical across every layer. Share one instance passed down from the
        # parent model to avoid duplicating the (max_seq_len × head_dim) FP32 caches
        # per layer (≈134 MB/layer at 131k context — many GB across a deep model).
        # Falls back to a private instance if constructed standalone.
        self.rope       = rope if rope is not None else RotaryEmbedding(
            self.head_dim, config.max_seq_len, config.rope_theta
        )
        self.turboquant = TurboQuant(config.kv_quant_bits)
        self.dropout_p  = config.dropout

    def forward(
        self,
        x:              torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids:   Optional[torch.Tensor] = None,
        kv_cache:       Optional[Tuple]        = None,
        use_kv_cache:   bool                   = False,
        persona_ctx:    Optional[torch.Tensor] = None,   # (B, composite_persona_dim)
    ) -> Tuple[torch.Tensor, Optional[Tuple]]:
        B, S, _ = x.shape

        # ── Project Q, K, V ───────────────────────────────────────────────
        q = self.q_proj(x).view(B, S, self.num_q,  self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, S, self.num_kv, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, S, self.num_kv, self.head_dim).transpose(1, 2)

        # ── Persona K/V additive shift ────────────────────────────────────
        if persona_ctx is not None:
            k_shift = self.persona_k_shift(persona_ctx).view(B, self.num_kv, 1, self.head_dim)
            v_shift = self.persona_v_shift(persona_ctx).view(B, self.num_kv, 1, self.head_dim)
            k = k + k_shift
            v = v + v_shift

        # ── RoPE ──────────────────────────────────────────────────────────
        q, k = self.rope(q, k, position_ids)

        # ── TurboQuant KV cache (decompress → append → re-compress) ───────
        if use_kv_cache and kv_cache is not None:
            k_past, v_past = self.turboquant.decompress_kv(*kv_cache)
            # [HOTFIX] Cast decompressed KV tensors back to query dtype.
            # Discovered during real BF16 hardware validation: TurboQuant
            # decompresses to FP32, which crashes strict scaled_dot_product_attention.
            k_past = k_past.to(q.dtype)
            v_past = v_past.to(q.dtype)
            k = torch.cat([k_past, k], dim=2)
            v = torch.cat([v_past, v], dim=2)

        new_kv_cache = None
        if use_kv_cache:
            new_kv_cache = self.turboquant.compress_kv(k, v)

        S_kv = k.shape[2]

        # ── GQA: expand KV heads to match Q heads ────────────────────────
        k = k.unsqueeze(2).expand(-1, -1, self.groups, -1, -1).reshape(
            B, self.num_q, S_kv, self.head_dim
        )
        v = v.unsqueeze(2).expand(-1, -1, self.groups, -1, -1).reshape(
            B, self.num_q, S_kv, self.head_dim
        )

        # ── Scaled dot-product attention (Flash-Attention compatible) ──────
        sdpa_mask = None
        # causal=True only during full-sequence prefill (S == S_kv, no past)
        is_causal = (S > 1 and S == S_kv)

        if attention_mask is not None:
            # Convert (B, S_kv) padding mask → additive float (B, 1, S, S_kv)
            pad_mask    = (attention_mask == 0).unsqueeze(1).unsqueeze(2)
            pad_mask    = pad_mask.expand(B, 1, S, S_kv)
            
            if S > 1:
                causal_mask = torch.triu(
                    torch.ones(S, S_kv, device=x.device, dtype=torch.bool), diagonal=1
                ).unsqueeze(0).unsqueeze(0)
                full_mask = pad_mask | causal_mask
            else:
                full_mask = pad_mask

            sdpa_mask = torch.zeros(B, 1, S, S_kv, device=x.device, dtype=q.dtype)
            sdpa_mask.masked_fill_(full_mask, float("-inf"))
            is_causal = False

        dp  = self.dropout_p if self.training else 0.0
        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=sdpa_mask, dropout_p=dp, is_causal=is_causal
        )
        out = out.transpose(1, 2).contiguous().view(B, S, self.num_q * self.head_dim)
        out = self.o_proj(out)
        return out, new_kv_cache


# ═══════════════════════════════════════════════════════════════════
# 8.  Dense SwiGLU FFN
# ═══════════════════════════════════════════════════════════════════

class DenseFFN(nn.Module):
    """SwiGLU FFN used in dense (non-MoE) transformer blocks."""

    def __init__(self, config: LLMConfig):
        super().__init__()
        self.gate_proj = build_linear(config.d_model, config.d_ff_dense, config, bias=False)
        self.up_proj   = build_linear(config.d_model, config.d_ff_dense, config, bias=False)
        self.down_proj = build_linear(config.d_ff_dense, config.d_model, config, bias=False)
        self.drop      = nn.Dropout(config.dropout) if config.dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(
            self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))
        )


# ═══════════════════════════════════════════════════════════════════
# 9.  MoE Expert
# ═══════════════════════════════════════════════════════════════════

class MoEExpert(nn.Module):
    """Single SwiGLU expert inside a MoE layer (hidden=1,536)."""

    def __init__(self, config: LLMConfig):
        super().__init__()
        self.gate_proj = build_linear(config.d_model, config.d_ff_moe, config, bias=False)
        self.up_proj   = build_linear(config.d_model, config.d_ff_moe, config, bias=False)
        self.down_proj = build_linear(config.d_ff_moe, config.d_model, config, bias=False)
        self.drop      = nn.Dropout(config.dropout) if config.dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(
            self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))
        )


# ═══════════════════════════════════════════════════════════════════
# 10. Persona-Aware Top-K Router
# ═══════════════════════════════════════════════════════════════════

class TopKRouter(nn.Module):
    """
    Persona-aware MoE router.
    Gate input: concat(token_hidden, composite_ctx).
    When composite_ctx is None (no persona), zeros are used — same gate, no
    branching logic, consistent weight layout for checkpointing.

    Router gate is intentionally kept in full fp32 — quantising it
    destabilises routing and causes expert collapse.
    """

    def __init__(self, config: LLMConfig):
        super().__init__()
        self.E      = config.num_experts
        self.top_k  = config.top_k_experts
        self.jitter = config.router_jitter
        self._composite_dim = config.composite_persona_dim
        # Single gate: always sees [token; persona_ctx]
        self.gate = nn.Linear(
            config.d_model + config.composite_persona_dim,
            config.num_experts,
            bias=False,
        )
        nn.init.normal_(self.gate.weight, std=0.01)

    def _apply(self, fn):
        """
        Custom _apply to ensure the router gate ALWAYS stays in float32,
        regardless of whether the rest of the model is cast to half/bf16.
        """
        super()._apply(fn)
        self.gate.to(torch.float32)
        return self

    def forward(
        self,
        x: torch.Tensor,                              # (T, d_model)
        persona_ctx_flat: Optional[torch.Tensor] = None,  # (T, composite_dim) or None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if persona_ctx_flat is None:
            persona_ctx_flat = torch.zeros(
                x.shape[0], self._composite_dim, device=x.device, dtype=x.dtype
            )

        # Enforce FP32 for the gating operation to prevent expert collapse.
        # The gate weight is pinned to FP32 in `_apply`, so the *input* must be
        # cast to FP32 as well — otherwise a BF16/FP16 model hits a dtype
        # mismatch inside the gate matmul (BF16 activations × FP32 weight).
        router_in = torch.cat([x, persona_ctx_flat], dim=-1).float()
        logits    = self.gate(router_in)

        if self.training and self.jitter > 0:
            noise  = torch.empty_like(logits).uniform_(1 - self.jitter, 1 + self.jitter)
            logits = logits * noise

        probs = F.softmax(logits, dim=-1)
        top_k_p, top_k_i = torch.topk(probs, self.top_k, dim=-1)

        # Re-normalise so selected expert weights sum to 1
        top_k_p = top_k_p / (top_k_p.sum(-1, keepdim=True) + 1e-9)

        # Cast routing weights back to input dtype for the rest of the computation
        top_k_p = top_k_p.to(x.dtype)

        # ── TELEMETRY ───────────────────────────────────────────────────────────
        if not hasattr(self, "_telemetry"):
            self._telemetry = {}
        with torch.no_grad():
            expert_counts = torch.bincount(top_k_i.flatten(), minlength=self.E)
            max_load = expert_counts.max().item()
            expected_load = (top_k_i.numel()) / self.E
            skew_pct = ((max_load / expected_load) - 1.0) * 100 if expected_load > 0 else 0.0

            self._telemetry["max_load"] = max_load
            self._telemetry["skew_pct"] = skew_pct
            self._telemetry["counts"] = expert_counts.tolist()
        # ────────────────────────────────────────────────────────────────────────

        # Load-balancing auxiliary loss.
        # f = fraction of tokens dispatched to each expert (hard count, no grad).
        # Across data-parallel ranks the LOCAL f only sees this GPU's tokens; average it
        # to the GLOBAL distribution so the router balances the whole cluster, not just the
        # local shard. Quality only — no effect on single-GPU runs.
        # (No-grad statistic, so the collective is autograd-free and symmetric across ranks.)
        with torch.no_grad():
            f = F.one_hot(top_k_i[:, 0], self.E).float().mean(0)   # (E,)
            if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
                dist.all_reduce(f, op=dist.ReduceOp.SUM)
                f = f / dist.get_world_size()
        p   = probs.mean(0)                                           # (E,)
        aux = self.E * (f * p).sum()

        return top_k_p, top_k_i, aux


# ═══════════════════════════════════════════════════════════════════
# 11. MoE Layer  (128 experts, top-2 routing, persona-aware)
# ═══════════════════════════════════════════════════════════════════

class MoELayer(nn.Module):
    """
    Sparse MoE FFN layer.  128 SwiGLU experts, top-2 per token, persona-conditioned routing.

    Batched grouped-GEMM implementation
    ───────────────────────────────────
    Experts are stored as THREE batched weight tensors (not 128 separate nn.Modules):
        w_gate (E, d_model, d_ff)   w_up (E, d_model, d_ff)   w_down (E, d_ff, d_model)
    and computed with torch.bmm over the expert dimension.

    Why: under DeepSpeed ZeRO-3 CPU-offload, 128 separate expert submodules forced
    ~128 parameter all-gathers PER MoE layer (a "parameter-fetching avalanche" that
    makes a large multi-GPU run hang / never finish). Batched params collapse this to
    3 large gathers per layer. Total parameter count is IDENTICAL to the old per-expert
    layout — only the storage shape and dispatch changed.

    Dispatch uses a fixed-capacity buffer (E, capacity, d_model) so per-step memory is
    deterministic and cannot spike on router skew (Switch-Transformer style token
    dropping; dropped tokens ride the block residual). The persona-aware TopKRouter is
    unchanged.
    """

    def __init__(self, config: LLMConfig):
        super().__init__()
        self.config  = config
        self.E       = config.num_experts
        self.top_k   = config.top_k_experts
        self.aux_w   = config.moe_aux_loss_coeff
        self.d_model = config.d_model
        self.d_ff    = config.d_ff_moe
        self.router  = TopKRouter(config)

        E, Dm, Fm = self.E, self.d_model, self.d_ff
        # Batched, bias-free expert weights. Layout is the TRANSPOSE of nn.Linear
        # (nn.Linear stores (out,in) and does x @ W.T; for bmm we want (E, in, out)).
        # NOTE: batched raw Parameters intentionally bypass the int8/int4 mock quant
        # path (build_linear) — experts stay full-precision. Fine for bf16 training.
        self.w_gate = nn.Parameter(torch.empty(E, Dm, Fm))   # x(...,Dm) @ w -> (...,Fm)
        self.w_up   = nn.Parameter(torch.empty(E, Dm, Fm))   # x(...,Dm) @ w -> (...,Fm)
        self.w_down = nn.Parameter(torch.empty(E, Fm, Dm))   # h(...,Fm) @ w -> (...,Dm)
        self._reset_expert_params()

    def _reset_expert_params(self):
        # Replicate nn.Linear default init: kaiming_uniform_(a=sqrt(5)) == U(±1/sqrt(fan_in)),
        # fan_in = in_features. gate/up fan_in = d_model; down fan_in = d_ff_moe.
        bound_in = 1.0 / math.sqrt(self.d_model)   # gate & up
        bound_ff = 1.0 / math.sqrt(self.d_ff)      # down
        nn.init.uniform_(self.w_gate, -bound_in, bound_in)
        nn.init.uniform_(self.w_up,   -bound_in, bound_in)
        nn.init.uniform_(self.w_down, -bound_ff, bound_ff)

    def forward(
        self,
        x: torch.Tensor,
        persona_ctx: Optional[torch.Tensor] = None,    # (B, composite_dim)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, S, D = x.shape
        T       = B * S
        tokens  = x.view(T, D)                                        # (T, D)

        # Expand persona_ctx: (B, composite_dim) → (T, composite_dim)
        persona_flat = None
        if persona_ctx is not None:
            persona_flat = persona_ctx.unsqueeze(1).expand(B, S, -1).reshape(T, -1)

        # Persona-aware routing (UNCHANGED). weights:(T,K) indices:(T,K) aux:fp32 scalar
        weights, indices, aux = self.router(tokens, persona_flat)
        E, K, Fm = self.E, self.top_k, self.d_ff

        # ── Flatten (token, slot) assignments ──
        flat_e = indices.reshape(-1)                                 # (T*K,) expert id, int64
        flat_w = weights.reshape(-1)                                 # (T*K,) routing prob, x.dtype
        tok_id = torch.arange(T, device=x.device).unsqueeze(1).expand(T, K).reshape(-1)  # (T*K,)

        # Position within each expert (segmented prefix count) → unique (e,pos) per kept slot
        onehot = F.one_hot(flat_e, E)                                # (T*K, E) int64
        pos = onehot.cumsum(0).gather(1, flat_e.unsqueeze(1)).squeeze(1) - 1  # (T*K,) 0-indexed

        # ── Capacity: deterministic buffer size, hard-caps per-expert VRAM ──
        # Fixed-capacity token dropping is a TRAINING-time VRAM guarantee (per-step
        # expert buffers can't spike on router skew; dropped tokens ride the residual).
        # At inference/eval it MUST be disabled: capacity is derived from T = B*S, so a
        # full forward (T = whole sequence) drops overflow tokens while incremental
        # KV-cache decode (T = 1 per step) never does — the two diverge, breaking the
        # KV-cache==no-cache invariant. When not training, size the buffer to the actual
        # max per-expert load so every token is kept and decode stays exact.
        if self.training:
            cf = getattr(self.config, "capacity_factor", 0.0)
            if cf is None or cf <= 0:
                cf = 1.25                                            # safe default when unset/0
            capacity = max(1, math.ceil(K * T / E * cf))
            keep = pos < capacity                                    # drop overflow (rides residual)
        else:
            capacity = int(pos.max().item()) + 1 if pos.numel() else 1
            keep = torch.ones_like(pos, dtype=torch.bool)           # eval: keep every token

        e_k = flat_e[keep]                                           # (N,)
        p_k = pos[keep]                                              # (N,) in [0, capacity)
        t_k = tok_id[keep]                                           # (N,)
        w_k = flat_w[keep]                                           # (N,) x.dtype

        # ── Dispatch tokens into (E, capacity, D); each (e,p) is unique → no collision ──
        x_disp = tokens.new_zeros(E, capacity, D)                    # (E, cap, D)
        x_disp[e_k, p_k] = tokens[t_k]

        # ── Batched expert compute (grouped GEMM over the E dim, no Python loop) ──
        h = torch.bmm(x_disp, self.w_gate)                          # (E, cap, Fm)
        g = torch.bmm(x_disp, self.w_up)                            # (E, cap, Fm)
        a = F.silu(h) * g                                           # (E, cap, Fm)
        y = torch.bmm(a, self.w_down)                              # (E, cap, D)

        # ── Weighted combine back to (T, D) ──
        contrib = y[e_k, p_k] * w_k.unsqueeze(1)                    # (N, D)
        out = torch.zeros_like(tokens).index_add(0, t_k, contrib)   # out-of-place (checkpoint-safe)

        return out.view(B, S, D), aux * self.aux_w


# ═══════════════════════════════════════════════════════════════════
# 12. Transformer Block  (Dense or MoE, persona-aware)
# ═══════════════════════════════════════════════════════════════════

class TransformerBlock(nn.Module):
    """
    Pre-norm transformer block:
      x → RMSNorm → Persona-Aware GQA → + residual
        → RMSNorm → Dense SwiGLU | Persona-Aware MoE → + residual
    """

    def __init__(self, config: LLMConfig, use_moe: bool = False,
                 rope: Optional["RotaryEmbedding"] = None):
        super().__init__()
        self.use_moe = use_moe
        self.norm1   = RMSNorm(config.d_model, config.rms_norm_eps)
        self.norm2   = RMSNorm(config.d_model, config.rms_norm_eps)
        self.attn    = GroupedQueryAttention(config, rope=rope)
        self.ffn     = MoELayer(config) if use_moe else DenseFFN(config)

    def forward(
        self,
        x:              torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids:   Optional[torch.Tensor] = None,
        kv_cache:       Optional[Tuple]        = None,
        use_kv_cache:   bool                   = False,
        persona_ctx:    Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[Tuple], torch.Tensor]:
        # Attention sub-layer (pre-norm)
        attn_out, kv_out = self.attn(
            self.norm1(x), attention_mask, position_ids,
            kv_cache, use_kv_cache, persona_ctx,
        )
        x = x + attn_out

        # FFN sub-layer (pre-norm)
        aux_loss = torch.tensor(0.0, device=x.device, dtype=torch.float32)
        normed   = self.norm2(x)
        if self.use_moe:
            ffn_out, aux_loss = self.ffn(normed, persona_ctx=persona_ctx)
        else:
            ffn_out = self.ffn(normed)
        x = x + ffn_out

        return x, kv_out, aux_loss


# ═══════════════════════════════════════════════════════════════════
# 13. Full model  (VantaTransformer — unified, persona-aware, trainable)
# ═══════════════════════════════════════════════════════════════════

class VantaTransformer(nn.Module):
    """
    The full VantaLLM model — a single, config-driven transformer class.

    Model size is set entirely by ``LLMConfig``; the largest scaling-reference
    config has an analytically estimated ~125B stored parameters with ~13B
    active per token. Active count = always-on dense FFN + attention + embedding
    + the sparse top-2 MoE experts (the dense FFN, not the MoE branch, dominates
    the active budget). Parameter counts for the large configs are analytical
    estimates, not measured; call ``model.parameter_count()`` to verify for a
    given config.

    Token Embedding
      + 0.1 × persona_signal  (injected at embedding layer)
      ↓
    N × TransformerBlock  (dense / MoE alternating)
      ↓
    Final RMSNorm → LM Head (tied weights)

    Persona modes
    ─────────────
    Training  : pass persona_ids (B, P)  → PersonaEmbeddingStore → EoE Router
    Inference : pass persona_vector (B, d_model) L2-normed  (server.py path)
    Both None : plain forward, no persona conditioning

    Return value
    ────────────
    With labels=None  (inference / server.py):
        (logits, kv_caches, aux_loss)  ← tuple, backward-compatible

    With labels given  (SFT / PPO training):
        {"logits": ..., "loss": ..., "aux_loss": ..., "kv_caches": ...}
    """

    def __init__(self, config: LLMConfig):
        super().__init__()
        self.config = config

        # ── Token embedding  (RENAMED from self.embed to avoid nn.Module.embed clash) ──
        self.token_embed = nn.Embedding(config.vocab_size, config.d_model)
        self.embed_drop  = nn.Dropout(config.dropout)

        # ── Persona system ──────────────────────────────────────────────────
        self.persona_store  = PersonaEmbeddingStore(config)
        self.persona_router = ExpertOfExpertsCompositionRouter(config)
        # Projects composite_ctx (composite_dim) → d_model for embedding injection (training)
        self.persona_to_hidden = nn.Linear(
            config.composite_persona_dim, config.d_model, bias=False
        )
        # Projects persona_vector (d_model) → composite_dim for GQA/MoE (inference)
        self.persona_vec_to_composite = nn.Linear(
            config.d_model, config.composite_persona_dim, bias=False
        )

        # ── Shared RoPE ─────────────────────────────────────────────────────
        # One rotary-embedding instance for the whole model. The cos/sin caches
        # are position-only (independent of layer), so a single shared module
        # avoids allocating identical (max_seq_len × head_dim) FP32 caches in
        # every attention layer. Buffers are persistent=False, so this does not
        # affect checkpoints; `.to()`/`.half()` remain idempotent on the shared
        # instance.
        self.rope = RotaryEmbedding(
            config.head_dim, config.max_seq_len, config.rope_theta
        )

        # ── Transformer layers ──────────────────────────────────────────────
        # MoE on every 2nd block: layers 1, 3, 5, … (odd indices)
        self.layers = nn.ModuleList([
            TransformerBlock(config, use_moe=(i % config.moe_every_n == 1), rope=self.rope)
            for i in range(config.num_layers)
        ])

        self.final_norm = RMSNorm(config.d_model, config.rms_norm_eps)

        # LM head stays nn.Linear so weight-tying works across all quant modes
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.lm_head.weight = self.token_embed.weight   # weight tying

        self._init_weights()
        log.info(
            f"VantaTransformer initialised | quant={config.quantization.upper()} | "
            f"layers={config.num_layers} | experts={config.num_experts}"
        )

    # ── Weight initialisation ───────────────────────────────────────────────
    def _init_weights(self):
        nn.init.normal_(self.token_embed.weight, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear) and m.weight is not self.token_embed.weight:
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            # Int8/Int4 scale buffers are already initialised to ones;
            # their random integer weights are intentional for mock testing.

    # ── Public helper: token embeddings  (used by server.py for persona vectors) ──
    def embed(self, ids: torch.Tensor) -> torch.Tensor:
        """
        Return raw token embeddings (before positional / persona injection).
        ids : (B, S) → (B, S, d_model)

        server.py calls this to compute persona vectors:
            emb = model.embed(tag_ids)   # (1, seq, d_model)
            vec = emb[:, -1, :]          # last-token representation
            vec = vec / vec.norm(...)    # L2-normalise
        """
        return self.token_embed(ids)

    # ── Forward pass ────────────────────────────────────────────────────────
    def forward(
        self,
        input_ids:      torch.Tensor,
        attention_mask: Optional[torch.Tensor]       = None,
        position_ids:   Optional[torch.Tensor]       = None,
        kv_caches:      Optional[List[Optional[Tuple]]] = None,
        use_kv_cache:   bool                         = False,
        # ── Persona conditioning (mutually exclusive; both None = no persona) ──
        persona_vector: Optional[torch.Tensor]       = None,   # inference (B, d_model) L2-normed
        persona_ids:    Optional[torch.Tensor]       = None,   # training  (B, num_persona_types)
        # ── Training loss ────────────────────────────────────────────────────
        labels:         Optional[torch.Tensor]       = None,   # (B, S) causal LM labels
        return_dict:    bool                         = False,  
    ) -> Union[Tuple, Dict[str, Any]]:
        """
        Returns
        ───────
        labels=None  →  (logits, kv_caches, aux_loss)          tuple  [server.py compat]
        labels given →  {"logits", "loss", "aux_loss", "kv_caches"}  dict  [training]
        """
        B, S = input_ids.shape

        # ── Token embeddings ─────────────────────────────────────────────────
        x = self.embed_drop(self.token_embed(input_ids))   # (B, S, d_model)

        # ── Persona context computation ──────────────────────────────────────
        composite_ctx: Optional[torch.Tensor] = None

        if persona_ids is not None:
            # Training path: learnable persona embedding system
            persona_embs  = self.persona_store(persona_ids)       # (B, P, persona_dim)
            raw_ctx       = self.persona_router(persona_embs)     # (B, composite_dim)
            composite_ctx = raw_ctx
            # Project composite → d_model for embedding-level injection
            ctx_signal    = self.persona_to_hidden(raw_ctx)       # (B, d_model)
            x = x + 0.1 * ctx_signal.unsqueeze(1)

        elif persona_vector is not None:
            # Inference path: server.py supplies L2-normed last-token hidden state
            x = x + 0.1 * persona_vector.unsqueeze(1)
            # Project d_model → composite_dim for GQA / MoE conditioning
            composite_ctx = self.persona_vec_to_composite(persona_vector)  # (B, composite_dim)

        # ── KV cache initialisation ──────────────────────────────────────────
        if kv_caches is None:
            kv_caches = [None] * self.config.num_layers

        # ── Position IDs (KV-cache-offset aware) ────────────────────────────
        if position_ids is None:
            past_seq_len = 0
            if use_kv_cache and kv_caches and kv_caches[0] is not None:
                # kv_caches[0] = (k_cache, v_cache); k_cache = (k_q, k_s)
                past_seq_len = kv_caches[0][0][0].shape[2]
            position_ids = torch.arange(
                past_seq_len, past_seq_len + S, device=input_ids.device
            ).unsqueeze(0).expand(B, -1)

        # ── 94 × Transformer layers ──────────────────────────────────────────
        new_kv_caches  = []
        total_aux_loss = torch.tensor(0.0, device=x.device, dtype=torch.float32)

        for i, layer in enumerate(self.layers):
            if getattr(self.config, "gradient_checkpointing", False) and self.training:
                from torch.utils.checkpoint import checkpoint
                x, kv_out, aux = checkpoint(
                    layer,
                    x, attention_mask, position_ids,
                    kv_caches[i], use_kv_cache, composite_ctx,
                    use_reentrant=False
                )
            else:
                x, kv_out, aux = layer(
                    x, attention_mask, position_ids,
                    kv_caches[i], use_kv_cache,
                    persona_ctx=composite_ctx,
                )
            new_kv_caches.append(kv_out)
            total_aux_loss = total_aux_loss + aux

        # ── Final norm + LM head ─────────────────────────────────────────────
        x      = self.final_norm(x)
        logits = self.lm_head(x)                # (B, S, vocab)

        # ── Causal LM Training / Dict Return Strategy ──
        if labels is not None or return_dict:
            res = {
                "logits":        logits,
                "kv_caches":     new_kv_caches,
                "aux_loss":      total_aux_loss,
                "hidden_states": x,
            }
            if labels is not None:
                shift_logits = logits[:, :-1].contiguous()
                shift_labels = labels[:, 1:].contiguous()
                lm_loss = F.cross_entropy(
                    shift_logits.view(-1, self.config.vocab_size),
                    shift_labels.view(-1),
                    ignore_index=-100,
                )
                res["loss"] = lm_loss + total_aux_loss
            return res

        # ── Inference: legacy tuple return (server.py backward-compatible) ──
        return logits, new_kv_caches, total_aux_loss

    # ── Parameter count breakdown ────────────────────────────────────────────
    def parameter_count(self) -> Dict[str, int]:
        embed_p   = sum(p.numel() for p in self.token_embed.parameters())
        persona_p = sum(
            p.numel() for m in [
                self.persona_store, self.persona_router,
                self.persona_to_hidden, self.persona_vec_to_composite,
            ]
            for p in m.parameters()
        )
        attn_p    = sum(p.numel() for l in self.layers for p in l.attn.parameters())
        dense_p   = sum(
            p.numel() for l in self.layers if not l.use_moe for p in l.ffn.parameters()
        )
        moe_p     = sum(
            p.numel() for l in self.layers if l.use_moe for p in l.ffn.parameters()
        )
        norm_p    = sum(
            p.numel() for l in self.layers
            for p in list(l.norm1.parameters()) + list(l.norm2.parameters())
        )
        total     = sum(p.numel() for p in self.parameters())
        return {
            "embedding":        embed_p,
            "persona_system":   persona_p,
            "attention (all)":  attn_p,
            "dense_ffn (all)":  dense_p,
            "moe_ffn (all)":    moe_p,
            "layer_norms":      norm_p,
            "lm_head (tied)":   0,
            "TOTAL":            total,
        }


# ═══════════════════════════════════════════════════════════════════
# 14. RL Training Scaffold 
# ═══════════════════════════════════════════════════════════════════

class ValueHead(nn.Module):
    """
    Scalar value estimator for PPO advantage computation.
    Takes the final hidden states of the policy network and outputs per-token values.
    """

    def __init__(self, config: LLMConfig):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.d_model, config.d_model // 2),
            nn.SiLU(),
            nn.Linear(config.d_model // 2, 1),
        )

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        """hidden : (B, T, d_model) → values : (B, T)"""
        return self.net(hidden).squeeze(-1)


class RewardModel(nn.Module):
    """
    Preference reward model.  Shares architecture with policy but has
    separate weights — never share these with the policy network.

    Forward: input_ids → mean-pooled hidden → scalar reward per sequence.
    """

    def __init__(self, config: LLMConfig):
        super().__init__()
        self.backbone    = VantaTransformer(config)
        self.reward_head = nn.Linear(config.d_model, 1, bias=True)

    def forward(
        self,
        input_ids:      torch.Tensor,
        persona_ids:    Optional[torch.Tensor] = None,
        persona_vector: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Returns scalar reward per sequence: (B,)

        Runs the full 94-layer backbone so the reward head sees the complete
        contextualized representation of the sequence — not raw token embeddings.
        Using labels=input_ids triggers the dict return which includes hidden_states.
        """
        out    = self.backbone(
            input_ids,
            persona_ids=persona_ids,
            persona_vector=persona_vector,
            return_dict=True,         
        )
        hidden = out["hidden_states"]                           
        reward = self.reward_head(hidden.mean(dim=1)).squeeze(-1)  
        return reward


class PPOTrainer:
    """
    PPO with clipped surrogate objective + KL penalty.

    Components
    ──────────
    policy       — VantaTransformer being fine-tuned
    ref_policy   — frozen SFT checkpoint (KL anchor)
    value_head   — ValueHead for advantage estimation
    reward_model — RewardModel for preference scoring

    Usage
    ─────
    trainer = PPOTrainer(policy, ref_policy, value_head, reward_model)

    # Rollout loop
    advantages, returns = trainer.compute_advantages(rewards, values)
    loss = trainer.compute_ppo_loss(input_ids, persona_ids, old_log_probs, advantages, returns)
    loss.backward()
    """

    def __init__(
        self,
        policy:       VantaTransformer,
        ref_policy:   VantaTransformer,
        value_head:   ValueHead,
        reward_model: RewardModel,
        clip_eps:     float = 0.2,
        kl_coef:      float = 0.1,
        gamma:        float = 0.99,
        lam:          float = 0.95,
    ):
        self.policy       = policy
        self.ref_policy   = ref_policy
        self.value_head   = value_head
        self.reward_model = reward_model
        self.clip_eps     = clip_eps
        self.kl_coef      = kl_coef
        self.gamma        = gamma
        self.lam          = lam

    def compute_ppo_loss(
        self,
        input_ids:     torch.Tensor,              # (B, T)
        persona_ids:   Optional[torch.Tensor],    # (B, P) or None
        old_log_probs: torch.Tensor,              # (B, T-1)
        advantages:    torch.Tensor,              # (B, T-1)  normalised
        returns:       torch.Tensor,              # (B, T-1)
    ) -> torch.Tensor:
        """
        PPO loss = clipped surrogate + KL penalty + 0.5 × value loss.
        All three terms are averaged over the batch.
        """
        # ── Current policy log probs ──────────────────────────────────────
        out           = self.policy(input_ids, persona_ids=persona_ids, labels=input_ids)
        logits        = out["logits"][:, :-1]                    # (B, T-1, V)
        log_probs     = F.log_softmax(logits, dim=-1)
        tok_log_probs = log_probs.gather(
            -1, input_ids[:, 1:].unsqueeze(-1)
        ).squeeze(-1)                                             # (B, T-1)

        # ── Reference policy log probs (KL regularisation) ───────────────
        with torch.no_grad():
            # ref_policy returns tuple when labels=None
            ref_out           = self.ref_policy(input_ids, persona_ids=persona_ids)
            ref_logits        = ref_out[0][:, :-1]               # (B, T-1, V)
            ref_log_probs     = F.log_softmax(ref_logits, dim=-1)
            ref_tok_log_probs = ref_log_probs.gather(
                -1, input_ids[:, 1:].unsqueeze(-1)
            ).squeeze(-1)                                         # (B, T-1)

        # ── PPO clipped surrogate objective ───────────────────────────────
        ratio         = (tok_log_probs - old_log_probs).exp()
        clipped_ratio = ratio.clamp(1.0 - self.clip_eps, 1.0 + self.clip_eps)
        policy_loss   = -torch.min(
            ratio * advantages, clipped_ratio * advantages
        ).mean()

        # ── KL penalty (prevents policy collapse) ────────────────────────
        kl_penalty = (tok_log_probs - ref_tok_log_probs).mean()

        # ── Value function loss ───────────────────────────────────────────
        # hidden_states is the output of the final RMSNorm — after all 94
        # transformer layers, before the lm_head projection.  The value head
        # must see the full contextualized representation to estimate state
        # values correctly; raw token embeddings (pre-transformer) have no
        # awareness of sequence context, making PPO degenerate.
        hidden     = out["hidden_states"][:, :-1]                # (B, T-1, d_model)
        values     = self.value_head(hidden)                      # (B, T-1)
        value_loss = F.mse_loss(values, returns)

        total_loss = policy_loss + self.kl_coef * kl_penalty + 0.5 * value_loss
        return total_loss

    @torch.no_grad()
    def compute_advantages(
        self,
        rewards: torch.Tensor,   # (B, T)
        values:  torch.Tensor,   # (B, T)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        GAE-Lambda advantage estimation.
        Returns (advantages, returns), both (B, T).
        advantages are mean-centred and std-normalised.
        """
        B, T    = rewards.shape
        advs    = torch.zeros_like(rewards)
        lastgae = torch.zeros(B, device=rewards.device)

        for t in reversed(range(T)):
            next_val = values[:, t + 1] if t + 1 < T else torch.zeros(B, device=rewards.device)
            delta    = rewards[:, t] + self.gamma * next_val - values[:, t]
            lastgae  = delta + self.gamma * self.lam * lastgae
            advs[:, t] = lastgae

        returns = advs + values
        advs    = (advs - advs.mean()) / (advs.std() + 1e-8)
        return advs, returns


# ═══════════════════════════════════════════════════════════════════
# 15. Weight Loading
# ═══════════════════════════════════════════════════════════════════

def load_weights(model: VantaTransformer, path: str, config: LLMConfig) -> None:
    """Quantization-aware weight loader."""
    def extract_model_state_dict(checkpoint_data: Any) -> Dict[str, torch.Tensor]:
        if isinstance(checkpoint_data, dict) and "model" in checkpoint_data:
            model_state = checkpoint_data["model"]
            if isinstance(model_state, dict):
                return model_state
            raise TypeError("Checkpoint 'model' entry is not a state_dict.")
        if isinstance(checkpoint_data, dict):
            return checkpoint_data
        raise TypeError("Unsupported checkpoint format: expected a state_dict or trainer checkpoint dict.")

    mode = config.quantization

    if mode in {"fp16", "bf16"}:
        log.info(f"Loading {mode.upper()} weights from: {path}")
        try:
            from safetensors.torch import load_file as load_safetensors
            state = load_safetensors(path)
            log.info("  Format: safetensors")
        except (ImportError, Exception):
            # weights_only=False: preserve backward-compatible loading for trusted local checkpoints
            state = torch.load(path, map_location="cpu", weights_only=False)
            log.info("  Format: torch checkpoint")
        state_dict = extract_model_state_dict(state)
        load_result = model.load_state_dict(state_dict, strict=False)
        loaded_keys = set(model.state_dict().keys()) & set(state_dict.keys())
        if not loaded_keys:
            raise ValueError(
                f"Checkpoint at '{path}' did not match any model parameters."
            )
        if load_result.missing_keys:
            log.warning(
                "  Missing keys during load (%d): %s",
                len(load_result.missing_keys),
                load_result.missing_keys[:8],
            )
        if load_result.unexpected_keys:
            log.warning(
                "  Unexpected keys during load (%d): %s",
                len(load_result.unexpected_keys),
                load_result.unexpected_keys[:8],
            )
        log.info("  FP16 weights loaded.")

    elif mode == "int8":
        raise NotImplementedError(
            "INT8 checkpoint loading is not implemented. "
            "Use QUANT_MODE=fp16/bf16 for real checkpoints, or add a real INT8 loader."
        )

    elif mode == "int4":
        raise NotImplementedError(
            "INT4 checkpoint loading is not implemented. "
            "Use QUANT_MODE=fp16/bf16 for real checkpoints, or add a real INT4 loader."
        )

    else:
        raise ValueError(f"Unknown quantization mode '{mode}'.")


# ═══════════════════════════════════════════════════════════════════
# 16. Model Loader
# ═══════════════════════════════════════════════════════════════════

def load_model(
    model_path: Optional[str] = None,
    tiny: bool = False,
) -> VantaTransformer:
    """
    Build and optionally load an VantaTransformer.

    Environment variables
    ─────────────────────
    QUANT_MODE   : "fp16" | "bf16" | "int8" | "int4"   (default: "bf16")
    TINY_MODEL   : "1"                         (4-layer dev model, no GPU needed)

    Args
    ────
    model_path : path to weights (.pt / .safetensors).  None → random weights.
    tiny       : override TINY_MODEL env var programmatically.
    """
    def _env_flag(name: str) -> bool:
        value = os.environ.get(name)
        if value is None:
            return False
        return value.strip().lower() in {"1", "true", "yes", "on"}

    def _load_config_from_env() -> Optional[LLMConfig]:
        config_path = os.environ.get("MODEL_CONFIG", "").strip()
        if not config_path:
            return None
        with open(config_path, "r", encoding="utf-8") as f:
            raw_config = json.load(f)
        if not isinstance(raw_config, dict):
            raise ValueError(f"MODEL_CONFIG must contain a JSON object: {config_path}")
        raw_config = dict(raw_config)
        if os.environ.get("QUANT_MODE"):
            raw_config["quantization"] = quant_mode
        else:
            raw_config.setdefault("quantization", quant_mode)
        log.info(f"Model config path  : {config_path}")
        return LLMConfig(**raw_config)

    quant_mode = os.environ.get("QUANT_MODE", "bf16").lower()
    use_tiny   = tiny or _env_flag("TINY_MODEL")

    # Safety Guard: Prevent accidental 125B instantiation on small GPUs
    if not use_tiny and not os.environ.get("MODEL_CONFIG", "").strip():
        raise RuntimeError(
            "CRITICAL OPERATIONAL ERROR: TINY_MODEL=0 (or unset) but no MODEL_CONFIG provided.\n"
            "The default VantaLLM architecture is ~125B parameters and will not fit on a single workstation GPU.\n"
            "To run on a single GPU, specify a smaller config:\n"
            "  export MODEL_CONFIG=configs/vanta_567m.json\n"
            "Or use TINY_MODEL=1 for local CPU smoke tests."
        )

    log.info(f"Quantization mode : {quant_mode.upper()}")
    log.info(f"Tiny model        : {use_tiny}")

    if use_tiny:
        config = LLMConfig(
            vocab_size             = 151_003,
            d_model                = 256,
            max_seq_len            = 512,
            num_layers             = 4,
            num_q_heads            = 8,
            num_kv_heads           = 2,
            head_dim               = 32,
            d_ff_dense             = 512,
            num_experts            = 8,
            top_k_experts          = 2,
            d_ff_moe               = 128,
            moe_every_n            = 2,
            quantization           = quant_mode,
            composite_persona_dim  = 128,   # scaled for tiny model
        )
    else:
        config = _load_config_from_env() or LLMConfig(quantization=quant_mode)

    model = VantaTransformer(config)

    if model_path:
        load_weights(model, model_path, config)

    return model


# ═══════════════════════════════════════════════════════════════════
# 17. Greedy Inference Helper
# ═══════════════════════════════════════════════════════════════════

@torch.no_grad()
def greedy_generate(
    model:          VantaTransformer,
    input_ids:      torch.Tensor,
    max_new_tokens: int = 50,
    persona_vector: Optional[torch.Tensor] = None,
    persona_ids:    Optional[torch.Tensor] = None,
    eos_token_id:   int = 1,
) -> torch.Tensor:
    """
    Simple greedy decoding with TurboQuant KV cache.
    Returns the full sequence (prompt + generated tokens).
    Supports both inference-mode (persona_vector) and training-mode (persona_ids) persona.
    """
    model.eval()
    kv_caches = None
    generated = input_ids

    for step in range(max_new_tokens):
        ids = generated if step == 0 else generated[:, -1:]
        out = model(
            ids,
            kv_caches=kv_caches,
            use_kv_cache=True,
            persona_vector=persona_vector,
            persona_ids=persona_ids,
        )
        # out is tuple (logits, kv_caches, aux_loss) since no labels
        logits, kv_caches, _ = out
        next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        generated  = torch.cat([generated, next_token], dim=1)

        # Stop when ALL sequences in the batch have generated EOS.
        # .item() on a (B,1) tensor crashes for B>1; use .all() first.
        if (next_token == eos_token_id).all().item():
            break

    return generated


# ═══════════════════════════════════════════════════════════════════
# 18. Demo / Sanity Check
# ═══════════════════════════════════════════════════════════════════

def _demo():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    torch.manual_seed(42)

    print("\n" + "=" * 70)
    print("  VantaLLM  —  Architecture Sanity Check")
    print("=" * 70)

    for mode in ["fp16", "int8", "int4"]:
        print(f"\n── MODE: {mode.upper()} ──────────────────────────────────────────")
        cfg = LLMConfig(
            vocab_size=151_003, d_model=256, max_seq_len=512,
            num_layers=4, num_q_heads=8, num_kv_heads=2, head_dim=32,
            d_ff_dense=512, num_experts=8, top_k_experts=2, d_ff_moe=128,
            moe_every_n=2, quantization=mode, composite_persona_dim=128,
        )
        model = VantaTransformer(cfg)
        counts = model.parameter_count()
        for k, v in counts.items():
            print(f"  {k:<25} {v:>12,}")

        B, S = 2, 16
        ids  = torch.randint(0, cfg.vocab_size, (B, S))

        # ── Test 1: plain forward (no persona) ──────────────────────────
        logits, kv, aux = model(ids)
        assert logits.shape == (B, S, cfg.vocab_size), "shape mismatch"
        assert not logits.isnan().any(), "NaN in logits"
        print(f"\n  [1] Plain forward   logits={tuple(logits.shape)}  aux={aux.item():.6f}")

        # ── Test 2: training forward with persona_ids ────────────────────
        p_ids = torch.randint(0, cfg.num_persona_types, (B, cfg.num_persona_types))
        out   = model(ids, persona_ids=p_ids, labels=ids)
        assert isinstance(out, dict), "expected dict return"
        assert "loss" in out and "logits" in out
        print(f"  [2] Training forward  loss={out['loss'].item():.4f}  aux={out['aux_loss'].item():.6f}")

        # ── Test 3: inference forward with persona_vector ────────────────
        p_vec = torch.randn(B, cfg.d_model)
        p_vec = p_vec / (p_vec.norm(dim=-1, keepdim=True) + 1e-6)
        logits2, _, _ = model(ids, persona_vector=p_vec)
        print(f"  [3] Inference forward logits={tuple(logits2.shape)}")

        # ── Test 4: embed() helper (server.py path) ──────────────────────
        emb = model.embed(ids[:1, :4])
        assert emb.shape == (1, 4, cfg.d_model), "embed() shape mismatch"
        vec = emb[:, -1, :]
        vec = vec / (vec.norm(dim=-1, keepdim=True) + 1e-6)
        print(f"  [4] embed() helper    shape={tuple(emb.shape)}  L2-norm={vec.norm().item():.4f}")

        # ── Test 5: greedy generate ───────────────────────────────────────
        prompt = torch.randint(0, cfg.vocab_size, (1, 5))
        gen    = greedy_generate(model, prompt, max_new_tokens=8, persona_vector=vec)
        print(f"  [5] greedy_generate   {prompt.shape[1]} → {gen.shape[1]} tokens")
        print(f"  ✓ {mode.upper()} mode passed")

    # ── TurboQuant check ─────────────────────────────────────────────────
    print("\n── TurboQuant KV Cache ─────────────────────────────────────────────")
    tq         = TurboQuant(bits=8)
    k_dummy    = torch.randn(2, 8, 16, 32)
    v_dummy    = torch.randn(2, 8, 16, 32)
    k_c, v_c   = tq.compress_kv(k_dummy, v_dummy)
    k_d, v_d   = tq.decompress_kv(k_c, v_c)
    k_err      = (k_dummy - k_d).abs().mean().item()
    v_err      = (v_dummy - v_d).abs().mean().item()
    compress   = k_dummy.element_size() / k_c[0].element_size()
    print(f"  Compression : {compress:.0f}× (FP32 → INT8)")
    print(f"  K recon err : {k_err:.6f}")
    print(f"  V recon err : {v_err:.6f}")
    assert k_err < 0.02 and v_err < 0.02, "TurboQuant reconstruction error too high"

    # ── PPO scaffold check ───────────────────────────────────────────────
    print("\n── PPO Scaffold ────────────────────────────────────────────────────")
    cfg_tiny = LLMConfig(
        vocab_size=1000, d_model=64, max_seq_len=32,
        num_layers=2, num_q_heads=4, num_kv_heads=2, head_dim=16,
        d_ff_dense=128, num_experts=4, top_k_experts=2, d_ff_moe=64,
        moe_every_n=2, quantization="fp16", composite_persona_dim=32,
        persona_embedding_dim=16, num_persona_types=3, num_persona_experts=2,
    )
    policy      = VantaTransformer(cfg_tiny)
    ref_policy  = VantaTransformer(cfg_tiny)
    value_head  = ValueHead(cfg_tiny)
    reward_model= RewardModel(cfg_tiny)
    trainer     = PPOTrainer(policy, ref_policy, value_head, reward_model)

    B, T = 2, 8
    ids       = torch.randint(0, cfg_tiny.vocab_size, (B, T))
    p_ids_t   = torch.randint(0, cfg_tiny.num_persona_types, (B, cfg_tiny.num_persona_types))
    rewards   = torch.randn(B, T)
    with torch.no_grad():
        # Must use the full forward pass to get contextualized hidden states.
        # token_embed alone gives uncontextualized vectors — value head needs
        # transformer output (after all layers + final RMSNorm) to be meaningful.
        out_for_values = policy(ids, persona_ids=p_ids_t, labels=ids)
        hidden = out_for_values["hidden_states"]   # (B, T, d_model) — post-transformer
        values = value_head(hidden)
    advs, returns = trainer.compute_advantages(rewards, values.detach())
    with torch.no_grad():
        out_ref = policy(ids, persona_ids=p_ids_t)
        lp = F.log_softmax(out_ref[0][:, :-1], dim=-1)
        old_lp = lp.gather(-1, ids[:, 1:].unsqueeze(-1)).squeeze(-1)
    loss = trainer.compute_ppo_loss(ids, p_ids_t, old_lp.detach(), advs[:, :-1], returns[:, :-1])
    print(f"  PPO loss : {loss.item():.4f}")
    assert not loss.isnan(), "NaN in PPO loss"
    print("  ✓ PPO scaffold passed")

    print("\n" + "=" * 70)
    print("  All checks passed ✓")
    print("=" * 70)


if __name__ == "__main__":
    _demo()
