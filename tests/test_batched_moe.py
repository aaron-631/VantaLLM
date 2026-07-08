"""
CPU correctness tests for the batched grouped-GEMM MoELayer.

Run:  .venv/bin/python tests/test_batched_moe.py

These validate (on a tiny config, fp32) the parts that cannot be checked on a GPU
before a large distributed run:
  1. gradients flow to w_gate/w_up/w_down and the input,
  2. the batched dispatch/combine == a naive per-expert weighted sum (unbounded capacity),
  3. the forward is checkpoint-safe (grads match with/without torch.utils.checkpoint),
  4. the T < E edge case runs and stays finite,
  5. persona-conditioned routing still delivers gradient to the router gate,
  6. bf16 activations + fp32-forced gate do not crash.
The persona-aware TopKRouter is used unchanged in every test.
"""
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import torch
import torch.nn.functional as F
import torch.utils.checkpoint as ckpt
from vantallm.model.transformer import LLMConfig, MoELayer

torch.manual_seed(0)
D_MODEL, E = 64, 8


def make_cfg(capacity_factor=0.0):
    return LLMConfig(
        d_model=D_MODEL, num_experts=E, d_ff_moe=32, top_k_experts=2,
        composite_persona_dim=16, router_jitter=0.0,
        capacity_factor=capacity_factor, moe_aux_loss_coeff=0.01, dropout=0.0,
    )


def naive_reference(layer, x, persona_ctx=None):
    """Per-expert weighted sum using the SAME batched weights + SAME router (no capacity cap)."""
    B, S, Dm = x.shape
    T = B * S
    tokens = x.reshape(T, Dm)
    persona_flat = None
    if persona_ctx is not None:
        persona_flat = persona_ctx.unsqueeze(1).expand(B, S, -1).reshape(T, -1)
    weights, indices, aux = layer.router(tokens, persona_flat)
    out = torch.zeros_like(tokens)
    for e in range(layer.E):
        t_idx, k_slot = (indices == e).nonzero(as_tuple=True)
        if t_idx.numel() == 0:
            continue
        et = tokens[t_idx]
        ew = weights[t_idx, k_slot, None]
        h = F.silu(et @ layer.w_gate[e]) * (et @ layer.w_up[e])
        out.index_add_(0, t_idx, ew * (h @ layer.w_down[e]))
    return out.view(B, S, Dm), aux * layer.aux_w


def test_grad_flow():
    layer = MoELayer(make_cfg())
    x = torch.randn(2, 5, D_MODEL, requires_grad=True)
    out, aux = layer(x)
    assert out.shape == x.shape, out.shape
    (out.sum() + aux).backward()
    for name, p in [("w_gate", layer.w_gate), ("w_up", layer.w_up), ("w_down", layer.w_down)]:
        assert p.grad is not None and p.grad.abs().sum() > 0, f"{name} got no gradient"
    assert x.grad is not None and x.grad.abs().sum() > 0, "input got no gradient"
    print("[PASS] 1. grad flow — w_gate/w_up/w_down/input grads all non-zero")


def test_equivalence():
    cf = float(E * 2)  # guarantees capacity >= K*T -> zero drops
    layer = MoELayer(make_cfg(capacity_factor=cf))
    x = torch.randn(2, 7, D_MODEL)
    ref_out, ref_aux = naive_reference(layer, x)
    got_out, got_aux = layer(x)
    diff = (got_out - ref_out).abs().max().item()
    assert torch.allclose(got_out, ref_out, atol=1e-4, rtol=1e-4), f"output diff {diff}"
    assert torch.allclose(got_aux, ref_aux, atol=1e-5), f"aux diff {(got_aux-ref_aux).abs().item()}"
    print(f"[PASS] 2. equivalence to naive per-expert loop — max|Δ|={diff:.2e}")


def test_checkpoint():
    cf = float(E * 2)
    layer = MoELayer(make_cfg(capacity_factor=cf))
    x = torch.randn(2, 6, D_MODEL)

    layer.zero_grad()
    (layer(x)[0]).sum().backward()
    g_plain = layer.w_down.grad.clone()

    layer.zero_grad()
    out_ck = ckpt.checkpoint(lambda inp: layer(inp)[0], x, use_reentrant=False)
    out_ck.sum().backward()
    g_ck = layer.w_down.grad.clone()

    diff = (g_plain - g_ck).abs().max().item()
    assert torch.allclose(g_plain, g_ck, atol=1e-4, rtol=1e-4), f"grad diff {diff}"
    print(f"[PASS] 3. checkpoint determinism — w_down grad max|Δ|={diff:.2e}")


def test_edge_small():
    layer = MoELayer(make_cfg(capacity_factor=1.25))
    x = torch.randn(1, 1, D_MODEL)  # T = 1 < E = 8
    out, aux = layer(x)
    assert out.shape == x.shape and torch.isfinite(out).all() and torch.isfinite(aux).all()
    print("[PASS] 4. edge T=1<E — runs, output finite")


def test_persona():
    layer = MoELayer(make_cfg(capacity_factor=float(E * 2)))
    x = torch.randn(2, 5, D_MODEL, requires_grad=True)
    pctx = torch.randn(2, 16)  # composite_persona_dim
    out, aux = layer(x, persona_ctx=pctx)
    (out.sum() + aux).backward()
    gw = layer.router.gate.weight.grad
    assert gw is not None and gw.abs().sum() > 0, "router gate got no gradient"
    print("[PASS] 5. persona path — router gate receives gradient")


def test_bf16_mixed_precision():
    """Reproduces the real GPU scenario: activations bf16 but router gate forced fp32."""
    layer = MoELayer(make_cfg(capacity_factor=1.25)).to(torch.bfloat16)
    assert layer.router.gate.weight.dtype == torch.float32, layer.router.gate.weight.dtype
    assert layer.w_gate.dtype == torch.bfloat16, layer.w_gate.dtype
    x = torch.randn(2, 6, D_MODEL, dtype=torch.bfloat16)
    pctx = torch.randn(2, 16, dtype=torch.bfloat16)
    out, aux = layer(x, persona_ctx=pctx)            # must not raise a dtype error
    assert out.dtype == torch.bfloat16 and torch.isfinite(out.float()).all()
    print("[PASS] 6. bf16 activations + fp32 gate — no dtype crash, output finite")


if __name__ == "__main__":
    test_grad_flow()
    test_equivalence()
    test_checkpoint()
    test_edge_small()
    test_persona()
    test_bf16_mixed_precision()
    print("\n✅ All batched-MoE correctness tests passed.")
