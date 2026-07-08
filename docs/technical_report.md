# VantaLLM Technical Report

**Author:** Aaron Chakraborty  

## 1. Executive Summary

VantaLLM is a modular framework for training decoder-only transformer language models from scratch, designed for experimentation, research, and scalable distributed pretraining. This report details the architectural components, distributed training mechanics, and scaling methodologies implemented within the VantaLLM framework.

## 2. Design Philosophy

VantaLLM is built upon three core tenets:
1. **Clarity over Complexity:** The codebase eschews heavy layers of abstraction, maintaining a clean PyTorch implementation that is easy to read, debug, and modify.
2. **Scalability:** The architecture naturally scales from minimal single-GPU debugging setups to multi-node DeepSpeed environments via a centralized JSON configuration system.
3. **Reproducibility:** Training runs, hardware validations, and data processing pipelines are fully deterministic and encapsulated.

## 2.1 Key Engineering Decisions

Several implementation choices are worth calling out because they reflect systems tradeoffs rather than generic "best practices."

- **Offline tokenization and binary sharding:** VantaLLM does not rely on on-the-fly text parsing inside the hot training loop. Raw text is normalized, deduplicated, tokenized, packed, and saved into flat binary shards so GPU training is not gated by JSON parsing or tokenizer throughput.
- **JSON-based architecture switching:** Model scale is expressed through config files rather than separate model branches. This keeps the transformer implementation stable while letting operators switch between small validation paths and larger analytical profiles.
- **Shared prompt construction:** Prompt formatting is centralized because prompt drift between training, tests, and inference is a real source of silent regressions, especially once retrieved context and chat markers are introduced.
- **Validation before orchestration escalation:** The repository is intentionally structured so that distributed backend checks, data contract checks, and tiny training runs happen before DDP or DeepSpeed ZeRO launches.

## 3. Architecture

VantaLLM implements a modern causal decoder-only Transformer. Key architectural deviations from the standard vanilla Transformer include:

- **Positional Encoding:** We utilize Rotary Position Embeddings (RoPE), injecting relative positional information directly into the attention mechanisms rather than using absolute token embeddings. A single RoPE instance is shared across all transformer blocks, and its cos/sin cache is grown lazily on demand rather than materialized at the full `max_seq_len` per layer — so the 128K capability ceiling costs no memory until sequences actually approach it.
- **Activation Functions:** SwiGLU replaces the standard ReLU/GELU in the Feed-Forward Networks (FFN), offering better gradient flow and empirical performance.
- **Normalization:** Pre-norm formulation using Root Mean Square Normalization (RMSNorm) is applied to stabilize deep network training.
- **Attention Mechanism:** Grouped-Query Attention (GQA) reduces KV-cache pressure while preserving multi-head expressivity. The codebase also includes hooks for integrating FlashAttention in future high-throughput paths.
- **Feedforward Layout:** The shipped transformer alternates dense blocks with sparse Mixture-of-Experts (MoE) blocks using top-k routing rather than treating MoE as an external add-on. The MoE layer stores its experts as three batched weight tensors and computes all experts with a grouped `torch.bmm` (see §3.1), instead of iterating expert submodules in Python.
- **Conditioning Hooks:** Persona-conditioning vectors are carried through the architecture so the same core stack can support controlled training and inference flows.
- **Inference Efficiency:** KV-cache compression and quantized cache handling are built into the inference path for lower-memory autoregressive decoding.
- **TurboQuant Contract:** The repository includes an INT8 KV-cache compression path with reconstruction checks in the built-in system tests and module self-test path.
- **MoE Safety Scaffolding:** `capacity_factor` bounds per-expert capacity so per-step memory is deterministic and cannot spike on router skew; overflow tokens are dropped to the block residual (Switch-Transformer style) rather than growing an unbounded buffer. The reference configs set `capacity_factor = 1.25`.

### 3.1 Distributed MoE at scale (batched grouped-GEMM)

The MoE feedforward is the one place where a naive implementation quietly breaks
large multi-GPU training, so it is worth describing precisely.

**The problem with the obvious implementation.** The direct way to write a
128-expert MoE is a Python loop: hold 128 separate expert submodules, and for each
expert gather its routed tokens and run them. This is correct and readable, but
under DeepSpeed ZeRO-3 with CPU parameter offload each expert submodule is a
separately-sharded parameter group, so touching it triggers its own gather from
CPU. With 128 experts across ~47 MoE layers that is thousands of small gathers per
step — a "parameter-fetching avalanche" in which the GPUs spend almost all their
time waiting on the host interconnect, and a large run appears to hang.

**The implementation used here.** Each MoE layer stores its experts as three
batched parameters — `w_gate (E, d_model, d_ff)`, `w_up (E, d_model, d_ff)`,
`w_down (E, d_ff, d_model)` — so a layer's experts are three large tensors, not
384 small ones. Routing is unchanged (the persona-aware top-k router is untouched).
Dispatch places each routed (token, slot) into a fixed-capacity buffer
`(E, capacity, d_model)` using a segmented prefix-count index so every slot lands
in a unique position with no scatter collision. All experts then run as a grouped
`torch.bmm` over the expert dimension, and the results are combined back with a
single weighted `index_add`. Parameter count is identical to the per-expert layout;
only the storage shape and the dispatch changed. Under ZeRO-3 this collapses the
per-layer gather count from ~384 to 3.

**Global load balancing.** The auxiliary load-balancing loss averages its
expert-usage histogram across data-parallel ranks with an `all_reduce` before the
loss is formed, so the router balances the global token distribution rather than
each GPU's local shard in isolation. This is a no-gradient statistic, so the
collective is autograd-free and symmetric across ranks.

The batched dispatch is verified on CPU against a reference per-expert loop
(bit-level agreement to ~1e-8 with unbounded capacity) and for gradient-checkpoint
determinism; see `tests/test_batched_moe.py`.

## 4. Tokenizer and Data Pipeline

VantaLLM employs a custom Byte-Pair Encoding (BPE) tokenizer pipeline. 
- **Vocabulary Contract:** The active tokenizer contract is `151003` tokens, including the persisted chat markers `<|user|>`, `<|assistant|>`, and `<|system|>`.
- **Lifecycle Terminology:** Human-readable `.jsonl` subsets are treated as **slices**; packed tensor `.bin` files are treated as **shards**.
- **Sharding Strategy:** Datasets are normalized, deduplicated, tokenized, and incrementally chunked into flat binary shards. This lets the `DataLoader` stream training data with low host-memory overhead.
- **Reproducibility Controls:** The append-only pipeline is tracked through `pipeline_manifest.json` and `global_hash_log.json` so repeated runs stay auditable and idempotent.

## 5. Optimizer & Checkpoint Format

- **Optimizer:** The current training reference path uses `bitsandbytes.optim.PagedAdamW8bit` to reduce optimizer memory pressure on constrained GPU setups. Learning rate schedules use cosine decay with linear warmup.
- **Checkpoints:** Model state, optimizer state, and step counters are serialized into standard PyTorch `.pt` files. Saving is rank-safe so only the correct process writes artifacts during distributed runs.

## 6. Distributed Training Strategy

The framework natively targets high-end compute topologies:
- **DDP (Distributed Data Parallel):** Employed for multi-GPU training on a single node through the packaged launch scripts.
- **DeepSpeed ZeRO:** Integrated via `launch_deepspeed.sh`, with ZeRO configurations for staged optimizer, gradient, and parameter partitioning.
- **Validation-First Execution:** `verify_distributed.py` supports both CPU/Gloo and GPU/NCCL validation so topology issues can be isolated before model bring-up.

## 6.1 Prompt Parity and Retrieval Contract

VantaLLM treats prompt construction as a shared systems contract rather than a UI detail.

- Training, evaluation, and inference are expected to use the same `build_prompt()` utility.
- The active prompt contract version is `v1_parity_hardened`.
- Retrieval context is inserted through the shared prompt builder so section ordering and boundaries remain deterministic.
- The repository includes dedicated RAG-oriented tests that measure both contextual generation behavior and direct probability amplification at the logits level.

## 6.2 Serving Interface

The FastAPI inference server exposes:

- `GET /health` for runtime readiness and queue/session visibility
- `POST /generate` for non-streaming generation
- `POST /generate/stream` for Server-Sent Event token streaming

Persona routing is implemented as an inference-time conditioning layer over the same model, not as separate checkpoints or branches.

## 6.3 Operational Hardening Lessons

The distributed path in VantaLLM reflects practical failure modes encountered during scaling work:

- **Memory pressure is treated as a first-class design constraint:** BF16, gradient checkpointing, ZeRO partitioning, and paged optimizer usage are part of the documented operator path rather than optional afterthoughts.
- **Collective symmetry matters in MoE systems:** sparse expert routing can deadlock distributed jobs if ranks diverge in collective participation, so the implementation is designed around rank-safe behavior rather than assuming balanced token flow.
- **Prompt and data contracts are guarded explicitly:** shard validation, tokenizer contract checks, LFS stub detection, and shared prompt assembly are all there to catch "looks fine but breaks later" failures early.

## 7. Scaling Methodology and Parameter Rationale

Scaling configurations are explicitly tracked via JSON files in the `configs/` directory. 

### 7.1 The 567M Reference Implementation (`vanta_567m.json`)
The 567M model is the core validated reference implementation of VantaLLM. 
- **Hardware Target:** Designed around NVIDIA A100-class GPUs as the primary public reference path.
- **Memory Management:** To fit the 567M parameter states, optimizer momentum, and KV-caches onto an A100, we heavily utilize BF16 precision.
- **Batching Strategy:** Micro-batch sizes are kept artificially small. True batch size is achieved via gradient accumulation, ensuring convergence stability without relying on massive VRAM pools. 
- **Status:** This is the public reference path for 20GB-class A100 bring-up and the baseline operators should validate first.

### 7.2 Analytical Scaling Configurations
To prove the framework's mathematical readiness for hyper-scale scenarios, two theoretical scaling profiles are included:
- **125B Configuration (`vanta_125b.json`)**
- **235B Configuration (`vanta_235b.json`)**

**Important Disclaimer:** The 125B and 235B configurations are analytical projections. They define the dimensional hyperparameters (layers, heads, embedding dimensions) required for these models but **have not been experimentally validated or trained to convergence**. They exist strictly as templates for researchers looking to deploy VantaLLM on supercomputing clusters.

## 8. Known Limitations & Architectural Trade-offs

These are deliberate boundaries of the current implementation, documented so that
they are read as engineering choices rather than oversights. None of them block
the 567M reference path; each is a scoped decision with a known upgrade route.

### 8.1 Persona conditioning: training and inference use different pathways

Persona conditioning is **not** wired identically across training and inference,
and this is a genuine (disclosed, not yet resolved) limitation:

- **Training** drives persona through a learnable stack —
  `persona_store` (`PersonaEmbeddingStore`) → `persona_router`
  (`ExpertOfExpertsCompositionRouter`) → `persona_to_hidden` — injected at the
  embedding layer. During pretraining the persona IDs are held constant, so this
  stack receives gradient but learns only a fixed persona.
- **Inference** (`server.py`) instead supplies an L2-normalized last-token hidden
  vector and projects it through `persona_vec_to_composite`. That projection is
  **randomly initialized and receives no gradient on the training path** (it is
  the sole unused parameter under DDP's `find_unused_parameters=True`).

The consequence is a train/serve representation mismatch: the persona signal the
model was optimized against is not the same signal it is conditioned on at serve
time. This does not affect base language-modeling quality, but persona steering
at inference is effectively untrained. Unifying the two — so the inference
projection is learned during training — requires a retraining run and is left as
future work rather than shipped as an unvalidated change.

### 8.2 Deliberate scope boundaries (not defects)

The following are intentional simplifications appropriate for a single-node,
from-scratch reference; each is labeled in-code:

- **MoE dispatch is a batched grouped-GEMM** with fixed-capacity token routing
  (see §3.1), not a per-expert Python loop. Every expert runs on every rank
  (ZeRO-3 data-parallel MoE, not expert-parallel), but experts are computed with a
  grouped `torch.bmm` over three batched weight tensors, which is what makes the
  128-expert layer viable under ZeRO-3 CPU offload. The remaining scale step —
  true expert-parallelism with all-to-all token routing so experts are sharded
  across GPUs rather than replicated — is deliberately **not** implemented here and
  is tracked as future work; it is a different tradeoff (throughput at many-GPU
  scale) beyond the single-node bring-up this report covers.
- **`Int8Linear` / `Int4Linear` are experimental scaffolds.** They dequantize to
  float for compute and do **not** yet deliver VRAM reduction; they exist to
  exercise the quantization interface, and are labeled as such in-code.
- **`TurboQuant` INT8 KV-cache** decompresses through FP32 before BF16 compute.
  Reconstruction error is within spec (MAE ≈ 0.004), but it is a lossy cache, not
  a lossless one.
- **The 128K `max_seq_len` is a capability ceiling, not a trained context length.**
  The validated bring-up sequence length is 2048; long-context behavior at the
  nominal maximum has not been trained or evaluated.
- **Test surface.** The pytest-native smoke suite is intentionally small (8 tests);
  the larger script-style suites are driver programs run directly (see
  `tests/conftest.py`), not pytest-collected.

### 8.3 Not validated at scale on this hardware

Two items are architecturally in place but only provable on a real multi-GPU run,
and are therefore **not** claimed as validated: the bitsandbytes `PagedAdamW8bit`
optimizer under ZeRO-3 CPU offload, and the embedding/LM-head weight tie
constructed inside `deepspeed.zero.Init()` surviving ZeRO-3 parameter
partitioning. Both are exercised by the 567M path's configuration but require a
GPU session to confirm end-to-end.
