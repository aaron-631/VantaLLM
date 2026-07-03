# VantaLLM Technical Report

**Author:** Aaron Chakraborty  

## 1. Executive Summary

VantaLLM is a modular framework for training decoder-only transformer language models from scratch, designed for experimentation, research, and scalable distributed pretraining. This report details the architectural components, distributed training mechanics, and scaling methodologies implemented within the VantaLLM framework.

## 2. Design Philosophy

VantaLLM is built upon three core tenets:
1. **Clarity over Complexity:** The codebase eschews heavy layers of abstraction, maintaining a clean PyTorch implementation that is easy to read, debug, and modify.
2. **Scalability:** The architecture naturally scales from minimal single-GPU debugging setups to multi-node DeepSpeed environments via a centralized JSON configuration system.
3. **Reproducibility:** Training runs, hardware validations, and data processing pipelines are fully deterministic and encapsulated.

## 3. Architecture

VantaLLM implements a modern causal decoder-only Transformer. Key architectural deviations from the standard vanilla Transformer include:

- **Positional Encoding:** We utilize Rotary Position Embeddings (RoPE), injecting relative positional information directly into the attention mechanisms rather than using absolute token embeddings.
- **Activation Functions:** SwiGLU replaces the standard ReLU/GELU in the Feed-Forward Networks (FFN), offering better gradient flow and empirical performance.
- **Normalization:** Pre-norm formulation using Root Mean Square Normalization (RMSNorm) is applied to stabilize deep network training.
- **Attention Mechanism:** Grouped-Query Attention (GQA) reduces KV-cache pressure while preserving multi-head expressivity. The codebase also includes hooks for integrating FlashAttention in future high-throughput paths.
- **Feedforward Layout:** The shipped transformer alternates dense blocks with sparse Mixture-of-Experts (MoE) blocks using top-k routing rather than treating MoE as an external add-on.
- **Conditioning Hooks:** Persona-conditioning vectors are carried through the architecture so the same core stack can support controlled training and inference flows.
- **Inference Efficiency:** KV-cache compression and quantized cache handling are built into the inference path for lower-memory autoregressive decoding.
- **TurboQuant Contract:** The repository includes an INT8 KV-cache compression path with reconstruction checks in the built-in system tests and module self-test path.
- **MoE Safety Scaffolding:** `capacity_factor` enables optional bounded expert capacity; overflow handling retains the highest-weight assignments deterministically instead of silently dropping tokens.

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
