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
- **Attention Mechanism:** Multi-Head Attention (MHA) forms the core, with hooks available for integrating FlashAttention to bound memory complexity from $O(N^2)$ to $O(N)$ for long sequence lengths.

## 4. Tokenizer and Data Pipeline

VantaLLM employs a custom Byte-Pair Encoding (BPE) tokenizer pipeline. 
- **Vocabulary:** Supports arbitrary vocabulary sizes defined during the tokenization stage. 
- **Sharding Strategy:** Datasets (such as example cybersecurity corpora) are processed, normalized, tokenized, and aggressively chunked into flat binary `.bin` shards. This enables the PyTorch `DataLoader` to stream gigabytes of training data efficiently with a minimal RAM footprint, completely eliminating dataset-loading bottlenecks.

## 5. Optimizer & Checkpoint Format

- **Optimizer:** The framework defaults to `AdamW`, utilizing weight decay masking for layer norms and biases. Learning rate schedules use a cosine decay with linear warmup.
- **Checkpoints:** Model states, optimizer momentum, and iteration counts are serialized into standard PyTorch `.pt` files. Checkpoint saving is non-blocking and atomic to prevent state corruption during distributed runs.

## 6. Distributed Training Strategy

The framework natively targets high-end compute topologies:
- **DDP (Distributed Data Parallel):** Employed for multi-GPU training on a single node. 
- **DeepSpeed ZeRO:** Integrated via `launch_deepspeed.sh`. ZeRO stages 1 and 2 partition optimizer states and gradients across ranks.

## 7. Scaling Methodology and Parameter Rationale

Scaling configurations are explicitly tracked via JSON files in the `configs/` directory. 

### 7.1 The 567M Reference Implementation (`vanta_567m.json`)
The 567M model is the core validated reference implementation of VantaLLM. 
- **Hardware Target:** Designed to train on NVIDIA A100 GPUs (AI Koch cluster).
- **Memory Management:** To fit the 567M parameter states, optimizer momentum, and KV-caches onto an A100, we heavily utilize BF16 precision.
- **Batching Strategy:** Micro-batch sizes are kept artificially small. True batch size is achieved via gradient accumulation, ensuring convergence stability without relying on massive VRAM pools. 
- **Status:** Fully validated execution environment and memory allocation compatibility.

### 7.2 Analytical Scaling Configurations
To prove the framework's mathematical readiness for hyper-scale scenarios, two theoretical scaling profiles are included:
- **125B Configuration (`vanta_125b.json`)**
- **235B Configuration (`vanta_235b.json`)**

**Important Disclaimer:** The 125B and 235B configurations are analytical projections. They define the dimensional hyperparameters (layers, heads, embedding dimensions) required for these models but **have not been experimentally validated or trained to convergence**. They exist strictly as templates for researchers looking to deploy VantaLLM on supercomputing clusters.
