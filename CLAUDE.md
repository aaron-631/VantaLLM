# Interview Preparation: VantaLLM

This document is for personal interview preparation and context anchoring.

## Technical Summary
VantaLLM is a custom, from-scratch decoder-only transformer framework written in PyTorch. It is designed to be a highly modular, dependency-light infrastructure for large-scale language model pre-training. It features native support for PyTorch Distributed Data Parallel (DDP) and DeepSpeed ZeRO partitioning, a highly configurable JSON-driven architectural scaling system, and a custom BPE tokenizer tightly integrated with a flat-binary dataset sharding pipeline. The primary validated implementation is a 567M parameter model specifically optimized to run in mixed-precision (BF16) on NVIDIA A100 GPUs.

## Key Architectural Decisions & Rationale
1. **Flat Binary Sharding for Data Loading:** Instead of streaming raw JSONL or relying on complex HuggingFace dataset iterators during training, the pipeline aggressively tokenizes and packs sequences into flat `.bin` tensors. **Why:** It completely eliminates CPU and disk I/O bottlenecks during PyTorch `DataLoader` iterations, ensuring the GPUs are constantly saturated with data.
2. **Dynamic JSON Configuration:** Architectural hyperparameters (layers, heads, dimensions) are decoupled from the Python code into JSON files (`vanta_567m.json`, `vanta_125b.json`). **Why:** It enables rapid theoretical scaling and mathematical validation of massive models (e.g. 125B, 235B) without touching the core runtime logic.
3. **SwiGLU & RoPE over GELU & Absolute Embeddings:** **Why:** Following modern best practices (e.g., LLaMA), Rotary Position Embeddings provide better extrapolation to long contexts, and SwiGLU empirically offers better gradient flow than standard feed-forward networks, despite the slight increase in parameter complexity.

## The Hardest Technical Problem Solved
**Memory Fragmentation and Distributed State Deadlocks:**
When initially scaling the training loop from a single GPU to multi-GPU DeepSpeed ZeRO topologies, the training loop would silently hang or crash with Out-Of-Memory (OOM) errors. The root cause was a combination of PyTorch's memory allocator becoming heavily fragmented during validation evaluations and asynchronous collective communication deadlocks when different GPUs processed varying sequence lengths. 
**The Fix:** I enforced strict, uniform sequence packing (2048 length) at the sharding level before training, decoupling padding logic from the runtime. I also integrated `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` and utilized ZeRO-1/ZeRO-2 to partition optimizer momentum states, which effectively halved the VRAM footprint and stabilized multi-node synchronization.

## What I Would Do Differently With More Resources
With unlimited compute and time, I would:
1. **Natively implement FlashAttention-2 in CUDA/Triton:** Currently, the attention mechanism relies on standard PyTorch scaled dot-product attention. A custom kernel would allow me to push the context length from 2048 up to 32k or 128k without blowing up the VRAM.
2. **Train the 125B model to convergence:** The 125B and 235B profiles are currently analytical configurations. With a multi-node cluster, I would love to physically train the 125B model over several trillion tokens to empirically study emergence and routing skew in Mixture of Experts (MoE) expansions.

## The Most Surprising Result / Failure
**Surprising Result (Tokenizer Efficiency):** I was surprised by how much of the training time in naive setups is actually spent just waiting on the CPU to parse text and tokenize. Once I moved tokenization entirely offline and streamed raw `.bin` tensors directly to the GPU, the throughput (tokens/sec) almost doubled on the exact same hardware.
**Surprising Failure (Loss Spikes with FP16):** Early attempts to use standard FP16 mixed precision resulted in catastrophic loss spikes and NaN gradients deep into the training run. I learned the hard way that the dynamic range of FP16 is too narrow for deep transformer gradients. Switching to BF16 (Bfloat16), which perfectly matches the exponent range of FP32, completely solved the stability issues without any code changes, proving the immense value of Ampere's Tensor Cores.
