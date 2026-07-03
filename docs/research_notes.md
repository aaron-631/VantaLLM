# Research Notes

This document records architectural observations, scaling methodologies, and empirical findings during the development of VantaLLM.

## Validated Implementation
The 567M parameter configuration (`vanta_567m.json`) serves as the primary validated implementation. Execution runs on NVIDIA A100 GPUs confirm stable convergence and effective mixed-precision (BF16) utilization. 

Empirical observations:
- Rotary Position Embeddings (RoPE) provided noticeable stability improvements over absolute encodings for long context lengths.
- SwiGLU activation networks required careful scaling of learning rates to prevent early divergence.

## Analytical Scaling Methodology

To prepare the framework for future supercomputing deployments, analytical profiles for 125B and 235B parameters have been established.
- **125B Configuration (`vanta_125b.json`)**: Configured with extreme depth and wide embedding dimensions.
- **235B Configuration (`vanta_235b.json`)**: Employs grouped-query attention (GQA) conceptually to bound KV-cache sizes at this scale.

*Note: These scaling configurations are purely analytical targets. They have not been experimentally trained to convergence.*

## Tokenizer Optimization
The custom BPE tokenizer pipeline was observed to be highly efficient when coupled with flat binary sharding (`.bin` files). By completely bypassing JSON/text parsing during the PyTorch `DataLoader` iterations, data loading bottlenecks were eliminated, ensuring GPUs remain fully saturated.
