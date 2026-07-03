# VantaLLM

A modular framework for training decoder-only transformer language models from scratch, designed for experimentation, research, and scalable distributed pretraining.

The model architecture, training pipeline, and configurations were designed and implemented by Aaron Chakraborty.

> **Note:** Trained model weights are not included in this repository due to size constraints. This repository provides the complete infrastructure, training code, and architectural configurations required to train the model from scratch.

## Overview

VantaLLM is an open-source research repository built for exploring large-scale LLM training dynamics. It provides a robust, dependency-light PyTorch foundation for modern LLM pretraining, featuring a highly configurable Transformer architecture, a custom tokenizer pipeline, and native support for distributed training (DDP and DeepSpeed). 

While the framework is dataset-agnostic, the data pipeline is bundled with tools that can process domain-specific texts (such as cybersecurity corpora) as example workloads for full-stack validation.

## Features

- **Modular Architecture**: A clean, hackable decoder-only transformer built in PyTorch.
- **Distributed Training**: First-class support for Multi-GPU training via PyTorch DDP and DeepSpeed ZeRO.
- **Dynamic Configuration**: JSON-based model configurations allowing for rapid scaling from tiny debug models to multi-billion parameter architectures.
- **Robust Data Pipeline**: Sharded data processing, fast tokenization, and seamless integration with HuggingFace datasets.
- **Inference Server**: Built-in FastAPI server for querying checkpoints locally.

## Architecture

VantaLLM implements a modern causal language model featuring:
- Rotary Position Embeddings (RoPE)
- SwiGLU Feed-Forward Networks
- RMSNorm for training stability
- Mixed-precision (BF16/FP16) support for hardware acceleration
- FlashAttention integration (where supported)

## Installation

```bash
git clone https://github.com/aaronchakraborty/VantaLLM.git
cd VantaLLM
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

*(Note: For GPU acceleration, ensure you have the correct PyTorch CUDA wheels installed for your system's CUDA version.)*

## Quick Start

### 1. Prepare the Environment
Copy the example environment file and configure it:
```bash
cp .env.example .env
```

### 2. Process Data
Run the standard pipeline to normalize, tokenize, and shard your dataset:
```bash
./run_pipeline.sh
```

### 3. Train a Tiny Model (Smoke Test)
Validate your environment by training a small configuration:
```bash
TINY_MODEL=1 QUANT_MODE=bf16 python3 vantallm/training/train.py
```

## Repository Layout

```
VantaLLM/
├── README.md
├── LICENSE
├── CONTRIBUTING.md
├── CODE_OF_CONDUCT.md
├── SECURITY.md
├── CHANGELOG.md
├── ROADMAP.md
├── CITATION.cff
├── .gitignore
├── .env.example
├── requirements.txt
├── demo_inference.py
├── verify_gpu_ready.py
├── verify_distributed.py
├── configs/            # JSON architectural scaling configurations
├── data_pipeline/      # Dataset ingestion, tokenization, and sharding tools
├── docs/               # Technical reports and extensive documentation
├── scripts/            # Utility and hardware validation scripts
├── vantallm/           # Core framework package
│   ├── distributed/
│   ├── inference/
│   ├── model/
│   ├── tokenizer/
│   └── training/
├── tests/              # Unit and integration tests
├── eval/               # Evaluation prompts and scripts
└── checkpoints/        # (gitignored) Local model weights storage
```

## Configuration System & Scaling

VantaLLM handles model hyperparameters via a lightweight JSON configuration system.

### The 567M Reference Configuration (`vanta_567m.json`)
The 567M parameter configuration acts as the heavily validated reference implementation. It is specifically designed and optimized to train on **NVIDIA A100 GPUs (AI Koch cluster)**. 

To maximize throughput and prevent Out-Of-Memory (OOM) errors on the A100's VRAM, the training loop relies on:
- **Batch Sizing & Gradient Accumulation:** Micro-batches are kept small to fit the KV-cache and optimizer states in memory. We heavily utilize gradient accumulation steps to simulate a globally massive batch size for stable convergence.
- **Precision:** BF16 (Bfloat16) mixed precision is enforced to utilize the A100 Ampere Tensor Cores effectively, cutting memory consumption in half with no loss in model accuracy.

### Analytical Scaling Configurations
The repository also includes scaling profiles for future expansion:
- `vanta_125b.json` (125 Billion Parameters)
- `vanta_235b.json` (235 Billion Parameters)

*Please note: The 125B and 235B configurations are purely analytical and theoretical scaling references. They have not been experimentally validated or trained to convergence in this repository.*

## Distributed Training

VantaLLM supports multi-node and multi-GPU training out of the box.

For DDP (Distributed Data Parallel):
```bash
bash vantallm/distributed/launch_ddp.sh
```

For DeepSpeed ZeRO:
```bash
bash vantallm/distributed/launch_deepspeed.sh
```

## Inference

To interact with a trained checkpoint, use the built-in FastAPI server:
```bash
python3 vantallm/inference/server.py
```
You can then submit generation requests via standard HTTP POST calls.

## Roadmap

- [ ] Implement FlashAttention-2 natively into the Transformer core.
- [ ] Add MoE (Mixture of Experts) architectural support.
- [ ] Expand the analytical multi-node DeepSpeed ZeRO-3 configurations.
- [ ] Release pre-trained base model weights.

## Acknowledgements

- Inspiration drawn from `nanoGPT`, `Megatron-LM`, and `DeepSpeed`.
- Specialized datasets used for validation were sourced from the open-source community, including various cybersecurity corpora.

## Citation

If you use VantaLLM in your research or project, please cite it as:
```bibtex
@software{VantaLLM2026,
  author = {Chakraborty, Aaron},
  title = {VantaLLM: A modular framework for training decoder-only transformer language models},
  year = {2026},
  publisher = {GitHub},
  url = {https://github.com/aaronchakraborty/VantaLLM}
}
```

## Contributing

We welcome community contributions. Please review the `CONTRIBUTING.md` (coming soon) before submitting PRs. Make sure to run all unit tests and verify your code formatting.
