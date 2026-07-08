# VantaLLM

[![ci](https://github.com/aaron-631/VantaLLM/actions/workflows/ci.yml/badge.svg)](https://github.com/aaron-631/VantaLLM/actions/workflows/ci.yml)

VantaLLM is a from-scratch PyTorch framework for training decoder-only language models, with a modular transformer core, a custom tokenizer, an incremental sharded data pipeline, and first-class distributed training support.

The model architecture, training pipeline, and configurations were designed and implemented by Aaron Chakraborty.

> **Note:** Trained model weights are not included in this repository due to size constraints. This repository ships the full training, inference, tokenizer, and distributed-systems stack required to reproduce runs from source. If you cloned the repo without hydrating Git-LFS, checkpoint-backed tests and resume flows will be present in code but the actual `.pt` files will be missing until you run `git lfs pull`.

## At A Glance

A from-scratch PyTorch framework for training decoder-only language models, built and trained end-to-end rather than assembled from high-level libraries.

- **Modern architecture, implemented from scratch:** RoPE, GQA, RMSNorm, SwiGLU, and alternating dense/MoE feedforward blocks with top-k expert routing over a batched grouped-GEMM expert layer.
- **Real distributed training:** PyTorch DDP and DeepSpeed ZeRO, rank-safe checkpoint and resume, free-port autodetection, and validation-first cluster bring-up.
- **Complete data pipeline:** offline normalization, deduplication, tokenization, and binary sharding, with manifest-backed reproducibility.
- **Tested and CI-backed:** architecture demo, distributed validator, GPU-readiness checks, and a pytest suite.
- **Concrete result:** completed a 30,000-step training run of the 567M-parameter config on a single 20GB A100. The same codepaths scale to the 125B and 235B configs through JSON configuration alone, with a dedicated `vanta_125b_bringup.json` profile and a locked 4-GPU launcher for multi-GPU ZeRO-3 validation.
- **Hardware-portable by design:** one codebase, no branches. A 20GB A100 runs the 567M reference under plain DDP with **no ZeRO partitioning**; a 4× RTX PRO 6000 Blackwell node runs the full 125B under ZeRO-3 CPU-offload; an H200 (or any box without `bitsandbytes`) is handled by the optimizer's automatic `PagedAdamW8bit → AdamW` fallback. You change a config and a launcher flag, not the model code.

Quick start:

```bash
pip install -e .
python3 -m vantallm.model.transformer   # architecture sanity check
python3 verify_distributed.py            # distributed bring-up check
```

## What This Project Demonstrates

VantaLLM is intended to read like a serious end-to-end LLM systems project, not just a model definition. The repository demonstrates:

- **Core model engineering:** a custom decoder-only transformer with RoPE, GQA, RMSNorm, SwiGLU, and alternating dense/MoE feedforward blocks.
- **Training systems work:** explicit distributed launch paths, DeepSpeed ZeRO integration, checkpoint-safe training, and hardware-aware validation flow.
- **Data pipeline discipline:** append-only normalization, deduplication, tokenization, sharding, and manifest-backed reproducibility.
- **Inference and verification:** local demos, distributed bring-up checks, GPU readiness checks, and test coverage around the active tokenizer/model contract.
- **Scalable design thinking:** the same codepaths support tiny local debugging, validated A100 20GB reference runs, and larger multi-GPU deployments through config changes rather than branch-specific logic.

## Why These Choices

Several design decisions in VantaLLM are deliberate engineering tradeoffs rather than aesthetic preferences:

- **Packed shard training instead of runtime text parsing:** tokenization and packing happen offline so the training loop can stream flat binary tensors with minimal CPU and disk overhead.
- **JSON-driven model configs instead of hardcoded runtime sizes:** architecture scale is decoupled from the core code so the same stack can move from tiny validation runs to larger analytical configurations without code forks.
- **One prompt builder for training, evaluation, and serving:** prompt parity is treated as a systems invariant because newline drift, double-wrapping, or mismatched chat markers quietly break RAG and checkpoint behavior.
- **Validation-first distributed bring-up:** topology checks, shard checks, and smoke training are staged before expensive ZeRO or multi-rank runs to reduce time lost to avoidable cluster failures.

## Two Lasting System Rules

These two rules are intended to stay true even as configs, hardware targets, and training scale change:

- **Streaming Data Law:** `read line -> tokenize -> append -> flush shard`. The data pipeline is designed around incremental processing so corpus size can grow from a few gigabytes to multi-terabyte-scale input without requiring the entire dataset to live in RAM at once.
- **Capability Layering Path:** `Base Model -> SFT -> Early Deploy -> RAG Integration -> RL Alignment -> Agent Integration`. The repo is structured around staged capability layering: first get the base model and serving path stable, then add retrieval, alignment, and higher-level tool use on top of a working core.

## Overview

VantaLLM is built as an end-to-end LLM engineering project rather than a model file in isolation. The repository covers:

- **Transformer architecture from scratch** in PyTorch.
- **Custom tokenizer + prompt contract** with stable chat markers.
- **Incremental data pipeline** for normalization, tokenization, and binary sharding.
- **Distributed training paths** for PyTorch DDP and DeepSpeed ZeRO.
- **Inference and validation tooling** for local debugging, audits, and system bring-up.

The codebase is GPU-agnostic by design: it runs on small local setups for structural testing, on A100-class hardware for the validated 567M reference path, and can scale forward to larger multi-GPU environments through the same codepaths and JSON configuration system.

## Read First

- [docs/technical_report.md](docs/technical_report.md): architecture, scaling rationale, and distributed design.
- [docs/gpu_cluster_guide.md](docs/gpu_cluster_guide.md): A100 reference guidance, RTX PRO 6000 Blackwell notes, DeepSpeed behavior, and high-capacity compute notes.
- [docs/inference_rag_guide.md](docs/inference_rag_guide.md): prompt parity, RAG routes, serving API, and persona routing.
- [docs/operational_handoff.md](docs/operational_handoff.md): live continuation path, checkpoint discipline, and operator sequence.
- [data_pipeline/DATASET_SPEC.md](data_pipeline/DATASET_SPEC.md): slice/shard contract and dataset manufacturing assumptions.
- [docs/future_scope.md](docs/future_scope.md): the two-track path from the current 567M framework to a published, domain-adapted cybersecurity model — resourcing strategy and engineering methodology.
- [CONTRIBUTING.md](CONTRIBUTING.md): contribution workflow.

The documentation set is intentionally small. Each file is supposed to answer a different operational question rather than restate the same material in different wording.

## Architecture At A Glance

VantaLLM implements a modern causal decoder-only stack featuring:

- **Rotary Position Embeddings (RoPE)** for long-context positional encoding, with a single shared, lazily-grown cache across all layers.
- **Grouped-Query Attention (GQA)** with quantized KV-cache support.
- **RMSNorm + SwiGLU** in a pre-norm transformer formulation.
- **Alternating dense and MoE blocks** with top-k expert routing, implemented as a **batched grouped-GEMM** over experts (not a per-expert Python loop) with fixed-capacity dispatch and global load-balancing — designed to scale under DeepSpeed ZeRO-3 without a parameter-fetch stall.
- **Persona-conditioning hooks** for both training-time and inference-time control paths.
- **JSON-driven scaling** from tiny debug models to large analytical reference configs, including a 125B bring-up profile for multi-GPU validation.

## Current Reference Contract

The current public reference path is:

- **Model config:** `configs/vanta_567m.json`
- **Tokenizer vocab:** `151003`
- **Persisted chat tokens:** `<|user|>=151000`, `<|assistant|>=151001`, `<|system|>=151002`
- **Pipeline entrypoint:** `./run_pipeline.sh`
- **Distributed launchers:** `vantallm/distributed/launch_ddp.sh` and `vantallm/distributed/launch_deepspeed.sh`

## Installation

```bash
git clone https://github.com/aaron-631/VantaLLM.git
cd VantaLLM
python3 -m venv .venv
source .venv/bin/activate
# Install the correct PyTorch wheel FIRST (see "Recommended installs" below) —
# deepspeed builds against an already-present torch.
pip install -r requirements.txt
```

The project is also packaged (`pyproject.toml`); to get the importable
`vantallm` package on your path, install it editable after torch:

```bash
pip install -e .            # add [dev] for pytest, [data] for the datasets pipeline dep
```

If you need tracked checkpoints rather than a source-only clone, hydrate Git-LFS objects explicitly:

```bash
git lfs install
git lfs pull
```

Notes:

- Install the correct PyTorch wheel separately for your CUDA runtime if you are targeting GPU execution.
- If `bitsandbytes` is installed, the trainer will use `bitsandbytes.optim.PagedAdamW8bit`; otherwise it falls back to `torch.optim.AdamW`.
- Copy `.env.example` to `.env` only if you want a local environment file for your own workflow.
- Do not use GitHub's "Download ZIP" flow if you expect checkpoint files; it can leave Git-LFS pointers unhydrated.
- If you plan to move checkpoints between machines, preserve the matching tokenizer assets under `vantallm/tokenizer/cybersec_tokenizer/` and the exact JSON model config used to create the weights.

Recommended installs:

```bash
# A100 / CUDA 12.1 reference path
pip install torch==2.11.0+cu121 --index-url https://download.pytorch.org/whl/cu121

# CPU-only local smoke testing
pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cpu
```

## Quick Start

### 1. Architecture Sanity Check

Run the built-in architecture demo to validate the transformer core, KV-cache compression path, and PPO scaffold:

```bash
python3 -m vantallm.model.transformer
```

### 2. Distributed Bring-Up Check

Validate device visibility, backend selection, and all-reduce behavior before launching heavy jobs:

```bash
python3 verify_distributed.py
torchrun --nproc_per_node=2 verify_distributed.py
```

### 3. Data Pipeline

Run the supported end-to-end pipeline entrypoint:

```bash
./run_pipeline.sh
python3 verify_gpu_ready.py
```

Drop general corpus files into `data_pipeline/raw/` and domain-specific pretraining files into `data_pipeline/raw/pretrain/`, then re-run the same pipeline entrypoint.

Important pipeline terminology:

- A **slice** is a human-readable `.jsonl` subset used for inspection, debugging, and early smoke tests.
- A **shard** is a packed binary `.bin` tensor file used for sustained training throughput.
- `data_pipeline/pipeline_manifest.json` tracks file lifecycle state, and `data_pipeline/global_hash_log.json` protects cross-run deduplication.

### 4. Local Inference Demo

Run a minimal generation demo with either random weights or a supplied checkpoint:

```bash
python3 demo_inference.py --prompt "Explain rotary position embeddings in one paragraph."
```

### 5. Training Entry Points

For a repo-native showcase smoke test:

```bash
export TRAIN_STEPS=10
export TINY_MODEL=1
export QUANT_MODE=bf16
export DATA_MODE=showcase
export DATA_PATH=data_pipeline/rag_synthetic_slice_v2_fixed.jsonl
python3 vantallm/training/train.py
```

For the 567M shard-backed reference path:

```bash
export MODEL_CONFIG=configs/vanta_567m.json
export TRAIN_STEPS=1000
export DATA_MODE=shard
export QUANT_MODE=bf16
bash vantallm/distributed/launch_ddp.sh 1
```

## Proof of Life

Everything below is real output produced on a CPU-only machine (no GPU) with `TINY_MODEL=1`,
except the parameter count, which uses the full `configs/vanta_567m.json`. It is reproducible from
a clean clone via the `ci` workflow.

**Parameter counts are exact, not estimated:**

```bash
$ python -c "import json; from vantallm.model.transformer import VantaTransformer, LLMConfig; \
  print(VantaTransformer(LLMConfig(**json.load(open('configs/vanta_567m.json')))).parameter_count()['TOTAL'])"
567068416          # the "567M" config is exact to the digit
```

**End-to-end train → generate on random tensors (pipeline mechanics):**

```
[step 0] loss=11.99  grad_norm=4.49
[step 5] loss=11.95  grad_norm=1.88     # loss ≈ ln(151003): correct uniform-prior entropy
[gen  ] prompt_len=8 -> output_len=20   # greedy decode with TurboQuant KV cache
```

**Test suite (CPU):**

| Suite | Result |
|---|---|
| `pytest tests/` (8 smoke + 3 suite wrappers) | 11/11 ✅ |
| `tests/test_task3.py` (structural) | 4/4 ✅ |
| `tests/test_system.py --phase 1` (unit) | 41/41 ✅ |
| `tests/test_system.py --phase 2` (integration) | 20/20 ✅ |
| `tests/test_system.py --phase 3` (live FastAPI server + SSE + sessions) | 25/25 ✅ |

KV-cache decode is bit-identical to no-cache decode; TurboQuant INT8 KV reconstruction MAE ≈ 0.004;
the BPE tokenizer round-trips code, SQLi, and CVE strings exactly with `vocab_size == 151,003`.

## Validation Path

For the cleanest bring-up sequence, follow this order:

1. `python3 -m vantallm.model.transformer`
2. `python3 verify_distributed.py`
3. `python3 verify_gpu_ready.py`
4. Tiny showcase training with explicit `TRAIN_STEPS`
5. `bash vantallm/distributed/launch_ddp.sh <gpus>`
6. `bash vantallm/distributed/launch_deepspeed.sh <gpus>`

This ordering matters. Distributed topology should be validated before expensive model execution, and shard integrity should be validated before shard-backed training.

## GPU Validation Matrix

| Stage | Command | Expected Outcome | If It Fails |
|---|---|---|---|
| CUDA visibility | `nvidia-smi` | Target GPU is visible with expected VRAM | Driver, runtime, or device issue |
| Torch CUDA check | `python3 -c "import torch; print(torch.cuda.device_count())"` | Correct GPU count | PyTorch/CUDA wheel mismatch |
| Distributed sanity | `python3 verify_distributed.py` | Single-node validation completes | Base environment issue |
| Multi-rank backend | `torchrun --nproc_per_node=<gpus> verify_distributed.py` | All-reduce succeeds | NCCL, topology, or rank wiring issue |
| Architecture sanity | `python3 -m vantallm.model.transformer` | Built-in forward/inference checks pass | Core architecture regression |
| Shard readiness | `python3 verify_gpu_ready.py` | Shards load and sample cleanly | Pipeline or shard integrity issue |
| Tiny smoke train | `TRAIN_STEPS=10 TINY_MODEL=1 DATA_MODE=showcase DATA_PATH=data_pipeline/rag_synthetic_slice_v2_fixed.jsonl python3 vantallm/training/train.py` | Training loop, optimizer, and checkpoint path work | Runtime or data contract issue |
| 567M single-GPU | `MODEL_CONFIG=configs/vanta_567m.json TRAIN_STEPS=1000 DATA_MODE=shard bash vantallm/distributed/launch_ddp.sh 1` | A100-class reference run starts cleanly | Memory or config issue |
| DDP multi-GPU | `MODEL_CONFIG=configs/vanta_567m.json TRAIN_STEPS=1000 DATA_MODE=shard bash vantallm/distributed/launch_ddp.sh <gpus>` | Rank-synchronised progress | DDP orchestration issue |
| DeepSpeed ZeRO | `MODEL_CONFIG=configs/vanta_567m.json TRAIN_STEPS=1000 DATA_MODE=shard bash vantallm/distributed/launch_deepspeed.sh <gpus>` | ZeRO initialises and trains | DeepSpeed config or offload issue |
| Prompt/RAG signal | `python3 tests/test_rag_logit_probe.py` | Context increases target-token probability | Prompt or attention-path issue |
| Server E2E | `python3 tests/test_e2e.py` | Tokenizer ↔ server ↔ model path works | LFS checkpoint, tokenizer, or API issue |

Treat the matrix as staged escalation. Do not skip directly to expensive distributed runs on new hardware.

## Prompt Parity And RAG

RAG support is implemented as a contract, not an ad hoc prompt convention:

- All training, evaluation, and inference prompt assembly is supposed to flow through `vantallm.utils.prompting.build_prompt`.
- The active prompt utility version is `v1_parity_hardened`.
- Retrieved context is inserted through the shared builder, which applies the `--- RETRIEVED CONTEXT ---` boundaries consistently.
- The repo includes two concrete validation paths: `tests/test_rag_logit_probe.py` for token-probability amplification and `tests/test_rag_readiness.py` for contradiction-style contextual response checks.

For integration details and serving payloads, see [docs/inference_rag_guide.md](docs/inference_rag_guide.md).

## Inference And Personas

The FastAPI inference server supports:

- `GET /health`
- `POST /generate`
- `POST /generate/stream`

The persona routing layer currently exposes:

- `cyber_researcher`
- `security_ciso`
- `technical_educator`
- `general_assistant`

These personas are not separate models; they are routing and conditioning modes over the same model stack. Retrieval-style routes can pass `context_chunks` and `chunk_ids`, and streaming responses are emitted as Server-Sent Events.

## Operational Notes

- `TRAIN_STEPS` is mandatory and is treated as an **absolute target step**, not “additional steps from the checkpoint.”
- On unfamiliar hardware or a new cluster image, make the first run a bounded validation with `TRAIN_STEPS=2` before attempting longer jobs.
- `./run_pipeline.sh` is the supported operator entrypoint for dataset processing.
- Do not manually edit `data_pipeline/pipeline_manifest.json` or `data_pipeline/global_hash_log.json`.
- Do not rename shard files by hand.
- The 567M config is the validated public reference path for A100 20GB-class hardware, exercised with a completed 30,000-step training run on a 20GB A100.
- Hardware scaling is driven by a small set of knobs: `MODEL_CONFIG`, `BATCH_SIZE`, `GRAD_ACCUM_STEPS`, `QUANT_MODE`, and the DeepSpeed JSON configs.
- On larger GPUs, start from the validated 567M contract and raise per-GPU batch or reduce accumulation only after the baseline path is stable.
- Change one major variable at a time when debugging distributed runs. Topology, ZeRO stage, batch size, and model config should not all move in the same iteration.
- For long or expensive distributed jobs, use `tmux` or `screen` so initialization delays do not kill the session.
- High-capacity GPUs, including RTX PRO 6000 Blackwell-class nodes, and larger multi-node deployments are documented as an additional capability in [docs/gpu_cluster_guide.md](docs/gpu_cluster_guide.md); they are not a separate codebase or special branch.

## Documentation Map

- [docs/technical_report.md](docs/technical_report.md): architectural explanation and scaling methodology.
- [docs/gpu_cluster_guide.md](docs/gpu_cluster_guide.md): hardware validation, DDP, DeepSpeed, and “big GPU” operational notes.
- [docs/inference_rag_guide.md](docs/inference_rag_guide.md): prompt parity, RAG validation, FastAPI routes, and persona behaviour.
- [docs/future_scope.md](docs/future_scope.md): two-track roadmap to a domain-adapted cybersecurity model (resourcing + engineering methodology).
- [data_pipeline/DATASET_SPEC.md](data_pipeline/DATASET_SPEC.md): dataset lifecycle, slice/shard terminology, and manifest contract.

## Repository Layout

```text
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
├── docs/               # Technical reports, guides, and future_scope.md
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

## Roadmap

Recently completed:

- [x] Batched grouped-GEMM MoE (replaces the per-expert Python loop; scales under ZeRO-3 without a parameter-fetch stall).
- [x] MoE routing load-balancing loss, averaged globally across data-parallel ranks.
- [x] 125B multi-GPU bring-up profile (`configs/vanta_125b_bringup.json`) with a locked 4-GPU ZeRO-3 launcher.
- [x] Multi-node ZeRO-3 wiring: DeepSpeed-native distributed init, hostfile launcher, 8-GPU CPU-optimizer-offload config, and automatic `DeepSpeedCPUAdam` selection — one codebase from single GPU to multi-node cluster ([`docs/multinode_deepspeed_guide.md`](docs/multinode_deepspeed_guide.md)).
- [x] MoE eval-mode capacity fix (gated on `self.training`) restoring exact KV-cache incremental decode.
- [x] Self-configuring multi-node launcher: auto-detects `MASTER_ADDR`/`NCCL_SOCKET_IFNAME` from the routing table (no hardcoded IPs) and fails fast on an unroutable topology (1:1 NAT / duplicate-IP) instead of hanging at 0% GPU.

In progress / planned:

- [ ] Expert-parallel MoE (all-to-all dispatch) for multi-node expert sharding.
- [ ] Implement FlashAttention-2 natively into the Transformer core.
- [ ] Expand the analytical multi-node DeepSpeed ZeRO-3 configurations.
- [ ] Release pre-trained base model weights.

**Where this is headed:** [docs/future_scope.md](docs/future_scope.md) lays out a
two-track plan to turn the framework into a published cybersecurity model —
(1) a resource-efficient path to a small-but-real 1B–3B model on free/low-cost
compute, and (2) the credible engineering methodology (adapt an open base, build
a cyber evaluation benchmark, and a defensive dual-use posture).

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
  url = {https://github.com/aaron-631/VantaLLM}
}
```

## Contributing

We welcome community contributions. Please review [CONTRIBUTING.md](CONTRIBUTING.md) before submitting PRs. Make sure to run all unit tests and verify your code formatting.
