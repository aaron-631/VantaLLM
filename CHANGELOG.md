# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Core modular decoder-only Transformer implementation.
- Distributed Data Parallel (DDP) and DeepSpeed ZeRO integration.
- Custom dataset sharding and tokenization pipeline.
- BF16 Tensor Core acceleration for A100 architectures.
- FastAPI-based inference server for checkpoint evaluation.
- Analytical scaling configs for 125B and 235B parameter scales.

### Initialized
- First open-source release of VantaLLM architecture.
