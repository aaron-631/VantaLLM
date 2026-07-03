# Operational Handoff

This file is the compact live continuation guide for VantaLLM.

## Current Continuation Path

- Active reference path: `configs/vanta_567m.json`
- Active precision path: `bf16`
- Active pipeline entrypoint: `./run_pipeline.sh`
- Active training entrypoint: `python3 vantallm/training/train.py`
- Primary validation flow: transformer self-test, distributed sanity check, shard validation, then smoke training

## Checkpoint Discipline

- Checkpoints are tracked through Git-LFS.
- A source-only clone can have the checkpoint references in code while the actual `.pt` files are absent.
- Before any resume or E2E server check that depends on weights, hydrate the repo with `git lfs pull`.
- Treat `TRAIN_STEPS` as an absolute target step count, not an increment from the resume checkpoint.

## Resume Sequence

Use this order on a new machine or cluster image:

1. `python3 -m vantallm.model.transformer`
2. `python3 verify_distributed.py`
3. `python3 verify_gpu_ready.py`
4. `TRAIN_STEPS=10 TINY_MODEL=1 DATA_MODE=showcase DATA_PATH=data_pipeline/rag_synthetic_slice_v2_fixed.jsonl python3 vantallm/training/train.py`
5. `MODEL_CONFIG=configs/vanta_567m.json TRAIN_STEPS=1000 DATA_MODE=shard bash vantallm/distributed/launch_ddp.sh 1`

If a checkpoint is being resumed explicitly, set `MODEL_PATH` to the hydrated checkpoint file and make sure the target `TRAIN_STEPS` is larger than the step encoded in that checkpoint.

## Hardware Policy

- Use the 567M path as the baseline on A100 20GB hardware.
- Keep `QUANT_MODE=bf16` as the default operator stance on modern NVIDIA GPUs.
- Increase only one major knob at a time: batch size, accumulation, ZeRO stage, or model config.
- Keep the Blackwell guidance in `docs/gpu_cluster_guide.md` as the scaling note for higher-memory single-node systems.

## Operational Rules

- Do not edit `data_pipeline/pipeline_manifest.json` or `data_pipeline/global_hash_log.json` by hand.
- Do not rename shard files by hand.
- Treat `./run_pipeline.sh` as the supported dataset-processing entrypoint.
- Use `tmux` or `screen` for long distributed runs.
