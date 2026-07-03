# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **Packaging via `pyproject.toml` (audit Finding F23).** The repo previously had
  no `pyproject.toml`/`setup.py`, so it ran only from a checkout and was not
  `pip install`-able. Added a PEP 621 `pyproject.toml` (setuptools backend,
  version pulled dynamically from `vantallm.__version__`) that packages the
  `vantallm` package plus the top-level `verify_gpu_ready` / `verify_distributed`
  helpers, with `[dev]` (pytest) and `[data]` (datasets) extras; torch stays out
  of the dependency set by design. *Verification:* `python -m pip wheel .`
  builds `vantallm-0.1.0-py3-none-any.whl`, and the wheel contains
  `vantallm/model/transformer.py` and `verify_gpu_ready.py`. README documents
  `pip install -e .`.
- **README install ordering clarified (audit Finding F24).** Made explicit that
  the PyTorch wheel must be installed before `requirements.txt` (deepspeed builds
  against an already-present torch) and documented the editable install.
- **Pytest-native smoke suite (`tests/test_smoke.py`) + `tests/conftest.py` (audit Finding 6).**
  `pytest tests/` now discovers and runs 8 fast tiny-CPU tests (build, forward shapes, weight
  tying, full-gradient coverage, KV-cache length, config validation, tokenizer round-trip); the
  script-style server/RAG suites are excluded from collection but still run directly and in CI. A
  pinned `requirements-dev.txt` (adds `pytest==9.1.1`) and a `pytest tests/` CI step were added.
  *Verification:* `pytest tests/ -q` → **8 passed**.
- Core modular decoder-only Transformer implementation.
- Distributed Data Parallel (DDP) and DeepSpeed ZeRO integration.
- Custom dataset sharding and tokenization pipeline.
- BF16 Tensor Core acceleration for A100 architectures.
- FastAPI-based inference server for checkpoint evaluation.
- Analytical scaling configs for 125B and 235B parameter scales.

### Fixed
- **Batched decode now honors the left-pad attention mask and true RoPE
  positions (audit Finding F12).** `vantallm/inference/server.py` extended
  `batched_masks` every decode step but only passed it to the model at step 0
  (`current_mask = batched_masks if step == 0 else None`), so decode queries in a
  mixed-length batch attended to left-pad KV positions; decode `position_ids`
  were also taken from the padded cache length, giving shorter sequences the
  wrong relative offsets. The mask is now passed on every step and decode
  positions are computed from each prompt's true length
  (`seq_true_len + (step-1)`). *Why:* correctness of multi-request batched
  generation. *Verification:* padded-vs-unpadded final-token logits for a
  length-5 prompt left-padded to 8, over a 4-step greedy decode —
  `max|Δ| = 1.83` (before) → `3.6e-07` (after); `pytest tests/` **8 passed**,
  system phase 3 server e2e **25/25**.
- **Cached chat sessions prefill the entire new turn, not just its last token
  (audit Finding F13).** The stateful path fed only `generated[:, -1:]` into the
  reused KV cache, silently dropping every token of the new user message except
  the last. It now prefills the full new turn one token at a time (each `S=1`
  step is correctly masked against the existing cache offset) before decoding.
  *Why:* the model must condition on the whole new message. *Verification:*
  next-token logits vs a full-context forward — `max|Δ| = 0.41` (last-token-only)
  → `4.4e-03` (full prefill; residual is TurboQuant INT8 KV-cache error, not
  logic); phase 3 server e2e **25/25**.
- **Shard dataset is memory-mapped instead of full-loading/thrashing (audit
  Finding F08).** `verify_gpu_ready.py::ShardedDataset` — used by the real
  training path (`train.py` imports it for shard mode) — loaded every shard fully
  into RAM at `__init__` just to read lengths, and reloaded an entire shard from
  disk on every cross-shard index in `__getitem__`, so `shuffle=True` thrashed.
  Shards are now opened once with `torch.load(..., mmap=True)`; lengths come from
  metadata and `__getitem__` binary-searches the shard and faults in only the
  requested row. *Why:* contradicted the streaming/low-memory claim and made
  shuffled multi-shard training pathological. *Verification:* 3 synthetic shards
  (9 sequences); shuffled cross-shard access returns all 9 rows correctly with no
  whole-shard reload; label pad→-100 masking preserved.
- **DeepSpeed micro-batch follows `BATCH_SIZE` instead of a hardcoded JSON value
  (audit Finding F01).** `zero3.json` pinned `train_micro_batch_size_per_gpu: 2`,
  silently overriding the `BATCH_SIZE` env var the rest of the trainer reads.
  `distributed_utils.py` now loads the config as a dict and injects
  `BATCH_SIZE` (and optional `DS_GRAD_ACCUM_STEPS`) before `deepspeed.initialize`.
  *Why:* a user setting `BATCH_SIZE` under DeepSpeed got a different batch than
  requested. *Verification:* with `BATCH_SIZE=8`, injected
  `train_micro_batch_size_per_gpu` = 8; `zero3.json` still valid JSON.
- **DDP cosine LR schedule sized by optimizer steps, not micro-batches (audit
  Finding F11).** On the DDP path `_ds_accum` was 1, so the cosine was sized for
  `num_steps` micro-batches while `lr_scheduler.step()` fires only at
  gradient-accumulation boundaries (`num_steps // GRAD_ACCUM_STEPS` times) — it
  never reached `eta_min`. The DDP divisor is now `GRAD_ACCUM_STEPS`.
  *Verification:* `num_steps=16000, GRAD_ACCUM_STEPS=16` → actual scheduler
  steps 1000; cosine `T` 16000 (before) → 1000 (after), matching.
- **Checkpoint loads report missing/unexpected keys (audit Finding F16).** Both
  `train.py` and `server.py` called `load_state_dict(strict=False)` without
  inspecting the result, so a partially-incompatible checkpoint loaded silently.
  Both now capture the return and log missing/unexpected key counts (and a clean
  "all keys matched" line in train). *Verification:* loading a state dict with a
  dropped key + a bogus key surfaces both lists.
- **Gradient clipping drops the per-parameter host↔device sync (audit Finding
  F20).** `clip_grad_norm_mixed_` called `param_norm.item()` inside a per-
  parameter loop (hundreds of CUDA syncs per step). Norms are now stacked and
  reduced on-device with a single `.item()` at the end. *Why:* throughput.
  *Verification:* global grad-norm matches the reference to `<1e-3`
  (8.506624 both), post-clip norm = 1.0 at `max_norm=1.0`; `pytest tests/`
  **8 passed**.
- **Corrected the `~2-4B active per token` docstring (audit Finding F07).**
  `transformer.py::VantaTransformer` claimed ~2-4B active for the 125B config;
  the analytic active count is ~13B (always-on dense FFN ~7.1B dominates, plus
  attention ~3.9B, embedding ~0.6B, and top-2 MoE ~1.8B). Docstring updated and
  now explains the always-on dense FFN dominates the active budget.
- **Removed the unused `transformers` dependency; scoped `datasets` to data-prep
  (audit Finding F19).** `transformers==5.8.0` was pinned but imported nowhere
  (`grep` → 0 hits); removed. `datasets` is only used by scrape/tokenizer-build
  scripts, so it moved to a documented "data-pipeline only" note instead of the
  runtime install list. *Verification:* `grep -rn 'import transformers'` → 0.
- **Corrected the DDP `find_unused_parameters` rationale (audit Finding 5).** The prior code
  comment ("sparse MoE experts can receive no tokens") was empirically false for this
  implementation — the dense per-expert dispatch loop gives every expert a gradient (zero, not
  `None`) even with 2 tokens across 8 experts. The actual (and only) unused parameter on the
  training path is the inference-only `persona_vec_to_composite` projection. The flag is kept
  `True` (robust to that, and to future conditional-compute / frozen-layer changes) with an
  accurate comment. *Verification:* `pytest`'s `test_backward_covers_expected_params` asserts the
  training-path unused set is exactly `["persona_vec_to_composite.weight"]` and that no MoE expert
  weight is ever unused.
- **README CPU torch install pulls the CPU wheel (audit Finding 4).** `README.md` "CPU-only local
  smoke testing" line was `pip install torch==2.11.0` (resolves to the multi-GB CUDA build); now
  `pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cpu`. *Verification:*
  line inspected post-edit.
- **Stale MoE dispatch comment (audit Finding 8).** `transformer.py` said `scatter_add_` while the
  code uses `index_add_`; comment corrected to describe the actual duplicate-index accumulation.
- **Continuous integration added (audit Finding 3).** New `.github/workflows/ci.yml` runs on push
  and pull requests to `main`: installs the pinned CPU torch + `requirements.txt`, then executes
  the built-in architecture demo, `tests/test_task3.py`, `test_system.py` phases 1/2/3, and
  `verify_distributed.py` under `TINY_MODEL=1`. A CI status badge was added to the top of
  `README.md`. *Why:* a 90-check suite existed but nothing enforced it; regressions could land
  silently. *Verification:* the workflow YAML parses (10 steps under job `cpu-smoke`) and every
  command it runs was re-executed locally with exit 0 — demo "All checks passed", test_task3
  passed, phase 1 41/41, phase 2 20/20, phase 3 25/25, verify_distributed "SINGLE-NODE CHECKS
  PASSED".
- **Dependencies pinned in `requirements.txt` (audit Finding 2).** All 14 runtime dependencies
  were previously unpinned, so a fresh install resolved to bleeding-edge versions and would drift.
  Each is now pinned with `==` to the exact set verified end-to-end during the audit; torch stays
  intentionally unpinned with a documented CUDA/CPU install matrix. *Why:* a reviewer must be able
  to reproduce the environment the results were produced in. *Verification:* the audit's
  `grep -vE '^#|^$' requirements.txt | grep -vE '[=<>~!]'` reported 14 unpinned lines before and
  **0 after**; all 14 pins match a resolvable, installed version (fastapi 0.136.1, uvicorn 0.46.0,
  pydantic 2.13.3, transformers 5.8.0, tokenizers 0.22.2, requests 2.33.1, numpy 2.4.4,
  pandas 3.0.2, ipykernel 7.2.0, wrapt 2.1.2, datasets 4.8.5, safetensors 0.7.0, ninja 1.13.0,
  deepspeed 0.19.0).
- **Repository URL corrected across docs (audit Finding 1).** `README.md` (clone command and
  BibTeX citation) and `CITATION.cff` (`url`, `repository-code`) pointed at
  `github.com/aaronchakraborty/VantaLLM`, which does not match the actual remote
  `github.com/aaron-631/VantaLLM` — meaning the first Quick Start command cloned a non-existent
  repo. *Why:* a clone URL that 404s is a blocking reproducibility defect. *Verification:*
  `grep -rn 'github.com/aaronchakraborty'` over `*.md`/`*.cff` → 4 hits before, 0 after; the
  README clone URL now equals `git remote get-url origin`; `CITATION.cff` still parses as valid
  YAML (`url = https://github.com/aaron-631/VantaLLM`).

### Initialized
- First open-source release of VantaLLM architecture.
