# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
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
