# Inference, RAG, and Prompt Parity Guide

This guide documents the serving interface and the prompt/RAG contract used across VantaLLM. The goal is simple: the same repository should support local smoke tests, checkpoint-backed inference, and retrieval-augmented serving without forked prompt templates or hidden route-specific logic.

## Prompt Parity Contract

VantaLLM treats prompt formatting as a shared systems invariant.

- The authoritative prompt builder lives in `vantallm/utils/prompting.py`.
- The active prompt contract version is `v1_parity_hardened`.
- Training, evaluation, RAG tests, and the FastAPI server are all expected to use `build_prompt()`.

The builder:

- applies the system prompt consistently
- wraps user content with `<|user|>` and `<|assistant|>`
- inserts retrieved context between deterministic `--- RETRIEVED CONTEXT ---` markers
- strips legacy raw tags to avoid double-wrapping bugs

Operational rule: do not hand-assemble chat markers or RAG separators in downstream scripts if `build_prompt()` can be used instead.

## RAG Request Contract

Retrieval-aware inference is passed through the same request schema as standard generation.

Relevant request fields:

- `prompt`: the user query
- `route`: one of `general`, `exact`, `hybrid`, or `oos`
- `context_chunks`: retrieved text passages
- `chunk_ids`: parallel identifiers for the retrieved passages

For retrieval-style routes (`exact` and `hybrid`), the server joins the supplied context chunks and passes them through the shared prompt builder before tokenization.

`oos` is a deliberate out-of-scope refusal route and returns the refusal response directly.

## Validation Paths

The repository includes multiple ways to verify that RAG behavior is real rather than cosmetic.

### 1. Logit Amplification Probe

`tests/test_rag_logit_probe.py` measures whether adding retrieved context increases the probability of a target token at the logits level.

Use this when you want evidence that attention is moving in the right direction even if the model is too undertrained to produce polished text.

### 2. Contradiction-Style Readiness Test

`tests/test_rag_readiness.py` uses contextual facts that contradict common priors and checks whether the model follows the supplied context.

Use this when you want to validate the path:

```text
instruction -> retrieved context -> answer
```

instead of:

```text
instruction -> internal prior -> answer
```

### 3. End-to-End Server Test

`tests/test_e2e.py` validates the tokenizer, checkpoint loading, FastAPI server, and `/generate/stream` route together.

If a checkpoint is still a Git-LFS pointer stub, this test will fail early and tell you to run `git lfs pull`.

## FastAPI Endpoints

The inference server lives at `vantallm/inference/server.py` and exposes three primary routes.

### `GET /health`

Returns runtime metadata including:

- server status
- active device
- approximate model parameter count
- KV-session count
- engine queue depth

This is the first endpoint to hit after boot.

### `POST /generate`

Non-streaming generation route.

Key request fields:

- `prompt`
- `route`
- `context_chunks`
- `chunk_ids`
- `max_new_tokens`
- `temperature`
- `top_k`
- `top_p`
- `repetition_penalty`
- `session_id`

Response fields include:

- `request_id`
- `route`
- `persona`
- `answer`
- `input_tokens`
- `output_tokens`
- `latency_ms`
- `oos`

### `POST /generate/stream`

Streaming generation route using Server-Sent Events.

The server buffers subword fragments into more readable chunks and emits events shaped like:

```json
{"token": "..."}
```

followed by:

```json
{"done": true}
```

This is the preferred route for interactive clients.

## Persona Routing

Persona selection is implemented as an inference-time conditioning layer over the same model checkpoint.

Documented personas:

- `cyber_researcher`: strict security-domain grounding
- `security_ciso`: strategic and executive-style security framing
- `technical_educator`: high-detail explanatory mode
- `general_assistant`: general helpful fallback
- `oos_refusal`: guardrail refusal mode for out-of-scope routes

The server selects personas based on the explicit route first and then on lightweight keyword heuristics.

These are not separate weights. They are conditioning paths over one shared transformer.

## Serving Notes

- `session_id` can be supplied to reuse KV-cache sessions across related requests.
- Prompt length plus `max_new_tokens` must fit under the server context window.
- For checkpoint-backed inference, hydrate any LFS-tracked checkpoints before server startup.
- For production-style use, tighten CORS and review the default open middleware policy.
