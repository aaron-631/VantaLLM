# Future Scope

VantaLLM today is a complete, from-scratch training framework validated end-to-end
at the 567M-parameter scale (a 30,000-step run on a single 20GB A100). The
architecture, training loop, tokenizer, and data pipeline are real and scale by
JSON configuration alone — the gap between the current state and a released
domain model is **data, compute, and evaluation**, not core engineering.

This document lays out the two complementary directions that gap can be closed
along. They are not mutually exclusive; Track 1 is the resourcing strategy, and
Track 2 is the engineering methodology that makes the result credible.

---

## Track 1 — A small-but-real cybersecurity model on student-grade compute

The realistic near-term goal is **not** to train a frontier base model from
scratch (that is a capital problem, not an engineering one). It is to produce a
**working 1B–3B cybersecurity language model, trained with this framework and
published with a model card** — a complete, downloadable artifact rather than an
un-run analytical config.

**Why this is the right target.** A 1B–3B model is trainable in days on free or
low-cost compute, is small enough to serve cheaply, and — critically — is a
*shipped* result. A published, `pip install`-able model with measurable behavior
is a stronger portfolio and research signal than a large configuration that was
never executed.

**Resourcing paths (free to low-cost).**

| Tier | Resource | Notes |
|---|---|---|
| Free (grant) | Google TPU Research Cloud (TRC) | Free TPU v3/v4 access for research proposals; VantaLLM + a cybersecurity training goal is a fitting application. *Caveat: requires a `torch_xla` backend port; the current CUDA/DeepSpeed path targets the GPU providers below.* |
| Free (quota) | Kaggle Kernels | ~30 GPU-hours/week (P100/T4), no card required — enough for small real runs. |
| Free (grant) | Hugging Face community compute grants | Aimed at open-source model projects. |
| Free (credits) | Cloud provider student/startup credits | Google Cloud / AWS / Azure typically offer $100–300 in credits. |
| Low-cost (spot) | Vast.ai / RunPod spot GPUs | 4090 / A100 at roughly $0.30–1.00/hr; small budgets buy meaningful training. |

**Steps.**

1. Add a **1B–3B config** (`configs/`) — the JSON scaling system already supports
   this without code changes.
2. Point the pipeline at a **larger, higher-quality cybersecurity corpus** (see
   Track 2) rather than the current example mixture.
3. Train, evaluate (Track 2), and **publish to Hugging Face** (target:
   `aaron-631/VantaLLM-cyber-1B`) with a model card, intended-use statement, and
   evaluation results.

---

## Track 2 — The credible engineering path to a domain model

Producing a model that can *honestly* be called a "cybersecurity model" requires
more than a training run. Three engineering decisions determine whether the
result is credible or merely a demo.

### 2.1 Base-model decision: adapt, don't restart

Training a competitive base model from scratch is not a good use of limited
compute. The resource-efficient, standard-practice move is to **domain-adapt an
existing open base model** (continued pretraining / instruction tuning on a
cybersecurity corpus). Concretely, this means the framework needs a clean
**"load external pretrained weights → continue training"** path.

- *Status:* the training and checkpoint/resume machinery already exists. The
  specific piece to confirm or build is **loading external (e.g. Hugging Face)
  base weights into VantaLLM's architecture** — an architecture/weight-mapping
  task, since VantaLLM's module layout is its own.
- The cheap, fast first proof is a **LoRA / QLoRA fine-tune of a ~7B open base**:
  it validates the corpus and the domain-adaptation loop for a few hundred
  dollars (or less on grant compute) before any from-scratch scale-up.

### 2.2 Evaluation: the claim needs a benchmark

A model cannot be called a cybersecurity model without an evaluation that shows
**measurable lift over the base model on cyber tasks**. Building this is a real,
separate sub-project:

- CTF-style reasoning tasks
- CVE / vulnerability reasoning and summarization
- Secure-code and vulnerable-code detection
- Comparison against the un-adapted base to demonstrate the delta

The evaluation set is as much a deliverable as the model itself.

### 2.3 Safety and dual-use posture

A capable security model has a genuine misuse surface. For a public,
resume-facing project this is a first-class design constraint, not an
afterthought. VantaLLM's intended framing is **defensive and analysis-oriented**:

- Scope the model toward *understanding, detection, triage, and explanation*
  (CVE analysis, log/alert reasoning, secure-code review) rather than offensive
  automation.
- Ship an explicit intended-use / out-of-scope statement in the model card, and
  keep the existing [SECURITY.md](../SECURITY.md) posture.
- Treat guardrails and refusal behavior on operational-attack requests as part of
  the evaluation, not a bolt-on.

Framed this way, a cybersecurity model is an asset. Framed carelessly, it is a
liability — the difference is deliberate scoping and disclosure.

---

## Realistic order of execution

1. **Confirm/build the load-external-pretrained-weights path** — adapt an open
   base instead of training from zero.
2. **Build the cybersecurity corpus** — the real moat; start small and
   high-quality (CVE/NVD, MITRE ATT&CK, exploit and advisory writeups, secure-code
   examples, security Q&A) and grow it under the pipeline's dedup/QA discipline.
3. **LoRA / QLoRA fine-tune a ~7B base first** — cheap, fast, proves the concept.
4. **Build the cyber evaluation set** — demonstrate measurable lift over the base.
5. **Scale up only then** — larger config and/or continued pretraining once the
   corpus and eval prove the approach is working.

This sequence front-loads the cheap, decisive experiments (does the corpus
produce measurable domain lift on a small adapted model?) before committing any
serious compute — which is exactly the discipline the rest of the framework is
built around.
