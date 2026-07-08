# VantaLLM Roadmap

The following outlines the immediate and long-term architectural priorities for the VantaLLM framework.

## Near-Term Priorities (Q3 2026)
- **FlashAttention-2 Native Integration:** Migrate core multi-head attention to support FlashAttention natively without relying on external non-standard dependencies.
- **Enhanced Data Mixing:** Implement robust sampling weights across multiple data shards to seamlessly mix pretraining data from different domains.
- **DeepSpeed ZeRO-3 Validations:** Run physical experiments using the analytical 125B configurations on multi-node topologies.

## Long-Term Vision (Q4 2026 and Beyond)
- **Expert-Parallel MoE (all-to-all):** The MoE layer is now a batched grouped-GEMM
  with global load balancing (see the technical report §3.1), which makes the
  128-expert layer viable under ZeRO-3 data-parallelism. The next scale step is true
  expert-parallelism — sharding experts across GPUs and routing tokens between them
  with all-to-all — for throughput at many-GPU scale.
- **Extended Context:** Implement sliding window attention or ring-attention mechanisms for >128k context lengths.
- **Open-Source Checkpoints:** Release fully converged foundational model weights based on the 567M architecture.

### Recently Completed
- **Batched grouped-GEMM MoE dispatch** replacing the per-expert Python loop, with a
  fixed-capacity buffer and CPU-verified equivalence/checkpoint tests.
- **Global (cross-rank) MoE load-balancing loss.**
- **Lazy shared RoPE cache** removing the full-`max_seq_len` per-layer allocation.
- **125B 4-GPU ZeRO-3 bring-up path**: `configs/vanta_125b_bringup.json`,
  `scripts/launch_vanta_4gpu.sh`, params-only offload, gradient clipping, and a
  fail-fast batch-consistency check.
- **Multi-node DeepSpeed enablement**: DeepSpeed-native distributed init (so ZeRO-3
  sees the true world size), a hostfile-based multi-node launcher, an 8-GPU ZeRO-3
  CPU-optimizer-offload config, and automatic use of `DeepSpeedCPUAdam` when the
  config owns the optimizer. Full cluster runbook + bring-up pitfalls in
  [`docs/multinode_deepspeed_guide.md`](docs/multinode_deepspeed_guide.md).
- **Self-configuring multi-node launcher**: `launch_deepspeed.sh` now derives
  `MASTER_ADDR` and `NCCL_SOCKET_IFNAME` from the live routing table (no hardcoded
  IPs or NIC names — portable across any cluster) and runs a topology sanity
  preflight that fails fast, instead of hanging at 0% GPU, when nodes lack a
  directly-routable private link (1:1 NAT / duplicate-IP detection).
- **MoE eval-mode capacity fix**: capacity token-dropping is gated on `self.training`
  so incremental KV-cache decode keeps every token and stays exact (KV-cache ==
  no-cache invariant restored).

## Task Backlog

- Resume shard-mode pre-training from the active 40k checkpoint against the current 15-shard corpus.
- Validate checkpoint-backed inference against the active medium-model path and reconcile any remaining training/inference checkpoint-loading differences.
- Review scraped raw sources for poisoning, junk, and low-value duplicates before the next long run.
- Import or archive any external GPU-day and cluster artifacts into a reproducible manifest-driven layout without polluting the active repo workflow.
- Resolve any remaining LR and scaling-policy differences between local training defaults and DeepSpeed configs before the next large cluster run.
- Add run-to-run MoE utilization summaries so routing skew can be compared across checkpoints.
- Add a documented corpus-mixing policy for future RAG or instruction-tuning data so those sets do not distort base pre-training.
- Evaluate post-40k and later checkpoints for long-form technical coherence once the next resume run completes.
- Tighten inference route validation and serving ergonomics after the active pre-training phase is stable.
- Add stronger corpus QA tooling around data ingestion and delta review.
