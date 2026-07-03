# VantaLLM Roadmap

The following outlines the immediate and long-term architectural priorities for the VantaLLM framework.

## Near-Term Priorities (Q3 2026)
- **FlashAttention-2 Native Integration:** Migrate core multi-head attention to support FlashAttention natively without relying on external non-standard dependencies.
- **Enhanced Data Mixing:** Implement robust sampling weights across multiple data shards to seamlessly mix pretraining data from different domains.
- **DeepSpeed ZeRO-3 Validations:** Run physical experiments using the analytical 125B configurations on multi-node topologies.

## Long-Term Vision (Q4 2026 and Beyond)
- **Mixture of Experts (MoE):** Introduce sparse expert routing to increase parameter counts without equivalent scaling in FLOPs.
- **Extended Context:** Implement sliding window attention or ring-attention mechanisms for >128k context lengths.
- **Open-Source Checkpoints:** Release fully converged foundational model weights based on the 567M architecture.

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
