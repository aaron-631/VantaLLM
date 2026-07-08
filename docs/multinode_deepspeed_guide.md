# Multi-Node DeepSpeed Guide

How VantaLLM scales from a single GPU to a multi-node cluster **without a separate
codebase** — the same model file, the same launchers, the same JSON config system.
This guide covers the wiring for a 2-node / 8-GPU ZeRO-3 run, the failures observed
while bringing one up, and how each was fixed.

> **Scope.** The single-node reference path (`vanta_567m.json`, DDP or ZeRO-2) is the
> validated baseline. This document is the escalation path to multi-node ZeRO-3 with
> CPU optimizer offload for the large analytical configs (e.g. the 125B profile).
> Bring one up only after the single-node path is stable — see
> [gpu_cluster_guide.md](gpu_cluster_guide.md).

---

## 1. Why one codebase spans every scale

VantaLLM is **dimension-dynamic by construction.** Every architectural size lives in a
JSON config consumed by `LLMConfig`; the model code hardcodes nothing about scale:

| Layer | What is dynamic | Where |
|---|---|---|
| Architecture | `d_model`, `num_layers`, `num_experts`, `d_ff_moe`, `max_seq_len`, `top_k_experts` | model config JSON → `LLMConfig` |
| Precision | `bf16` / `fp16` / `int8` / `int4` | config `quantization` → `build_linear()` |
| Memory strategy | DDP · ZeRO-1/2 · ZeRO-3 · CPU offload | DeepSpeed JSON, **independent of the model** |
| Topology | 1 GPU · N GPUs · N nodes | launcher arg + `hostfile` |
| Batch | per-rank micro-batch, grad-accum, global batch | `BATCH_SIZE` env + DeepSpeed JSON |

The consequence: **moving A100 20 GB → RTX PRO 6000 96 GB → H200 / multi-node is a
config + topology change, not a code fork.** The same `transformer.py` serves the 567M
reference and the 125B analytical profile. This guide is the topology layer of that story.

**Rule of thumb (analytic memory sizing):**
- ≲ 4B params → single GPU or ZeRO-2, no offload.
- ≳ 30B params → ZeRO-3 with CPU offload.
- 100B+ params on commodity nodes → ZeRO-3 + optimizer offload + multi-node, and the
  host RAM per node becomes the binding constraint (see §5).

---

## 2. Cluster topology (reference)

| Role | Private IP | GPUs | Host RAM |
|---|---|---|---|
| Master (`node_rank=0`) | `10.0.0.2` | 4 × 96 GB | ~700 GB |
| Worker (`node_rank=1`) | `10.0.0.3` | 4 × 96 GB | ~700 GB |

- **Not a shared filesystem.** Each node has its own repo checkout and its own Python
  env at the **same absolute path**. Any code/config edit must be synced to every node
  before a run, or ranks silently run different code.
- **Interconnect is standard Ethernet, not InfiniBand** (`NCCL_IB_DISABLE=1`,
  `NCCL_SOCKET_IFNAME=eth0`). At full scale, ZeRO-3's cross-node parameter traffic — not
  compute — is the likely bottleneck (§6).

---

## 3. Repo files that make multi-node work

| File | Role |
|---|---|
| `vantallm/distributed/init_distributed.py` | Initializes distributed **through DeepSpeed** so ZeRO sees the real world size (§4, Pitfall 6). |
| `vantallm/distributed/launch_deepspeed.sh` | Single-node **and** hostfile-based multi-node launch. |
| `vantallm/distributed/hostfile` | Node list + `slots=` (GPUs/node). Determines process count. |
| `vantallm/distributed/deepspeed_configs/zero3_offload_multinode.json` | 8-GPU ZeRO-3 config with CPU optimizer offload. |
| `vantallm/training/train.py` | Fail-fast batch check + config-owned-optimizer detection (§4, Pitfall 9). |

### hostfile

```
10.0.0.2 slots=4
10.0.0.3 slots=4
```

`slots=` is how many GPUs DeepSpeed uses on that node **for this run** — it need not equal
the physical GPU count. This value, not the launcher's `<num_gpus>` argument, sets the
process count in hostfile mode. For a 2-GPU smoke test (1/node) set both to `slots=1`.

### Launch

```bash
# Single-node (unchanged):
bash vantallm/distributed/launch_deepspeed.sh 4 vantallm/distributed/deepspeed_configs/zero2.json

# Multi-node (8 GPUs, ZeRO-3 + offload):
bash vantallm/distributed/launch_deepspeed.sh 4 \
     vantallm/distributed/deepspeed_configs/zero3_offload_multinode.json \
     vantallm/distributed/hostfile
```

`train_batch_size` in the config must equal `micro_batch × grad_accum × (total GPUs across
the hostfile)`. The shipped multi-node config uses `1 × 16 × 8 = 128`.

---

## 4. Pitfalls observed and fixed

These were hit, in order, bringing a fresh 2-node cluster to a working run. Infra items
are noted so an operator can reproduce the environment; the code items are already fixed
in this repo.

| # | Symptom | Root cause | Fix |
|---|---|---|---|
| 1 | `No such rcmd module "ssh"` at launch | `pdsh` built without SSH support | Install the pdsh ssh rcmd module on the **master** |
| 2 | `pdsh: connect: Connection refused` after installing the module | pdsh defaulted to legacy `rsh` (port 514) | `export PDSH_RCMD_TYPE=ssh` on the master |
| 3 | Worker `No route to host` to `10.0.0.2:29500` | Host firewall blocked the rendezvous port + NCCL's ephemeral range on the private subnet | Trust the `10.0.0.0/24` source zone (or disable the host firewall on both nodes) |
| 4 | Training hung at 0% GPU, no crash | Stale `NCCL_SOCKET_IFNAME=lo` / `MASTER_ADDR=localhost` left from a single-node test | Set `NCCL_SOCKET_IFNAME=eth0`, `MASTER_ADDR=10.0.0.2` on **both** nodes |
| 5 | `AssertionError: train_batch_size != micro × grad_acc × world_size` | Config batch math assumed a different GPU count than the run | Recompute `gradient_accumulation_steps` for the world size actually in use (the trainer's fail-fast check now catches this in seconds, before the long ZeRO-3 init) |
| **6** | Same assertion, but **`world_size` always shows as 1** regardless of cluster size | **Code bug:** raw `torch.distributed.init_process_group()` never initializes DeepSpeed's own comm backend, which `zero.Init()` reads from | **Fixed:** `init_distributed.py` now calls `deepspeed.init_distributed()` |
| 7 | `torch.OutOfMemoryError` during model construction | Full 125B sharded across only 2 GPUs still needs ~125 GB/GPU for params | Not a bug — use a smaller config, or scale to the full 8-GPU + offload setup (§5) |
| 8 | Config edits didn't take effect on the worker | No shared filesystem — the edit reached only the master | Sync every changed file to **every** node (§7) |
| **9** | `ZeRORuntimeException: You are using ZeRO-Offload with a client provided optimizer` | ZeRO-Offload requires its own `DeepSpeedCPUAdam` and rejects a client optimizer | **Fixed:** the offload config defines an `"optimizer"` block, and `train.py` now detects it and passes `optimizer=None` so DeepSpeed builds `DeepSpeedCPUAdam` |
| **10** | `NCCL error: invalid usage ... receiving N bytes instead of M`, collective timeout on the first forward | **Historical model bug:** an MoE forward that skipped an expert (`continue`) on ranks with zero routed tokens desynced the ZeRO-3 param-gather collective order across ranks | **Already fixed** by the batched grouped-GEMM MoE: every rank runs all experts via `torch.bmm` over a fixed-capacity buffer, so the collective sequence is identical on every rank by construction |

### 4.1 The MoE eval-mode invariant (a related correctness fix)

The batched MoE applies **fixed-capacity token dropping**, which is derived from
`T = B·S`. A full forward (`T` = whole sequence) can drop overflow tokens, but incremental
KV-cache decode (`T = 1` per step) never does — so the two produce different MoE outputs
from the **same weights**, diverging the token streams and breaking the
`KV-cache == no-cache` invariant. This surfaced as a data-dependent test failure.

The fix gates capacity dropping on `self.training`:
- **Training** → fixed capacity kept → per-step VRAM guarantee unchanged.
- **Eval / inference** → capacity sized to the actual max per-expert load, every token
  kept → KV-cache decode is exact again.

This fix is verified by the KV-cache incremental-decode parity test in the system suite
(test 2.4, "KV-cache tokens match no-cache tokens").

---

## 5. Memory math at full 125B / 8 GPUs

ZeRO-3 mixed-precision Adam costs ~16 bytes/param fully on-GPU: 2 (bf16 param) + 2 (bf16
grad) + 4 (fp32 master) + 4 (fp32 momentum) + 4 (fp32 variance).

| Scenario | Per-GPU | Fits 96 GB? |
|---|---|---|
| 125B, ZeRO-3, no offload, 8 GPUs | `125e9 × 16 / 8 ≈ 250 GB` | ✗ |
| 125B, ZeRO-3, **optimizer offloaded to CPU**, 8 GPUs (params+grads only, 4 B/param) | `125e9 × 4 / 8 ≈ 62.5 GB` | ✓ ~33 GB headroom for activations |

The offloaded optimizer state (12 B/param: fp32 master + momentum + variance) totals
`125e9 × 12 ≈ 1.5 TB`, sharded across 8 ranks ≈ **~187 GB/rank ≈ ~750 GB/node** (4
ranks/node) — right at the ~700 GB host RAM per node, **with essentially no margin.** Two
mitigations are baked into the shipped config and model:

- `"pin_memory": false` — pinned memory can't be reclaimed by the OS; leaving it unpinned
  trades some transfer speed for lower OOM-kill risk.
- **gradient checkpointing** (`gradient_checkpointing: true` in the model config) reduces
  GPU activation memory, which matters more here than usual because the CPU side has no
  offload margin left.

**Single-node vs multi-node optimizer choice — they differ, and both are correct:**

- **Single node (e.g. one 4×96 GB box):** the fp32 optimizer state can't be sharded across
  nodes, so an fp32 `DeepSpeedCPUAdam` would need ~1.5 TB on one host and OOM. Here
  `bitsandbytes.PagedAdamW8bit` (8-bit state, ~250 GB, self-paging) is used and the config
  offloads **params only** (no `offload_optimizer`). This is the `zero3.json` path.
- **Multi node (2× box):** ZeRO-3 shards the fp32 optimizer state across all 8 ranks, so it
  fits per-node host RAM. Here the config offloads the **optimizer** and lets DeepSpeed
  build `DeepSpeedCPUAdam` (Pitfall 9). This is the `zero3_offload_multinode.json` path.

The trainer picks the right optimizer automatically: if the DeepSpeed config defines an
`"optimizer"` block, it skips the local bnb optimizer; otherwise it builds `PagedAdamW8bit`.

**Watch host RAM live on both nodes during the first run** — given zero margin, catching an
OOM-kill on a short run beats discovering it hours into a real one:

```bash
watch -n2 free -h
```

---

## 6. Network caveat at full scale

The nodes are connected over standard Ethernet, not InfiniBand. At 8 GPUs, ZeRO-3's
parameter all-gather / reduce-scatter crosses the network **every step**, not just within a
node. `zero3_offload_multinode.json` tunes for this: `reduce_bucket_size: 2e8` and
`stage3_prefetch_bucket_size: 1e8` keep individual messages smaller so they don't
monopolize the link, and `overlap_comm: true` overlaps communication with compute. If step
time is dominated by communication rather than compute once running, this is the first place
to look.

A note on MoE cost at scale: with `max_seq_len` in the thousands and top-2-of-128 routing, a
single micro-batch already selects thousands of (token, expert) pairs — statistically enough
that nearly every expert is touched on every rank anyway. The dominant cost is ZeRO-3's
per-expert parameter all-gather over Ethernet at 125B scale, not MoE routing itself.

A bring-up run's bar is narrow: **complete a small fixed number of steps (e.g. 32) with the
MoE active and no OOM** — a "prove it runs" milestone, not a converged run.

---

## 7. Operating discipline (no shared filesystem)

Before every multi-node run, if any file changed locally, sync it to **every** node — there
is no shared FS:

```bash
# once per changed file, once per node, from wherever you drive the cluster
scp <path> root@<master>:/root/VantaLLM/<path>
scp <path> root@<worker>:/root/VantaLLM/<path>
```

Only `MASTER_ADDR`, `NCCL_SOCKET_IFNAME`, `NCCL_IB_DISABLE`, `NCCL_DEBUG`, and `PYTHONPATH`
are auto-forwarded to workers by DeepSpeed's launcher. Everything else — `MODEL_CONFIG`,
`BATCH_SIZE`, `TRAIN_STEPS`, `DATA_MODE` — is read independently by `train.py` on each rank
from its own environment, so these must be kept in sync in **each node's own** shell profile,
not just the master's.

---

## 8. Preflight checklist

1. `ssh <master> hostname` and `ssh <worker> hostname` from the master — instant, no prompt.
2. `pdsh -w <master>,<worker> hostname` from the master — both hostnames, no rcmd/connection errors.
3. Host firewall inactive on both nodes (or `10.0.0.0/24` trusted).
4. `ip -4 -o addr show | grep 10.0.0` on both nodes — the interface matches `NCCL_SOCKET_IFNAME`.
5. `python3 -c "import torch, deepspeed; print(torch.__version__, deepspeed.__version__)"` — identical on every node.
6. Diff `MODEL_CONFIG`, `BATCH_SIZE`, `TRAIN_STEPS` between nodes' shell profiles — DeepSpeed does not forward these.
7. Check `train_batch_size == micro × grad_accum × (total GPUs across the hostfile)`.
8. If GPU utilization sits at 0% with no error for >30s, suspect network/rendezvous (wrong `MASTER_ADDR`, wrong NIC, firewall) rather than a slow load — real crashes surface within seconds once actually connected.
9. `free -h` on both nodes during an offload run — host RAM has effectively zero margin at full 125B / 8-GPU scale (§5).

---

## 9. DeepSpeed native MoE — why it is not used

`deepspeed.moe`'s expert-parallel all-to-all dispatch **cannot** be combined with ZeRO Stage
3 in the installed DeepSpeed line — the engine asserts `MoE not supported with Stage 3` and
hard-crashes on `deepspeed.initialize()`. Since the 125B model needs ZeRO-3 parameter
sharding to fit in 96 GB/GPU at all, adopting native MoE would mean dropping ZeRO-3 entirely
and redesigning around ZeRO-1/2 plus manual expert-parallel process groups and MoE-aware
optimizer parameter-group splitting — a genuine architecture redesign, not a config change.
The batched grouped-GEMM MoE (which also fixes the Pitfall 10 collective-desync by
construction) is the correct approach for this ZeRO-3 stack.
