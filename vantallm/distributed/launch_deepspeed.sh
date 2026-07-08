#!/bin/bash
# Minimal DeepSpeed Launcher for VantaLLM
# Single-node usage: ./launch_deepspeed.sh <num_gpus> <config_path>
# Multi-node usage:  ./launch_deepspeed.sh <num_gpus> <config_path> <hostfile>
#
# Multi-node requires:
#   - Passwordless SSH from this (master) node to every node in <hostfile>,
#     including itself (ssh <ip> hostname must return with no prompt).
#   - Identical repo + Python env path on every node (no shared FS: sync by hand).
#   - pdsh installed on the master (with the ssh rcmd module; PDSH_RCMD_TYPE=ssh).
#   - Training env vars (MODEL_CONFIG, BATCH_SIZE, TRAIN_STEPS, ...) present in EACH
#     node's own environment — DeepSpeed only auto-forwards NCCL/master-address vars.
#   - A real, directly-routable private network between the hostfile IPs: a shared
#     VLAN/VPC with a DISTINCT on-link IP per node and NO 1:1 NAT between them.
#     NCCL exchanges each rank's own NIC address in-band and CANNOT traverse NAT or
#     reach a duplicated IP; the preflight below fails fast if that is the case.
#
# REQUIRED env vars (export before calling; this script does not set them):
#   export MODEL_CONFIG=configs/vanta_567m.json   # mandatory unless TINY_MODEL=1
#   export TINY_MODEL=1                            # for CPU/tiny smoke tests only
#   export QUANT_MODE=bf16                         # recommended on modern GPUs
# See the README and docs/gpu_cluster_guide.md for the full command sequence.
#
# NETWORK AUTO-CONFIG: in multi-node mode this script discovers MASTER_ADDR and the
# correct NCCL NIC straight from the routing table — no hardcoded IPs or interface
# names — so the same launcher works on any cluster/addressing. Every value stays
# overridable from the environment. Set VANTA_SKIP_NET_PRECHECK=1 to bypass the
# topology sanity check (not recommended).
#
# NOTE: Activation checkpointing is model-level (torch.utils.checkpoint), gated by the
# gradient_checkpointing flag in the model config. It is NOT wired through DeepSpeed's
# activation_checkpointing config block. See transformer.py for the flag.

# Default to ZeRO-2 for a smooth single-GPU out-of-box experience: the 567M
# reference fits an A100-class card under ZeRO-2 without CPU offload. ZeRO-3
# (zero3.json) additionally offloads params to CPU — needed only for the large
# analytical configs, and it can OOM/stall a small single-GPU bring-up.
# Pass an explicit second arg to override, e.g.:
#   ./launch_deepspeed.sh 4 vantallm/distributed/deepspeed_configs/zero3.json
NUM_GPUS=${1:-1}
CONFIG_PATH=${2:-"vantallm/distributed/deepspeed_configs/zero2.json"}
HOSTFILE=${3:-}

# Static, always-overridable defaults.
export MASTER_PORT=${MASTER_PORT:-29500}
export NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-1}
export NCCL_DEBUG=${NCCL_DEBUG:-INFO}

_is_private() {
    # RFC1918 test for an IPv4 dotted-quad.
    case "$1" in
        10.*|192.168.*|172.1[6-9].*|172.2[0-9].*|172.3[0-1].*) return 0 ;;
        *) return 1 ;;
    esac
}

echo "Using config: $CONFIG_PATH"

if [ -n "$HOSTFILE" ]; then
    # ---- Dynamic cluster network auto-configuration --------------------------
    # Discover, from the live routing table, the master address and the NIC that
    # actually reaches the peers. Nothing about the cluster's IPs or NIC names is
    # baked in, so this launcher is portable across any addressing.
    _local_ips=$(ip -4 -o addr show scope global 2>/dev/null | awk '{print $4}' | cut -d/ -f1)
    _hosts=$(grep -vE '^[[:space:]]*(#|$)' "$HOSTFILE" | awk '{print $1}')
    _master=$(printf '%s\n' "$_hosts" | head -n1)

    # First hostfile entry that is not one of our own addresses = a peer.
    _peer=""
    for _h in $_hosts; do
        if ! printf '%s\n' "$_local_ips" | grep -qxF "$_h"; then
            _peer="$_h"; break
        fi
    done

    _auto_if=""; _auto_src=""; _via=""
    if [ -n "$_peer" ]; then
        _route=$(ip -o route get "$_peer" 2>/dev/null)
        _auto_if=$(printf '%s'  "$_route" | sed -n 's/.*[[:space:]]dev[[:space:]]\([^[:space:]]*\).*/\1/p')
        _auto_src=$(printf '%s' "$_route" | sed -n 's/.*[[:space:]]src[[:space:]]\([^[:space:]]*\).*/\1/p')
        printf '%s' "$_route" | grep -q '[[:space:]]via[[:space:]]' && _via=1
        [ -n "$_auto_if" ] && export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-$_auto_if}
    fi
    export MASTER_ADDR=${MASTER_ADDR:-$_master}

    # ---- Network sanity preflight (fail fast, before the long ZeRO-3 init) ----
    # NCCL exchanges each rank's own NIC address in-band; it cannot cross 1:1 NAT
    # or reach a duplicated/self IP. Catch the topologies that otherwise hang at
    # 0% GPU with "connection refused":
    _fatal=""
    #  (a) We would advertise a PRIVATE address to a PUBLIC peer -> behind NAT
    #      (this box's public IP is not on the NIC; it only knows a private addr).
    if [ -n "$_peer" ] && [ -n "$_auto_src" ] && _is_private "$_auto_src" && ! _is_private "$_peer"; then
        _fatal="behind-nat"
    fi
    #  (b) A private peer is only reachable via the default gateway (not on-link)
    #      -> the "private network" is not a real shared subnet.
    if [ -n "$_peer" ] && _is_private "$_peer" && [ -n "$_via" ]; then
        _fatal="no-private-link"
    fi
    #  (c) The peer address is also one of OUR addresses -> duplicate/self IP.
    if [ -n "$_peer" ] && printf '%s\n' "$_local_ips" | grep -qxF "$_peer"; then
        _fatal="duplicate-ip"
    fi

    if [ -n "$_fatal" ] && [ -z "$VANTA_SKIP_NET_PRECHECK" ]; then
        echo "============================================================"
        echo "FATAL ($_fatal): the nodes have no directly-routable private link."
        echo "  local IPs on this node : $(printf '%s ' $_local_ips)"
        echo "  peer (from hostfile)   : ${_peer:-<none>}"
        echo "  addr we would advertise: ${_auto_src:-<none>} via NIC ${_auto_if:-<none>}"
        echo
        echo "NCCL exchanges each rank's own NIC address during setup and cannot"
        echo "cross 1:1 NAT or reach a duplicated/self IP -- peers end up dialing an"
        echo "address that points back at themselves and fail with 'connection"
        echo "refused' (the run then hangs at 0% GPU)."
        echo
        echo "Fix (one-time infra): put all nodes on a shared private VLAN/VPC with"
        echo "DISTINCT, on-link IPs and NO NAT between them (e.g. 10.0.0.2 and"
        echo "10.0.0.3), list those in the hostfile, and relaunch -- this script then"
        echo "auto-detects the right NIC and master address with no further edits."
        echo "Verify: 'ip -4 -o addr show' (IPs differ per node) and"
        echo "        'ip route get <peer-ip>' (direct route, no 'via <gateway>')."
        echo "Override only if you know better: VANTA_SKIP_NET_PRECHECK=1"
        echo "============================================================"
        exit 1
    fi

    echo "Launching DeepSpeed multi-node via hostfile: $HOSTFILE"
    echo "  auto-config: MASTER_ADDR=$MASTER_ADDR NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-<nccl-auto>} MASTER_PORT=$MASTER_PORT"
    # In hostfile mode DeepSpeed derives the process count from slots= in the
    # hostfile; the NUM_GPUS argument is ignored.
    deepspeed \
        --hostfile="$HOSTFILE" \
        --master_addr="$MASTER_ADDR" \
        --master_port="$MASTER_PORT" \
        vantallm/training/train.py \
        --deepspeed \
        --deepspeed_config="$CONFIG_PATH"
else
    export MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
    echo "Launching DeepSpeed single-node with $NUM_GPUS GPUs..."
    deepspeed \
        --num_gpus=$NUM_GPUS \
        vantallm/training/train.py \
        --deepspeed \
        --deepspeed_config="$CONFIG_PATH"
fi
