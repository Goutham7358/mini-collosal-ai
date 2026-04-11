#!/bin/bash
# Phase 3 Launch Script — 2× g4dn.12xlarge (4 GPUs per node, 8 total)
#
# Usage:
#   bash benchmarks/launch_phase3.sh --tp_size 2 --pp_size 2 --zero_stage 0
#   bash benchmarks/launch_phase3.sh --tp_size 1 --pp_size 1 --zero_stage 0  # DP(8)
#   bash benchmarks/launch_phase3.sh --tp_size 4 --pp_size 1                 # DP(2)xTP(4)
#
# Single-node mode (4 GPUs on 1 node, no inter-node):
#   SINGLE_NODE=1 bash benchmarks/launch_phase3.sh --tp_size 4 --pp_size 1
#
# Environment variables:
#   MASTER_ADDR  — master node private IP (required for 2-node)
#   WORKER_IP    — worker node private IP (required for 2-node)
#   PEM          — path to SSH key
#   SINGLE_NODE  — set to 1 to run on 1 node only (4 GPUs)
#   MASTER_PORT  — port for rendezvous (default 29500)

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"

PEM=${PEM:-/home/ubuntu/workspace/gpooloth-experiment-DPDKML.pem}
MASTER=${MASTER_ADDR:-$(hostname -I | awk '{print $1}')}
WORKER=${WORKER_IP:-""}
PORT=${MASTER_PORT:-29500}
SINGLE_NODE=${SINGLE_NODE:-0}

SCRIPT_ARGS="$@"

if [ "$SINGLE_NODE" = "1" ]; then
    echo "=== Phase 3 Benchmark: SINGLE NODE (4 GPUs, all PCIe) ==="
    echo "  Args: $SCRIPT_ARGS"

    # Kill old processes
    pkill -9 -f bench_unified 2>/dev/null || true
    sleep 1

    NCCL_DEBUG=ERROR \
    PATH=$HOME/.local/bin:$PATH \
    torchrun --standalone --nproc_per_node=4 \
        "$REPO_DIR/benchmarks/bench_unified.py" $SCRIPT_ARGS

    echo "=== Done ==="
    exit 0
fi

# --- Two-node mode ---
if [ -z "$WORKER_IP" ]; then
    echo "ERROR: Set WORKER_IP for 2-node mode, or use SINGLE_NODE=1 for single-node."
    exit 1
fi

echo "=== Phase 3 Benchmark: 2 NODES × 4 GPUs (8 total) ==="
echo "  Master: $MASTER  Worker: $WORKER"
echo "  Args: $SCRIPT_ARGS"

# Kill old processes on both nodes
ssh -o StrictHostKeyChecking=no -i $PEM ubuntu@$WORKER "pkill -9 -f bench_unified 2>/dev/null" &
pkill -9 -f bench_unified 2>/dev/null || true
wait
sleep 1

# Copy latest code to worker
rsync -az -e "ssh -i $PEM" "$REPO_DIR/minicolossal/" ubuntu@$WORKER:/home/ubuntu/workspace/mini-colossal-ai/minicolossal/
rsync -az -e "ssh -i $PEM" "$REPO_DIR/benchmarks/" ubuntu@$WORKER:/home/ubuntu/workspace/mini-colossal-ai/benchmarks/

# Detect network interface (may differ from Phase 2's ens5)
IFACE=$(ip route get 1 | grep -oP 'dev \K\S+' | head -1)
echo "  Network interface: $IFACE"

# Launch worker (node_rank=1)
echo "Starting worker (node_rank=1) on $WORKER"
ssh -o StrictHostKeyChecking=no -i $PEM ubuntu@$WORKER \
    "cd /home/ubuntu/workspace/mini-colossal-ai && \
     NCCL_SOCKET_IFNAME=$IFACE NCCL_DEBUG=ERROR \
     PATH=\$HOME/.local/bin:\$PATH \
     torchrun --nnodes=2 --nproc_per_node=4 --node_rank=1 \
     --master_addr=$MASTER --master_port=$PORT \
     benchmarks/bench_unified.py $SCRIPT_ARGS" 2>&1 | sed "s/^/[worker] /" &

sleep 3

# Launch master (node_rank=0)
echo "Starting master (node_rank=0)"
PATH=$HOME/.local/bin:$PATH \
NCCL_SOCKET_IFNAME=$IFACE NCCL_DEBUG=ERROR \
torchrun --nnodes=2 --nproc_per_node=4 --node_rank=0 \
    --master_addr=$MASTER --master_port=$PORT \
    "$REPO_DIR/benchmarks/bench_unified.py" $SCRIPT_ARGS

wait
echo "=== Done ==="
