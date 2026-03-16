#!/bin/bash
# Launch data parallelism benchmark across nodes
# Usage: bash benchmarks/launch_dp.sh <num_nodes> <method>
#   num_nodes: 2 or 4
#   method: naive, ring, or ring_bucketed

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"

NUM_NODES=${1:-4}
METHOD=${2:-ring_bucketed}
PEM=${PEM:-/home/ubuntu/workspace/gpooloth-experiment-DPDKML.pem}
MASTER=${MASTER_ADDR:-10.0.3.175}
PORT=29600
SCRIPT=/home/ubuntu/benchmarks/bench_data_parallel.py
ALL_IPS=(${WORKER_IPS:-10.0.3.181 10.0.3.115 10.0.3.34})

echo "=== Data Parallelism Benchmark: ${NUM_NODES} nodes, method=${METHOD} ==="

# Kill old processes
for IP in "${ALL_IPS[@]}"; do
    ssh -o StrictHostKeyChecking=no -i $PEM ubuntu@$IP "pkill -9 -f bench_data_parallel 2>/dev/null" &
done
pkill -9 -f bench_data_parallel 2>/dev/null
wait
sleep 1

# Copy latest code to all nodes
for IP in "${ALL_IPS[@]}"; do
    rsync -az -e "ssh -i $PEM" "$REPO_DIR/minicolossal/" ubuntu@$IP:/home/ubuntu/minicolossal/ &
    rsync -az -e "ssh -i $PEM" "$REPO_DIR/benchmarks/" ubuntu@$IP:/home/ubuntu/benchmarks/ &
done
wait

# Launch remote nodes
NODES_TO_USE=$((NUM_NODES - 1))
for i in $(seq 0 $((NODES_TO_USE - 1))); do
    RANK=$((i + 1))
    IP=${ALL_IPS[$i]}
    echo "Starting rank $RANK on $IP"
    ssh -o StrictHostKeyChecking=no -i $PEM ubuntu@$IP \
        "NCCL_SOCKET_IFNAME=ens5 NCCL_DEBUG=ERROR DP_METHOD=$METHOD \
         PATH=\$HOME/.local/bin:\$PATH \
         torchrun --nnodes=$NUM_NODES --nproc_per_node=1 --node_rank=$RANK \
         --master_addr=$MASTER --master_port=$PORT $SCRIPT" 2>&1 | sed "s/^/[node$RANK] /" &
done

sleep 2
echo "Starting master (rank 0)"
NCCL_SOCKET_IFNAME=ens5 NCCL_DEBUG=ERROR DP_METHOD=$METHOD \
    PATH=$HOME/.local/bin:$PATH \
    torchrun --nnodes=$NUM_NODES --nproc_per_node=1 --node_rank=0 \
    --master_addr=$MASTER --master_port=$PORT \
    "$REPO_DIR/benchmarks/bench_data_parallel.py" 2>&1

wait
echo "=== Done ==="
