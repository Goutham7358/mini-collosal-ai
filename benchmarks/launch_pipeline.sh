#!/bin/bash
# Launch pipeline parallelism benchmark across nodes
# Usage: bash benchmarks/launch_pipeline.sh <num_nodes> <schedule>
#   schedule: naive or 1f1b

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"

NUM_NODES=${1:-4}
SCHEDULE=${2:-1f1b}
PEM=${PEM:-/home/ubuntu/workspace/gpooloth-experiment-DPDKML.pem}
MASTER=${MASTER_ADDR:-10.0.3.175}
PORT=29630
SCRIPT=/home/ubuntu/benchmarks/bench_pipeline.py
ALL_IPS=(${WORKER_IPS:-10.0.3.181 10.0.3.115 10.0.3.34})

echo "=== Pipeline Parallelism Benchmark: ${NUM_NODES} nodes, schedule=${SCHEDULE} ==="

# Kill old processes
for IP in "${ALL_IPS[@]}"; do
    ssh -o StrictHostKeyChecking=no -i $PEM ubuntu@$IP "pkill -9 -f bench_pipeline 2>/dev/null" &
done
pkill -9 -f bench_pipeline 2>/dev/null
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
        "NCCL_SOCKET_IFNAME=ens5 NCCL_DEBUG=ERROR PP_SCHEDULE=$SCHEDULE \
         PATH=\$HOME/.local/bin:\$PATH \
         torchrun --nnodes=$NUM_NODES --nproc_per_node=1 --node_rank=$RANK \
         --master_addr=$MASTER --master_port=$PORT $SCRIPT" 2>&1 | sed "s/^/[node$RANK] /" &
done

sleep 2
echo "Starting master (rank 0)"
NCCL_SOCKET_IFNAME=ens5 NCCL_DEBUG=ERROR PP_SCHEDULE=$SCHEDULE \
    PATH=$HOME/.local/bin:$PATH \
    torchrun --nnodes=$NUM_NODES --nproc_per_node=1 --node_rank=0 \
    --master_addr=$MASTER --master_port=$PORT \
    "$REPO_DIR/benchmarks/bench_pipeline.py" 2>&1

wait
echo "=== Done ==="
