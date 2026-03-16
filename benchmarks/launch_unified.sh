#!/bin/bash
# Launch unified 3D parallelism benchmark on 4 nodes
#
# Usage:
#   bash benchmarks/launch_unified.sh --tp_size 2 --pp_size 1 --zero_stage 0
#   bash benchmarks/launch_unified.sh --tp_size 1 --pp_size 2 --zero_stage 1
#   bash benchmarks/launch_unified.sh --tp_size 1 --pp_size 1 --zero_stage 0

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"

NUM_NODES=4
PEM=${PEM:-/home/ubuntu/workspace/gpooloth-experiment-DPDKML.pem}
MASTER=${MASTER_ADDR:-10.0.3.175}
PORT=29680
SCRIPT=/home/ubuntu/benchmarks/bench_unified.py
ALL_IPS=(${WORKER_IPS:-10.0.3.181 10.0.3.115 10.0.3.34})

# Pass all script arguments through
SCRIPT_ARGS="$@"

echo "=== Unified 3D Benchmark: 4 nodes ==="
echo "  Args: $SCRIPT_ARGS"

# Kill old processes
for IP in "${ALL_IPS[@]}"; do
    ssh -o StrictHostKeyChecking=no -i $PEM ubuntu@$IP "pkill -9 -f bench_unified 2>/dev/null" &
done
pkill -9 -f bench_unified 2>/dev/null
wait
sleep 1

# Copy latest code
for IP in "${ALL_IPS[@]}"; do
    rsync -az -e "ssh -i $PEM" "$REPO_DIR/minicolossal/" ubuntu@$IP:/home/ubuntu/minicolossal/ &
    rsync -az -e "ssh -i $PEM" "$REPO_DIR/benchmarks/" ubuntu@$IP:/home/ubuntu/benchmarks/ &
done
wait

# Launch remote nodes
for i in $(seq 0 2); do
    RANK=$((i + 1))
    IP=${ALL_IPS[$i]}
    echo "Starting rank $RANK on $IP"
    ssh -o StrictHostKeyChecking=no -i $PEM ubuntu@$IP \
        "NCCL_SOCKET_IFNAME=ens5 NCCL_DEBUG=ERROR \
         PATH=\$HOME/.local/bin:\$PATH \
         torchrun --nnodes=$NUM_NODES --nproc_per_node=1 --node_rank=$RANK \
         --master_addr=$MASTER --master_port=$PORT $SCRIPT $SCRIPT_ARGS" 2>&1 | sed "s/^/[node$RANK] /" &
done

sleep 2
echo "Starting master (rank 0)"
NCCL_SOCKET_IFNAME=ens5 NCCL_DEBUG=ERROR \
    PATH=$HOME/.local/bin:$PATH \
    torchrun --nnodes=$NUM_NODES --nproc_per_node=1 --node_rank=0 \
    --master_addr=$MASTER --master_port=$PORT \
    "$REPO_DIR/benchmarks/bench_unified.py" $SCRIPT_ARGS 2>&1

wait
echo "=== Done ==="
