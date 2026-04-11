#!/bin/bash
# Phase 3 — Run ALL experiments and save results
#
# Usage:
#   bash benchmarks/run_all_phase3.sh
#
# Prerequisites:
#   - Set MASTER_ADDR and WORKER_IP environment variables
#   - Set PEM to SSH key path
#   - Both nodes have the code at /home/ubuntu/workspace/mini-colossal-ai/
#   - Both nodes have PyTorch, tiktoken, datasets installed
#
# Results are saved to results/phase3/

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
RESULTS_DIR="$REPO_DIR/results/phase3"
LAUNCH="$SCRIPT_DIR/launch_phase3.sh"
LAUNCH_BAD="$SCRIPT_DIR/launch_phase3_bad_placement.sh"

# Verify required env vars for 2-node experiments
if [ -z "$MASTER_ADDR" ] || [ -z "$WORKER_IP" ]; then
    echo "WARNING: MASTER_ADDR and WORKER_IP not set. Groups 2-3 (multi-node) will fail."
    echo "  Set them with: export MASTER_ADDR=<master_private_ip> WORKER_IP=<worker_private_ip>"
fi

mkdir -p "$RESULTS_DIR"

echo "============================================="
echo "  Phase 3: All Experiments"
echo "  Results → $RESULTS_DIR"
echo "============================================="

# Helper: run and save output
run_exp() {
    local name=$1
    shift
    echo ""
    echo ">>> [$name] Starting..."
    echo ">>> Command: $@"
    "$@" 2>&1 | tee "$RESULTS_DIR/${name}.txt"
    echo ">>> [$name] Done. Saved to $RESULTS_DIR/${name}.txt"
    echo ""
    sleep 5  # cooldown between experiments
}

# =============================================
# GROUP 1: Single-Node Baselines (4 GPUs, PCIe)
# =============================================
echo ""
echo "===== GROUP 1: Single-Node PCIe Baselines (4 GPUs, 1 node) ====="

run_exp "1a_single_gpu" \
    python "$REPO_DIR/benchmarks/bench_single_gpu.py"

# Group 1 uses single-node mode (all PCIe, no inter-node TCP)
export SINGLE_NODE=1

run_exp "1b_dp4_pcie" \
    bash "$LAUNCH" --tp_size 1 --pp_size 1 --zero_stage 0 --num_steps 30

run_exp "1c_tp4_pcie" \
    bash "$LAUNCH" --tp_size 4 --pp_size 1 --zero_stage 0 --num_steps 30

run_exp "1d_pp4_1f1b_pcie" \
    bash "$LAUNCH" --tp_size 1 --pp_size 4 --zero_stage 0 --num_steps 20

run_exp "1e_zero1_pcie" \
    bash "$LAUNCH" --tp_size 1 --pp_size 1 --zero_stage 1 --num_steps 30

unset SINGLE_NODE

# =============================================
# GROUP 2: Full 3D Mesh — Good Placement (8 GPUs)
# =============================================
echo ""
echo "===== GROUP 2: Full 3D Mesh (TP intra-node) ====="

run_exp "2a_dp2_pp2_tp2_good" \
    bash "$LAUNCH" --tp_size 2 --pp_size 2 --zero_stage 0 --num_steps 20

run_exp "2b_dp2_pp2_tp2_zero1_good" \
    bash "$LAUNCH" --tp_size 2 --pp_size 2 --zero_stage 1 --num_steps 20

# =============================================
# GROUP 3: Bad Placement — TP inter-node (8 GPUs)
# =============================================
echo ""
echo "===== GROUP 3: Bad Placement (TP inter-node) ====="

run_exp "3a_dp2_pp2_tp2_bad" \
    bash "$LAUNCH_BAD" --tp_size 2 --pp_size 2 --zero_stage 0 --num_steps 20

# =============================================
echo ""
echo "============================================="
echo "  ALL EXPERIMENTS COMPLETE"
echo "  Results saved in: $RESULTS_DIR/"
echo "============================================="
ls -la "$RESULTS_DIR/"
