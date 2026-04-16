#!/bin/bash
# Phase 3 Post-Bugfix — Rerun experiments affected by the TP+PP bug
#
# BUG: configure() had if/elif that made TP dead code when PP was active.
#      Result: "dp=2,pp=2,tp=2" was actually 2 independent dp=2,pp=2 runs.
#      TP groups existed but carried zero traffic.
#
# FIX: Added TensorParallelPipelineStage / TensorParallelT5PipelineStage
#      so PP stages use TP-sharded blocks. Also added TP sync for replicated
#      parameters (embeddings, LayerNorm, lm_head).
#
# This script reruns the 3D hybrid experiments that were affected, plus
# increased batch sizes enabled by TP's memory savings.
#
# Single-GPU baselines are NOT rerun — they are unaffected by this bug.
# Reference baselines from results/phase3/:
#   GPT-2 Small:  3,879 tok/s
#   GPT-2 Medium: 1,404 tok/s
#   T5-base:      3,359 tok/s
#
# Usage:
#   export MASTER_ADDR=<master_private_ip>
#   export WORKER_IP=<worker_private_ip>
#   bash benchmarks/run_phase3_postfix.sh
#
# Results are saved to results/phase3_postfix/

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
RESULTS_DIR="$REPO_DIR/results/phase3_postfix"
LAUNCH="$SCRIPT_DIR/launch_phase3.sh"
LAUNCH_BAD="$SCRIPT_DIR/launch_phase3_bad_placement.sh"

if [ -z "$MASTER_ADDR" ] || [ -z "$WORKER_IP" ]; then
    echo "ERROR: MASTER_ADDR and WORKER_IP must be set."
    echo "  export MASTER_ADDR=<master_private_ip>"
    echo "  export WORKER_IP=<worker_private_ip>"
    exit 1
fi

mkdir -p "$RESULTS_DIR"

echo "============================================="
echo "  Phase 3 Post-Bugfix: True 3D Experiments"
echo "  Results → $RESULTS_DIR"
echo "============================================="

run_exp() {
    local name=$1
    shift
    echo ""
    echo ">>> [$name] Starting..."
    echo ">>> Command: $@"
    "$@" 2>&1 | tee "$RESULTS_DIR/${name}.txt"
    echo ">>> [$name] Done. Saved to $RESULTS_DIR/${name}.txt"
    echo ""
    sleep 5
}

# =============================================
# GROUP B: True 3D — Good Placement (PP inter-node)
#   GPT-2 Medium: dp=2, pp=2, tp=2
#   Default batch + increased batch (enabled by TP memory savings)
# =============================================
echo ""
echo "===== GROUP B: True 3D — Good Placement (GPT-2 Medium) ====="

run_exp "B1_3d_good_gpt2med_bs8" \
    bash "$LAUNCH" --tp_size 2 --pp_size 2 --zero_stage 0 --num_steps 20 --model medium --batch_size 8

run_exp "B2_3d_good_gpt2med_bs16" \
    bash "$LAUNCH" --tp_size 2 --pp_size 2 --zero_stage 0 --num_steps 20 --model medium --batch_size 16

# =============================================
# GROUP C: True 3D — Bad Placement (DP inter-node)
# =============================================
echo ""
echo "===== GROUP C: True 3D — Bad Placement (GPT-2 Medium) ====="

run_exp "C1_3d_bad_gpt2med_bs8" \
    bash "$LAUNCH_BAD" --tp_size 2 --pp_size 2 --zero_stage 0 --num_steps 20 --model medium --batch_size 8

# =============================================
# GROUP D: True 3D — TP inter-node (worst placement)
# =============================================
echo ""
echo "===== GROUP D: True 3D — TP Inter-node (GPT-2 Medium) ====="

run_exp "D1_3d_worst_gpt2med_bs8" \
    bash "$LAUNCH" --tp_size 2 --pp_size 2 --zero_stage 0 --num_steps 20 --model medium --batch_size 8 --worst_placement

# =============================================
# GROUP E: Architecture Comparison — all with good placement
# =============================================
echo ""
echo "===== GROUP E: Architecture Comparison (True 3D, Good Placement) ====="

run_exp "E1_3d_good_gpt2small" \
    bash "$LAUNCH" --tp_size 2 --pp_size 2 --zero_stage 0 --num_steps 20 --model small --batch_size 8

run_exp "E2_3d_good_gpt2med" \
    bash "$LAUNCH" --tp_size 2 --pp_size 2 --zero_stage 0 --num_steps 20 --model medium --batch_size 8

run_exp "E3_3d_good_t5base" \
    bash "$LAUNCH" --tp_size 2 --pp_size 2 --zero_stage 0 --num_steps 20 --model t5_base --batch_size 8

# =============================================
# GROUP F: True 3D + ZeRO-1
# =============================================
echo ""
echo "===== GROUP F: True 3D + ZeRO-1 (GPT-2 Medium) ====="

run_exp "F1_3d_good_gpt2med_zero1" \
    bash "$LAUNCH" --tp_size 2 --pp_size 2 --zero_stage 1 --num_steps 20 --model medium --batch_size 8

# =============================================
echo ""
echo "============================================="
echo "  ALL POST-BUGFIX EXPERIMENTS COMPLETE"
echo "  Results saved in: $RESULTS_DIR/"
echo "============================================="
ls -la "$RESULTS_DIR/"
