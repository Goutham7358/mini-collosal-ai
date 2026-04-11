#!/bin/bash
# Phase 3 BAD Placement Launch — TP forced inter-node (anti-pattern)
#
# Wraps launch_phase3.sh but adds --bad_placement flag.
# The plugin swaps the DP/TP axes in the mesh so that:
#   - TP groups span ACROSS nodes (slow TCP) instead of intra-node
#   - DP groups stay WITHIN a node (fast PCIe) instead of inter-node
#
# Compare throughput of this vs launch_phase3.sh (good placement) to
# demonstrate the impact of communication-aware placement.
#
# Usage:
#   bash benchmarks/launch_phase3_bad_placement.sh --tp_size 2 --pp_size 2 --zero_stage 0
#
# Environment variables (same as launch_phase3.sh):
#   MASTER_ADDR, WORKER_IP, PEM, MASTER_PORT

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "=== Phase 3 BAD Placement: TP forced INTER-node ==="
echo "  Adding --bad_placement flag to launch_phase3.sh"

exec bash "$SCRIPT_DIR/launch_phase3.sh" "$@" --bad_placement
