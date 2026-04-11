# Phase 3 Experiment Plan — Communication-Aware Placement

## Hardware
- **2× g4dn.12xlarge** instances (4× Tesla T4 16GB each)
- 8 GPUs total: 4 intra-node (PCIe ~15 GB/s) + inter-node (TCP ~0.6 GB/s)
- Same T4 GPU as Phase 2 → results are directly comparable

## Setup Checklist
1. Spin up 2× g4dn.12xlarge in the same VPC/subnet
2. Note their private IPs: `NODE0_IP` (master) and `NODE1_IP` (worker)
3. Ensure SSH access between nodes with PEM key
4. Copy `/home/ubuntu/workspace` contents to both nodes
5. Install same PyTorch version: `pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cu121`
6. Install dependencies: `pip install tiktoken datasets minted`
7. Verify GPUs: `nvidia-smi` should show 4× T4 on each node
8. Verify NCCL sees all local GPUs: 
   ```
   python -c "import torch; print(torch.cuda.device_count())"  # should print 4
   ```
9. Cache dataset on both nodes:
   ```
   python -c "from datasets import load_dataset; load_dataset('wikitext', 'wikitext-2-raw-v1')"
   ```

## Key Difference from Phase 2
- Phase 2: `--nnodes=4 --nproc_per_node=1` (4 nodes, 1 GPU each, all TCP)
- Phase 3: `--nnodes=2 --nproc_per_node=4` (2 nodes, 4 GPUs each, PCIe intra + TCP inter)
- `LOCAL_RANK` (0-3) selects which GPU on the node
- `RANK` (0-7) is the global rank
- NCCL auto-detects PCIe for intra-node, TCP for inter-node

## How Rank Assignment Controls Placement
With `torchrun --nnodes=2 --nproc_per_node=4`:
- Node 0 gets global ranks 0,1,2,3 (local_rank 0,1,2,3)
- Node 1 gets global ranks 4,5,6,7 (local_rank 0,1,2,3)

The ProcessGroupMesh maps ranks to coordinates in row-major order.
For a mesh shape (dp=2, pp=2, tp=2):
```
Rank 0 → (dp=0, pp=0, tp=0)    Rank 4 → (dp=1, pp=0, tp=0)
Rank 1 → (dp=0, pp=0, tp=1)    Rank 5 → (dp=1, pp=0, tp=1)
Rank 2 → (dp=0, pp=1, tp=0)    Rank 6 → (dp=1, pp=1, tp=0)
Rank 3 → (dp=0, pp=1, tp=1)    Rank 7 → (dp=1, pp=1, tp=1)
```

**Default torchrun mapping** (ranks 0-3 on node 0, 4-7 on node 1):
- TP groups: {0,1}, {2,3}, {4,5}, {6,7} → **TP is INTRA-node (PCIe)** ✓
- DP groups: {0,4}, {1,5}, {2,6}, {3,7} → **DP is INTER-node (TCP)** ✓
- PP groups: {0,2}, {1,3}, {4,6}, {5,7} → **PP is INTRA-node (PCIe)**

This is **Placement A (communication-aware)** — TP on fast link, DP on slow link.

For **Placement B (anti-pattern)**, we swap rank assignment so TP goes inter-node.
We do this by reordering ranks across nodes using environment variables.

## New Launch Script
Use `benchmarks/launch_phase3.sh` (to be created). It replaces the Phase 2 
multi-node SSH launch scripts with a 2-node, 4-GPU-per-node launcher.

---

## Experiments to Run

### GROUP 1: Single-Node Baselines (PCIe, 4 GPUs on 1 node)
Compare directly with Phase 2 all-TCP results on same T4 hardware.
Run on a single g4dn.12xlarge (no inter-node communication).

```bash
# 1a. Single GPU baseline (sanity check — should match Phase 2: ~1,361 tok/s)
python benchmarks/bench_single_gpu.py

# 1b. DP 4-way on PCIe (Phase 2 got 1,393 tok/s over TCP)
torchrun --standalone --nproc_per_node=4 benchmarks/bench_unified.py \
  --tp_size 1 --pp_size 1 --zero_stage 0 --num_steps 30

# 1c. TP 4-way on PCIe (Phase 2 got 959 tok/s over TCP)
torchrun --standalone --nproc_per_node=4 benchmarks/bench_unified.py \
  --tp_size 4 --pp_size 1 --zero_stage 0 --num_steps 30

# 1d. PP 4-way 1F1B on PCIe (Phase 2 got 3,070 tok/s over TCP)
torchrun --standalone --nproc_per_node=4 benchmarks/bench_unified.py \
  --tp_size 1 --pp_size 4 --zero_stage 0 --num_steps 20

# 1e. ZeRO-1 on PCIe (Phase 2 got 934 tok/s over TCP)
torchrun --standalone --nproc_per_node=4 benchmarks/bench_unified.py \
  --tp_size 1 --pp_size 1 --zero_stage 1 --num_steps 30

# 1f. ZeRO-2 on PCIe (Phase 2 got 915 tok/s over TCP)
# (Use bench_zero.py with ZERO_STAGE=2 since unified plugin only supports ZeRO-1 with PP)
ZERO_STAGE=2 torchrun --standalone --nproc_per_node=4 benchmarks/bench_zero.py
```

**Expected outcome**: DP and TP should show MUCH higher throughput than Phase 2 
because PCIe is ~25× faster than TCP. PP may not change much since it was 
already fast (small point-to-point messages).

### GROUP 2: Full 3D Mesh — Communication-Aware Placement (dp=2, pp=2, tp=2)
8 GPUs across 2 nodes. Default rank mapping puts TP intra-node (good).

```bash
# 2a. Full 3D: dp=2, pp=2, tp=2 — TP on PCIe, DP on TCP (GOOD placement)
# On BOTH nodes, run:
# Node 0 (master):
NCCL_SOCKET_IFNAME=ens5 torchrun \
  --nnodes=2 --nproc_per_node=4 --node_rank=0 \
  --master_addr=$MASTER_IP --master_port=29500 \
  benchmarks/bench_unified.py --tp_size 2 --pp_size 2 --zero_stage 0 --num_steps 20
# Node 1 (worker):
NCCL_SOCKET_IFNAME=ens5 torchrun \
  --nnodes=2 --nproc_per_node=4 --node_rank=1 \
  --master_addr=$MASTER_IP --master_port=29500 \
  benchmarks/bench_unified.py --tp_size 2 --pp_size 2 --zero_stage 0 --num_steps 20

# 2b. Full 3D + ZeRO-1: dp=2, pp=2, tp=2, zero=1
# Same as 2a but add --zero_stage 1
```

### GROUP 3: Communication-Aware vs Anti-Pattern Placement
Same (dp=2, pp=2, tp=2) config but with TP forced inter-node.

To force **bad placement** (TP inter-node), we remap ranks so that
TP-adjacent ranks land on different nodes. This requires a custom
rank remapping — see "Rank Remapping for Bad Placement" section below.

```bash
# 3a. BAD placement: TP on TCP, DP on PCIe
# Uses rank remapping so ranks 0,2,4,6 go to node 0 and 1,3,5,7 to node 1
# This makes TP groups {0,1},{2,3},{4,5},{6,7} span ACROSS nodes
# See launch_phase3_bad_placement.sh

# 3b. ALL-TCP baseline (simulates Phase 2 — 1 GPU per node equivalent)
# Run with --nproc_per_node=1 on each of 8 "virtual" nodes, or
# just use 4 GPUs across 2 nodes with 2-per-node to get some TCP:
NCCL_SOCKET_IFNAME=ens5 torchrun \
  --nnodes=2 --nproc_per_node=1 --node_rank=0 \
  --master_addr=$MASTER_IP --master_port=29500 \
  benchmarks/bench_unified.py --tp_size 2 --pp_size 1 --zero_stage 0 --num_steps 20
# (This puts TP across nodes on TCP — same as Phase 2 DP×TP config)
```

---

## How Bad Placement Works (Group 3)

We added a `--bad_placement` flag to `bench_unified.py` and `MiniColossalPlugin`.
When enabled, the plugin swaps the DP and TP axes in the mesh:

**Normal mesh** `(dp, pp, tp)`: TP is the last axis (fastest-changing).
With torchrun assigning ranks 0-3 to node 0 and 4-7 to node 1:
- TP groups: {0,1}, {2,3}, {4,5}, {6,7} → all **INTRA-node** (PCIe) ✓

**Bad placement mesh** `(tp, pp, dp)`: TP is the first axis (slowest-changing).
Same rank-to-node mapping, but now:
- TP groups: {0,4}, {1,5}, {2,6}, {3,7} → all **INTER-node** (TCP) ✗
- DP groups: {0,1}, {2,3}, {4,5}, {6,7} → all **INTRA-node** (PCIe)

No rank remapping needed — same `torchrun` command, just `--bad_placement`.

---

## Summary Table of All Experiments

| # | Group | Config | GPUs | Nodes | Placement | Purpose |
|---|-------|--------|------|-------|-----------|---------|
| 1a | Baseline | Single GPU | 1 | 1 | — | Sanity check |
| 1b | PCIe | DP(4) | 4 | 1 | All PCIe | Compare vs Phase 2 TCP |
| 1c | PCIe | TP(4) | 4 | 1 | All PCIe | Compare vs Phase 2 TCP |
| 1d | PCIe | PP(4) 1F1B | 4 | 1 | All PCIe | Compare vs Phase 2 TCP |
| 1e | PCIe | ZeRO-1 | 4 | 1 | All PCIe | Compare vs Phase 2 TCP |
| 1f | PCIe | ZeRO-2 | 4 | 1 | All PCIe | Compare vs Phase 2 TCP |
| 2a | 3D | dp2×pp2×tp2 | 8 | 2 | TP intra ✓ | Best 3D config |
| 2b | 3D | dp2×pp2×tp2+Z1 | 8 | 2 | TP intra ✓ | 3D + ZeRO |
| 3a | Anti | dp2×pp2×tp2 | 8 | 2 | TP inter ✗ | Bad placement |
| 3b | TCP | DP(1)×TP(2) | 2 | 2 | TP inter | Phase 2 comparison |

---

## GROUP 5: Multi-Model Experiments (Faculty Feedback)

Faculty feedback: *"Experiments with more models showing similar trend should be done."*
*"If the embedding size changes, the results might change. Higher embedding size may exceed memory block size."*

### Models to Benchmark

| Model | Type | Params | Hidden | Layers | Heads | Key Difference from GPT-2 Medium |
|---|---|---|---|---|---|---|
| GPT-2 Small | Decoder-only | 117M | 768 | 12 | 12 | Smaller embeddings → lighter communication, fits easily on 1 GPU |
| GPT-2 Medium | Decoder-only | 354M | 1024 | 24 | 16 | *(Phase 2 baseline)* |
| GPT-2 XL | Decoder-only | 1.5B | 1600 | 48 | 25 | Won't fit on single T4 → forces PP/ZeRO. TP all-reduce 2.4× larger |
| T5-base | Encoder-Decoder | 220M | 768 | 12+12 | 12 | Cross-attention adds 50% more TP comm on decoder. PP stages imbalanced |

### Why T5 Shows Maximum Difference

GPT-2's uniform layer structure is an ideal case for hybrid parallelism. T5 breaks
this in three ways:

1. **TP overhead increases**: T5 decoder has cross-attention → 3 all-reduces per
   block vs GPT-2's 2. For dp=2,pp=2,tp=2 with 12 blocks per stage:
   GPT-2: 24 all-reduces/stage (uniform). T5 decoder stage: 36 all-reduces.

2. **PP becomes imbalanced**: Encoder layers (no cross-attn) are cheaper than
   decoder layers (have cross-attn). The fast encoder stage idles waiting for
   the slow decoder stage, increasing effective pipeline bubble.

3. **DP gradient sync varies**: Decoder stages have more parameters (cross-attn
   weights) than encoder stages. DP all-reduce is bounded by the slowest stage.

### Expected Results

Same hardware, same hybrid config (dp=2, pp=2, tp=2), different speedup:

| Model | Baseline (1 GPU) | Hybrid (8 GPU) | Expected Speedup |
|---|---|---|---|
| GPT-2 Small | ~1,800 tok/s | ~6,000 tok/s | ~3.3× (ideal — small comm, balanced) |
| GPT-2 Medium | ~1,361 tok/s | ~4,500 tok/s | ~3.3× (balanced layers) |
| GPT-2 XL | OOM | ~1,200 tok/s | ∞ (can't run without parallelism) |
| T5-base | ~1,300 tok/s | ~3,200 tok/s | ~2.5× (imbalanced stages, extra TP comm) |

Key observations to verify:
- GPT-2 Small: TP overhead is relatively smaller → higher scaling efficiency
- GPT-2 XL: **Requires** PP or ZeRO to train at all on T4 16GB
- T5-base: Hybrid speedup is lower than GPT-2 despite similar param count,
  because architectural asymmetry degrades pipeline balance and increases TP overhead

### Experiments to Run (Per Model)

For each model, run:
1. Single GPU baseline
2. DP(4) on PCIe (single node)
3. TP(4) on PCIe (single node)  [skip for GPT-2 XL if OOM with tp=4]
4. PP(4) 1F1B on PCIe (single node)
5. Hybrid dp=2, pp=2, tp=2 (2 nodes, good placement)
6. Hybrid dp=2, pp=2, tp=2 (2 nodes, bad placement) — for GPT-2 Medium and T5-base only

### Code Changes Needed for Multi-Model
1. **`minicolossal/gpt2.py`**: Add `GPT2Config.small()` and `GPT2Config.xl()` class methods
2. **`minicolossal/t5.py`**: New file — T5Config + T5Model with encoder-decoder
   architecture and cross-attention. Reuse TransformerBlock, add CrossAttention sublayer.
3. **`minicolossal/tensor_parallel.py`**: Extend TP to handle cross-attention layer
4. **`minicolossal/pipeline_parallel.py`**: Handle asymmetric encoder/decoder split
5. **`benchmarks/bench_unified.py`**: Add `--model` flag (gpt2_small, gpt2_medium,
   gpt2_xl, t5_base) to select model
6. DP and ZeRO code need **zero changes** — they are model-agnostic

---

## Code Changes Made
1. **`benchmarks/launch_phase3.sh`** — 2-node, 4-GPU-per-node launcher. 
   Supports `SINGLE_NODE=1` for intra-node-only experiments (Group 1).
2. **`benchmarks/launch_phase3_bad_placement.sh`** — Thin wrapper that calls 
   `launch_phase3.sh` with `--bad_placement` appended.
3. **`benchmarks/bench_unified.py`** — Added `--bad_placement` CLI flag, 
   passed through to the plugin.
4. **`minicolossal/plugin.py`** — Added `bad_placement` parameter to 
   `MiniColossalPlugin.__init__()`. When True, swaps DP/TP axes in the mesh 
   so TP groups span nodes instead of staying intra-node.
5. **`benchmarks/run_all_phase3.sh`** — Master script that runs all 16 
   experiments sequentially and saves results to `results/phase3/`.
6. **NCCL_SOCKET_IFNAME**: Auto-detected from `ip route` in the launch script. 
   May need manual override if the g4dn.12xlarge has a different interface name.
