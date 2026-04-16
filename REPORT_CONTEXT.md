# Report Context for Phase 3

## Task

Create `report_phase3.tex` following the same structure as `report_phase2.tex` (IEEEtran conference format). Must NOT exceed **4 pages** when compiled. The bibliography file is `mini-colossal-ai/references.bib` (relative to the report). A T5 citation key `raffel2020exploring` has been added to references.bib.

## Previous Report Feedback

- **Phase 1** feedback: "Proposal lacks clear goal. What are the metrics will you be considering? The volume of work proposed is less."
- **Phase 2** feedback: "Good report. Good set of results. Experiments with more models showing similar trend should be done."

Phase 3 addresses the Phase 2 feedback by adding T5-base (encoder-decoder) alongside GPT-2 (decoder-only), and moving to a heterogeneous-interconnect cluster.

## report_phase2.tex Structure (follow this)

1. **Abstract** — one paragraph summary
2. **Keywords**
3. **Introduction** — motivation, what we built, what's new
4. **Related Work** — brief lit review (DP, ZeRO, TP, PP, Colossal-AI)
5. **Methodology** — subsections for each technique + unified plugin
6. **Results and Discussion** — subsections per experiment group, tables
7. **Conclusion** — summary + what's next

Authors (same as Phase 2):
- Dinesh Chand (dineshchand@iisc.ac.in)
- Goutham P (gouthamp@iisc.ac.in)
- Meena Chidambaram (meenac@iisc.ac.in)
- Nadhiya R S (nadhiyar@iisc.ac.in)
All MTech-AI, IISc-Bangalore.

## What's New in Phase 3 (vs Phase 2)

1. **Hardware upgrade**: 2× g4dn.12xlarge (4× T4 GPUs each, 8 GPUs total). Intra-node PCIe ~15 GB/s, inter-node TCP ~0.6 GB/s (25× gap). Phase 2 was 4× g4dn.xlarge (1 GPU each, all-TCP).

2. **True 3D hybrid parallelism**: dp=2, pp=2, tp=2 on 8 GPUs. Phase 2 was limited to 2 axes at a time on 4 GPUs.

3. **Communication-aware placement**: Which parallelism axis goes on which network tier. Three placements tested (good, bad, worst).

4. **T5-base model**: Encoder-decoder architecture (12 enc + 12 dec layers, 768H, 12 heads, 237M params). Implemented from scratch with TP and PP support.

5. **Bug discovery and fix**: The original plugin had an if/elif that made TP dead code when PP was active. Fixed by adding `TensorParallelPipelineStage`. Results are reported for BOTH pre-fix and post-fix.

## Hardware

```
2× AWS g4dn.12xlarge
  4× NVIDIA Tesla T4 (16 GB) per node = 8 GPUs total
  Intra-node: PCIe Gen3 ~15 GB/s
  Inter-node: TCP over VPC ~0.6 GB/s
  Bandwidth gap: ~25×
```

## Models

| Model | Architecture | Layers | Hidden | Heads | Params |
|-------|-------------|--------|--------|-------|--------|
| GPT-2 Small | Decoder-only | 12 | 768 | 12 | 117M |
| GPT-2 Medium | Decoder-only | 24 | 1024 | 16 | 354M |
| GPT-2 Large | Decoder-only | 36 | 1280 | 20 | 774M |
| T5-base | Encoder-decoder | 12+12 | 768 | 12 | 237M |

## Single-GPU Max Batch Sizes

| Model | bs=4 | bs=8 | bs=16 | bs=32 | Max bs |
|-------|------|------|-------|-------|--------|
| GPT-2 Small | 3,619 (3.3G) | 4,147 (5.1G) | 4,426 (8.7G) | OOM | 16 |
| GPT-2 Medium | 1,404 (7.7G) | 1,561 (11.1G) | OOM | — | 8 |
| T5-base | 3,125 (4.9G) | 3,853 (5.7G) | 4,441 (8.6G) | 4,747 (14.3G) | 32 |

## The TP+PP Bug

**Bug**: `plugin.py configure()` used `if/elif` for PP and TP, making them mutually exclusive:
```python
# BROKEN:
if self.pp_size > 1:
    model = create_pipeline_stage(...)      # uses plain TransformerBlock
elif self.tp_size > 1:
    model = TensorParallelGPT2(...)         # NEVER reached when pp > 1
```
When running dp=2, pp=2, tp=2, only PP was active. TP groups existed in the mesh but carried zero traffic. The system ran two independent DP(2)×PP(2) training runs, with 4 GPUs duplicating work.

**Fix**: Added `TensorParallelPipelineStage` (GPT-2) and `TensorParallelT5PipelineStage` (T5) that use TP-sharded blocks inside pipeline stages. Fixed `configure()` to create these when both PP and TP are active. Added TP sync for replicated parameters (embeddings, LayerNorm, lm_head).

## Placement Strategies

With 8 GPUs and dp=2, pp=2, tp=2, the plugin creates a (2,2,2) 3D process group mesh. The first axis maps to the slowest (inter-node) link.

| Name | Mesh layout | Inter-node axis | Description |
|------|------------|-----------------|-------------|
| Good | (pp, dp, tp) | PP | PP inter-node, DP+TP intra-node |
| Bad | (dp, pp, tp) | DP | DP inter-node, PP+TP intra-node |
| Worst | (tp, pp, dp) | TP | TP inter-node, PP+DP intra-node |

**Verified by rank assignment inspection**: In D1 (worst), all TP partners (e.g., Rank 0 tp=0 on Node 1 ↔ Rank 4 tp=1 on Node 2) communicate inter-node. DP and PP partners are intra-node. The code, flags, and actual rank logs are all consistent.

---

## RESULTS — PRE-FIX (TP was dead code in 3D configs)

These results have the bug: TP groups exist but carry zero traffic. Effectively DP(2)×PP(2) with 4 idle duplicates.

### GROUP 1: Single-Node PCIe Baselines (4 GPUs, 1 node, GPT-2 Medium)

| Config | Throughput | Peak Mem | Phase 2 (TCP) | PCIe Speedup |
|--------|-----------|----------|---------------|-------------|
| Single GPU | 1,399 tok/s | 7.67 GB | 1,361 tok/s | 1.0× (ref) |
| DP(4) | 3,389 tok/s* | 7.67 GB | 1,393 tok/s* | 2.4× |
| TP(4) | 2,518 tok/s | 3.38 GB | 959 tok/s | 2.6× |
| PP(4) 1F1B | 3,200 tok/s | 2.77 GB | 3,070 tok/s | 1.04× |
| DP+ZeRO-1 | 2,832 tok/s* | 9.47 GB | 934 tok/s* | 3.0× |

(* = aggregate across all GPUs)

**Key insight**: DP and TP benefit hugely from PCIe (2.4–3×). PP barely changes (1.04×) because it only sends small point-to-point activation tensors. This motivates placing PP on the slow inter-node link.

### GROUP 2: Pre-fix Good Placement (8 GPUs, PP inter-node)

| Config | Throughput | Peak Mem |
|--------|-----------|----------|
| dp=2, pp=2, tp=2 | 3,503 tok/s | 4.08 GB |
| dp=2, pp=2, tp=2 +ZeRO-1 | 3,044 tok/s | 5.74 GB |

### GROUP 3: Pre-fix Bad Placement (8 GPUs, DP inter-node)

| Config | Throughput | Peak Mem | vs Good |
|--------|-----------|----------|---------|
| dp=2, pp=2, tp=2 | 2,904 tok/s | 4.08 GB | -17.1% |
| dp=2, pp=2, tp=2 +ZeRO-1 | 2,416 tok/s | 5.74 GB | -20.6% |

### GROUP 5: Pre-fix Multi-Model Placement (8 GPUs, dp=2, pp=2, tp=2)

| Model | Good (tok/s) | Bad (tok/s) | Delta |
|-------|-------------|------------|-------|
| GPT-2 Small (117M) | 8,546 | 7,156 | +19.4% |
| GPT-2 Medium (354M) | 3,494 | 2,926 | +19.4% |
| GPT-2 Large (774M) | 1,746 | hangs | — |

### GROUP 6: T5-base Single-Node Baselines (4 GPUs, PCIe)

| Config | T5-base | GPT-2 Med | GPT-2 Small |
|--------|---------|-----------|-------------|
| Single GPU | 3,359 tok/s | 1,404 tok/s | 3,879 tok/s |
| DP(4) | 6,477 tok/s | 3,389 tok/s | 8,546 tok/s |
| TP(4) | 5,029 tok/s | 2,518 tok/s | 5,458 tok/s |
| PP(2) PCIe | 4,297 tok/s | — | — |

T5 PP note: encoder=stage 0, decoder=stage 1. Memory imbalance: encoder 2.51 GB, decoder 3.87 GB (54% heavier).

### GROUP 7: Pre-fix T5 Placement (8 GPUs, dp=2, pp=2, tp=2)

| Placement | T5-base | GPT-2 Medium |
|-----------|---------|-------------|
| Good (PP inter-node) | 5,790 tok/s | 3,519 tok/s |
| TP inter-node | 5,838 tok/s | 3,543 tok/s |
| Bad (DP inter-node) | 4,558 tok/s | 2,926 tok/s |
| Good vs Bad | +27.0% | +20.3% |

**Pre-fix surprise**: TP inter-node showed NO penalty because TP carried zero traffic (bug). This was corrected post-fix.

### GROUP 8: Pre-fix Architecture Comparison (good placement, 8 GPUs)

| Model | Params | 1-GPU | 8-GPU Hybrid | Speedup | PP Mem Balance |
|-------|--------|-------|-------------|---------|---------------|
| GPT-2 Small | 117M | 3,879 tok/s | 8,678 tok/s | 2.24× | 1.65 vs 1.71 GB (symmetric) |
| GPT-2 Medium | 354M | 1,404 tok/s | 3,519 tok/s | 2.51× | 4.07 vs 4.08 GB (symmetric) |
| T5-base | 237M | 3,359 tok/s | 5,790 tok/s | 1.72× | 2.51 vs 3.87 GB (54% imbalance) |

---

## RESULTS — POST-FIX (True 3D: TP actually active)

These are the CORRECT results with true TP+PP working together.

### GROUP B: True 3D Good Placement (GPT-2 Medium, dp=2, pp=2, tp=2)

| Config | Throughput | Speedup* | Peak Mem |
|--------|-----------|----------|----------|
| B1: bs=8 | 4,851 tok/s | 3.11× | 2.57 GB |
| B2: bs=16 | 6,012 tok/s | 3.85× | 2.89 GB |
| B3: bs=32 | 6,618 tok/s | 4.24× | 3.60 GB |
| B4: bs=64 | 7,245 tok/s | 4.64× | 5.14 GB |
| B5: bs=128 | 7,387 tok/s | 4.73× | 8.21 GB |
| bs=256 | OOM | — | >14.5 GB |

*Speedup vs single-GPU max (1,561 tok/s at bs=8) for fair comparison.
Throughput plateaus at bs=64–128: GPUs become compute-bound once communication latency is fully hidden.

Pre-fix reference: 3,519 tok/s, 2.26× (vs 1-GPU bs=8), 4.07/4.08 GB

### GROUP C: True 3D Bad Placement (GPT-2 Medium, DP inter-node)

| Config | Throughput | Speedup |
|--------|-----------|---------|
| C1: bs=8 | 4,126 tok/s | 2.94× |
| bs≥16 | DEADLOCKS | — |

Pre-fix reference: 2,926 tok/s, 2.08×

### GROUP D: True 3D Worst Placement (GPT-2 Medium, TP inter-node)

| Config | Throughput | Speedup |
|--------|-----------|---------|
| D1: bs=8 | 3,776 tok/s | 2.69× |
| bs≥16 | DEADLOCKS | — |

Pre-fix reference: 3,543 tok/s, 2.52×

Bad and worst placements deadlock at bs≥16 due to NCCL P2P contention when larger activation tensors compete with DP/TP traffic on the same inter-node TCP link. Good placement avoids this by isolating PP on inter-node.

### GROUP E: Architecture Comparison (good placement, bs=8)

| Model | 1-GPU | Post-Fix 8-GPU | Speedup | Pre-Fix 8-GPU | Pre-Fix Speedup |
|-------|-------|---------------|---------|--------------|----------------|
| GPT-2 Small (117M) | 3,879 tok/s | 10,297 tok/s | 2.65× | 8,678 tok/s | 2.24× |
| GPT-2 Medium (354M) | 1,404 tok/s | 5,014 tok/s | 3.57× | 3,519 tok/s | 2.51× |
| T5-base (237M) | 3,359 tok/s | 5,973 tok/s | 1.78× | 5,790 tok/s | 1.72× |

Post-fix memory:
- T5 peak mem: encoder stage 1.69 GB, decoder stage 2.75 GB (63% heavier)
- GPT-2 Med: both stages ~2.57 GB (balanced)
- GPT-2 Small: stage 0 = 1.24 GB, stage 1 = 1.32 GB (balanced)

### GROUP F: True 3D + ZeRO-1 (GPT-2 Medium, good placement)

| Config | Throughput | Speedup | Peak Mem (S0/S1) |
|--------|-----------|---------|-----------------|
| F1: bs=8 | 4,405 tok/s | 3.14× | 3.61/3.58 GB |

Pre-fix reference: 3,044 tok/s, 2.17×

---

## KEY FINDINGS

### 1. Placement Impact (Post-Fix, True 3D)

| Placement | Throughput | vs Good |
|-----------|-----------|---------|
| Good (PP inter-node) | 4,851 tok/s | — |
| Bad (DP inter-node) | 4,126 tok/s | -14.9% |
| Worst (TP inter-node) | 3,776 tok/s | -22.2% |

With real TP traffic, **TP inter-node is the actual worst placement** (not DP inter-node as appeared pre-fix). The pre-fix showed no TP penalty because TP was dead code carrying zero traffic.

Correct placement heuristic: TP on fastest link (most latency-sensitive, many all-reduces per step), PP on slowest link (small P2P sends), DP in between.

### 2. Throughput Improvement from True TP

| Model | Pre-Fix | Post-Fix | Improvement |
|-------|---------|----------|------------|
| GPT-2 Small | 8,678 tok/s | 10,297 tok/s | +18.7% |
| GPT-2 Medium | 3,519 tok/s | 5,014 tok/s | +42.5% |
| T5-base | 5,790 tok/s | 5,973 tok/s | +3.2% |

T5 gains least because cross-attention adds more TP all-reduce overhead.

### 3. Memory Savings from True TP

- Pre-fix: ~4.07 GB/GPU (full model blocks, TP inactive)
- Post-fix: ~2.57 GB/GPU (TP-sharded blocks)
- **37% memory reduction**
- Params per GPU: 127.3M (vs 354M full) for GPT-2 Medium
- Single GPU: max bs=8 (11.07 GB peak, OOMs at bs=16)
- 8-GPU with TP: max bs=128 (8.21 GB peak, OOMs at bs=256)
- **TP enables 16× larger batch size**

### 4. Batch Size Scaling (enabled by TP memory savings)

| Batch Size | Throughput | Peak Mem | Δ vs prev |
|-----------|-----------|----------|----------|
| 8 | 4,851 tok/s | 2.57 GB | (baseline) |
| 16 | 6,012 tok/s | 2.89 GB | +24% |
| 32 | 6,618 tok/s | 3.60 GB | +10% |
| 64 | 7,245 tok/s | 5.14 GB | +9% |
| 128 | 7,387 tok/s | 8.21 GB | +2% |

Diminishing returns: at large batches, communication overhead is fully hidden and GPU compute (FLOPS) becomes the bottleneck. The +2% from bs=64→128 shows the T4s are nearly saturated.

### 5. Architecture Scaling — Two Views

**At bs=8 (same batch, apples-to-apples):**

| Model | Hybrid Speedup |
|-------|---------------|
| GPT-2 Small (117M) | 2.65× |
| GPT-2 Medium (354M) | 3.57× |
| T5-base (237M) | 1.78× |

**At max batch (each config at peak throughput, bs=128 on 8-GPU):**

| Model | 1-GPU max | 8-GPU (bs=128) | Speedup |
|-------|-----------|---------------|--------|
| GPT-2 Small | 4,426 (bs=16) | 15,546 tok/s | 3.51× |
| GPT-2 Medium | 1,561 (bs=8) | 7,387 tok/s | 4.73× |
| T5-base | 4,747 (bs=32) | 16,555 tok/s | 3.49× |

**8-GPU batch sweep (good placement, all models):**

| Model | bs=8 | bs=32 | bs=64 | bs=128 |
|-------|------|-------|-------|--------|
| GPT-2 Small | 10,297 | 14,534 | 15,509 | 15,546 |
| GPT-2 Medium | 5,014 | 6,618 | 7,245 | 7,387 |
| T5-base | 5,973 | 13,454 | 15,589 | 16,555 |

At bs=8, T5 scales worst (1.78×). At max batch, T5 catches up to GPT-2 Small (3.49× vs 3.51×) because larger batches hide its communication overhead. GPT-2 Medium leads (4.73×) because it has the highest compute-to-communication ratio.

### 6. Placement Stability

Good placement works at all batch sizes up to bs=128. Bad and worst placements **deadlock at bs≥16**. This makes good placement not just faster (+17–29%) but essential for production use.

GPT-2 scales much better than T5. Two architectural factors:

**Pipeline imbalance**: GPT-2 has identical blocks → balanced stages. T5 decoder stage is 63% heavier than encoder (post-fix: 2.75 vs 1.69 GB). In 1F1B, encoder completes each microbatch faster and waits for the slower decoder.

**Cross-attention TP overhead**: Each T5 decoder block has self-attn + cross-attn + MLP = 3 all-reduces (vs 2 for GPT-2). T5-base total: 12×2 + 12×3 = 60 all-reduces. GPT-2 Medium: 24×2 = 48. T5 has 25% more despite fewer params.

### 7. Theoretical Analysis

With true 3D (dp=2, pp=2, tp=2):
- TP does NOT multiply throughput — it's a memory optimization that adds comm overhead
- Effective parallelism for throughput: DP × PP = 2 × 2 = 4× max
- Pipeline bubble: (S-1)/(S-1+M) = 1/(1+8) ≈ 11%
- Realistic ceiling: ~3.2-3.5× (after DP sync + TP all-reduce overhead)

---

## Bugs Fixed During Phase 3

1. **pipeline_parallel.py**: Replaced blocking `dist.send`/`dist.recv` with `dist.batch_isend_irecv()` — NCCL deadlock fix.
2. **plugin.py**: Fixed hardcoded `DP_AXIS` in parameter broadcast; used `self._dp_axis` to support bad_placement mesh axis swap.
3. **plugin.py**: Corrected mesh layout from `(dp, pp, tp)` to `(pp, dp, tp)` so good placement puts PP inter-node.
4. **plugin.py**: Fixed `if/elif` for PP/TP → added `if pp_size > 1 and tp_size > 1` branch with `TensorParallelPipelineStage`.

## BibTeX Keys Available in references.bib

- `10.1145/3605573.3605613` — Colossal-AI
- `DBLP:journals/corr/abs-1810-04805` — BERT
- `DBLP:journals/corr/abs-2005-14165` — GPT-3
- `DBLP:journals/corr/abs-1909-08053` — Megatron-LM
- `rajbhandari2019zero` — ZeRO
- `DBLP:journals/corr/abs-1811-06965` — GPipe
- `10.1145/3341301.3359646` — PipeDream
- `radford2019language` — GPT-2
- `patarasuk2009bandwidth` — Ring all-reduce
- `li2020pytorch` — PyTorch Distributed
- `rasley2020deepspeed` — DeepSpeed
- `li2021chimera` — Chimera
- `narayanan2021memory` — PipeDream-2BW
- `raffel2020exploring` — T5

## Notes for Report Writing

- Phase 2 report was ~285 lines of LaTeX and fit ~4 pages. Aim for similar length.
- Use `\usepackage{booktabs}` for tables (already in Phase 2).
- Figures directory: `figures/` (contains `fig_dp_throughput.png`, `fig_summary_all.png`, `fig_throughput_vs_memory.png` from Phase 2). New figures may need to be generated.
- The report should tell a coherent story: Phase 2 was all-TCP single-GPU-per-node → Phase 3 is heterogeneous interconnect with true 3D parallelism and multi-model analysis.
- Include BOTH pre-fix and post-fix results. The bug discovery is itself a finding (shows how subtle distributed bugs can hide — TP appeared to work but was doing nothing).
- The post-fix results are the "correct" numbers to use for the main findings. Pre-fix numbers are useful for the bug discussion section.
