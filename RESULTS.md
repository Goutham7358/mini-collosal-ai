# Benchmark Results

## Hardware Setup

| Spec | Value |
|------|-------|
| **Cluster** | 4x AWS g4dn.xlarge |
| **GPU per node** | 1x NVIDIA Tesla T4 (16 GB VRAM) |
| **vCPUs** | 4 per node |
| **RAM** | 16 GB per node |
| **Network** | VPC private network (TCP sockets, no EFA/InfiniBand) |
| **NCCL interface** | `ens5` (VPC ENI) |

## Software

| Component | Version |
|-----------|---------|
| **PyTorch** | 2.4.1 |
| **CUDA** | 12.1 |
| **Python** | 3.8 |
| **Backend** | NCCL |

## Model & Data

| Item | Details |
|------|---------|
| **Model** | GPT-2 Medium (354M params, 24 layers) for most benchmarks; GPT-2 Large (774M, 36 layers) for 4-stage pipeline |
| **Dataset** | WikiText-2 (2.4M tokens) |
| **Tokenizer** | tiktoken (`gpt2` encoding) |
| **Sequence length** | 256 |
| **Batch size** | 4 (standalone), 8 (pipeline/hybrid) |

---

## Standalone Component Results (4 GPUs)

### Single GPU Baseline

| Metric | Value |
|--------|-------|
| Throughput | 1,361 tokens/s |
| Peak memory | 7.67 GB |
| Steps | 30 in 22.6s |

### Data Parallelism (4-way, `allreduce_bucketed`)

| Metric | Value |
|--------|-------|
| Aggregate throughput | 1,322 tokens/s |
| Per-GPU throughput | 330 tokens/s |
| Peak memory | 7.67 GB |
| Scaling efficiency | 24.3% |

> Low scaling efficiency is expected on g4dn.xlarge — no EFA/InfiniBand means gradient all-reduce goes over TCP sockets. This is a good datapoint showing why network bandwidth matters.

### ZeRO Stage 1 (4-way)

| Metric | Value |
|--------|-------|
| Aggregate throughput | 926 tokens/s |
| Peak memory | 9.47 GB |
| Optimizer memory saving | 4x (0.71 GB vs 2.83 GB standard) |

### ZeRO Stage 2 (4-way)

| Metric | Value |
|--------|-------|
| Aggregate throughput | 873 tokens/s |
| Peak memory | 6.96 GB |
| Grad + optimizer memory saving | 4x |

### 1D Tensor Parallelism (4-way)

| Metric | Value |
|--------|-------|
| Throughput | 915 tokens/s |
| Peak memory | 3.39 GB (56% reduction) |
| Communication | 48 all-reduces per forward pass (2 per transformer block) |

### Pipeline Parallelism — 1F1B (4 stages)

| Metric | Value |
|--------|-------|
| Throughput | 3,040 tokens/s |
| Peak memory (stage 0) | 2.77 GB |
| Peak memory (stage 1) | 1.75 GB |
| Peak memory (stage 2) | 1.59 GB |
| Peak memory (stage 3) | 2.56 GB |
| Theoretical bubble ratio | 27.3% (8 microbatches) |
| Memory reduction | 64% vs single GPU |

---

## Hybrid Parallelism Results (4 GPUs, Unified Plugin)

All hybrid configurations use the unified `MiniColossalPlugin` with `ProcessGroupMesh`.

### DP(2) x TP(2)

| Metric | Value |
|--------|-------|
| Aggregate throughput | 2,327 tokens/s |
| Per-GPU throughput | 582 tokens/s |
| Peak memory | 4.80 GB |
| Layout | TP groups [0,1] [2,3]; DP groups [0,2] [1,3] |

### DP(2) x PP(2)

| Metric | Value |
|--------|-------|
| Aggregate throughput | 2,281 tokens/s |
| Peak memory | 4.08 GB |
| Theoretical bubble | 11.1% (8 microbatches, 2 stages) |
| Layout | Pipelines [0,1] [2,3]; DP groups [0,2] [1,3] |

### PP(2) + ZeRO-1 within DP(2)

| Metric | Value |
|--------|-------|
| Aggregate throughput | 1,742 tokens/s |
| Peak memory | 5.74 GB (stage 0), 6.31 GB (stage 1) |
| Theoretical bubble | 11.1% |
| ZeRO-1 optimizer saving | 2x (0.81 GB vs 1.62 GB) |
| Layout | Pipelines [0,1] [2,3]; DP groups [0,2] [1,3] |

> Follows Colossal-AI's constraint: ZeRO-2 is forbidden with PP due to prohibitive gradient synchronization costs.

---

## Summary Table

| Configuration | Throughput | Peak Mem/GPU | Key Benefit |
|---------------|-----------|--------------|-------------|
| Single GPU (baseline) | 1,361 tok/s | 7.67 GB | -- |
| DP (4-way) | 1,322 tok/s* | 7.67 GB | Scale batch size |
| ZeRO Stage 1 (4-way) | 926 tok/s* | 9.47 GB | 4x optimizer mem saving |
| ZeRO Stage 2 (4-way) | 873 tok/s* | 6.96 GB | 4x grad+optim saving |
| TP (4-way) | 915 tok/s | 3.39 GB | 56% model mem reduction |
| PP 1F1B (4 stages) | 3,040 tok/s | 2.77 GB | 64% mem, best throughput |
| **DP(2) x TP(2)** | 2,327 tok/s* | 4.80 GB | Balanced hybrid |
| **DP(2) x PP(2)** | 2,281 tok/s* | 4.08 GB | Low bubble + DP scaling |
| **PP(2)+ZeRO-1 in DP(2)** | 1,742 tok/s* | 5.74 GB | PP + 2x optim saving |

\* = aggregate throughput across all GPUs

---

## Key Findings

1. **Pipeline parallelism achieves the best throughput** (3,040 tok/s) because communication is only between adjacent stages (point-to-point), not all-to-all.

2. **Tensor parallelism gives the best per-GPU memory reduction** for model weights (56%), but 48 all-reduces per forward pass limit throughput vs single GPU.

3. **ZeRO Stage 2 reduces total memory** but adds communication overhead from reduce-scatter + broadcast operations.

4. **Hybrid configurations** (DP x TP, DP x PP, PP + ZeRO-1) provide good tradeoffs between memory, throughput, and scalability.

5. **PP + ZeRO-1 follows Colossal-AI's constraint**: ZeRO-2 is forbidden with PP due to prohibitive gradient synchronization costs.

6. **Network bandwidth is the main bottleneck** on g4dn.xlarge (no EFA/InfiniBand), which explains the low DP scaling efficiency (24.3%).
