# Mini Colossal-AI

A simplified distributed training library inspired by [Colossal-AI](https://github.com/hpcaitech/ColossalAI), built from scratch using only `torch.distributed` primitives (`send`, `recv`, `all_reduce`, `all_gather`, etc.). No PyTorch DDP, FSDP, or Pipeline APIs are used.

Trains GPT-2 (Medium/Large) on WikiText-2 across a multi-node GPU cluster.

## Repository Structure

```
mini-colossal-ai/
├── minicolossal/                  # The library
│   ├── __init__.py
│   ├── gpt2.py                    # GPT-2 model (Medium 354M, Large 774M)
│   ├── data.py                    # WikiText-2 data loading with tiktoken
│   ├── utils.py                   # Metrics, timers, print utilities
│   ├── data_parallel.py           # Ring all-reduce, bucketed all-reduce
│   ├── zero_optim.py              # ZeRO Stage 1 & Stage 2 optimizers
│   ├── tensor_parallel.py         # 1D tensor parallelism (Megatron-LM style)
│   ├── pipeline_parallel.py       # Pipeline stages, naive + 1F1B schedules
│   ├── hybrid_parallel.py         # Process group helpers for pairwise combos
│   └── plugin.py                  # Unified 3D plugin (ProcessGroupMesh + MiniColossalPlugin)
│
├── benchmarks/                    # Benchmark scripts and launchers
│   ├── bench_single_gpu.py        # Single GPU baseline
│   ├── bench_data_parallel.py     # Data parallelism (4 methods)
│   ├── bench_zero.py              # ZeRO Stage 1 & 2
│   ├── bench_tensor_parallel.py   # 1D Tensor Parallelism
│   ├── bench_pipeline.py          # Pipeline Parallelism (naive + 1F1B)
│   ├── bench_unified.py           # Unified 3D benchmark (any config via CLI)
│   ├── launch_dp.sh               # Multi-node launcher for DP
│   ├── launch_zero.sh             # Multi-node launcher for ZeRO
│   ├── launch_tp.sh               # Multi-node launcher for TP
│   ├── launch_pipeline.sh         # Multi-node launcher for PP
│   └── launch_unified.sh          # Multi-node launcher for unified benchmark
│
├── minicolossal_design.txt        # Full design document
├── RESULTS.md                     # Benchmark results and analysis
├── claude_context.md              # Development conversation log
└── README.md
```

## Prerequisites

- **Python** 3.8+
- **PyTorch** 2.4+ with CUDA
- **GPU cluster**: N nodes, each with 1+ NVIDIA GPU (tested on 4x g4dn.xlarge with Tesla T4)
- **Network**: All nodes must be reachable via SSH from the master node
- **NCCL**: Installed (comes with PyTorch)

## Installation

### 1. Install dependencies on ALL nodes

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install datasets tiktoken
```

### 2. Clone the repo on the master node

```bash
git clone <repo-url> ~/mini-colossal-ai
cd ~/mini-colossal-ai
```

The launch scripts automatically `rsync` code to worker nodes before each run.

### 3. Configure your cluster

Each launch script uses these environment variables (with defaults):

| Variable | Description | Default |
|----------|-------------|---------|
| `PEM` | Path to SSH private key | `/home/ubuntu/workspace/gpooloth-experiment-DPDKML.pem` |
| `MASTER_ADDR` | Master node IP | `10.0.3.175` |
| `WORKER_IPS` | Space-separated worker IPs | `10.0.3.181 10.0.3.115 10.0.3.34` |

Override them for your cluster:

```bash
export PEM=~/.ssh/my-key.pem
export MASTER_ADDR=192.168.1.100
export WORKER_IPS="192.168.1.101 192.168.1.102 192.168.1.103"
```

### 4. Verify SSH access

```bash
for IP in $WORKER_IPS; do
    ssh -i $PEM ubuntu@$IP "hostname && nvidia-smi --query-gpu=name --format=csv,noheader"
done
```

### 5. Sync HuggingFace dataset cache

Download WikiText-2 on master, then sync to all workers:

```bash
python3 -c "from datasets import load_dataset; load_dataset('wikitext', 'wikitext-2-raw-v1')"

for IP in $WORKER_IPS; do
    rsync -az -e "ssh -i $PEM" ~/.cache/huggingface/ ubuntu@$IP:~/.cache/huggingface/
done
```

## Running Benchmarks

### Single GPU Baseline

```bash
python3 benchmarks/bench_single_gpu.py
```

### Standalone Components (4 nodes)

```bash
# Data Parallelism (methods: naive, ring, ring_bucketed, allreduce_bucketed)
bash benchmarks/launch_dp.sh 4 allreduce_bucketed

# ZeRO Optimizer (stage: 1 or 2)
bash benchmarks/launch_zero.sh 4 1
bash benchmarks/launch_zero.sh 4 2

# 1D Tensor Parallelism
bash benchmarks/launch_tp.sh 4

# Pipeline Parallelism (schedule: naive or 1f1b)
bash benchmarks/launch_pipeline.sh 4 1f1b
```

### Unified 3D Hybrid Parallelism

The unified plugin (`bench_unified.py`) handles **any combination** of DP, TP, PP, and ZeRO via CLI arguments. The DP size is auto-calculated: `dp_size = world_size / (tp_size * pp_size)`.

```bash
# Pure DP (4-way)
bash benchmarks/launch_unified.sh --tp_size 1 --pp_size 1 --zero_stage 0

# DP + ZeRO-1 (4-way)
bash benchmarks/launch_unified.sh --tp_size 1 --pp_size 1 --zero_stage 1

# DP(2) x TP(2)
bash benchmarks/launch_unified.sh --tp_size 2 --pp_size 1 --zero_stage 0

# DP(2) x PP(2)
bash benchmarks/launch_unified.sh --tp_size 1 --pp_size 2 --zero_stage 0

# PP(2) + ZeRO-1 within DP(2)
bash benchmarks/launch_unified.sh --tp_size 1 --pp_size 2 --zero_stage 1
```

Additional options:

| Flag | Default | Description |
|------|---------|-------------|
| `--batch_size` | 4 | Batch size per GPU |
| `--num_microbatches` | 8 | Microbatches for pipeline parallelism |
| `--num_steps` | 30 | Training steps |

**Constraint** (from Colossal-AI): `zero_stage` must be 0 or 1 when `pp_size > 1`. ZeRO-2 is forbidden with pipeline parallelism.

## Architecture

### Composable Building Blocks

Each standalone component exposes a clean interface with optional `group` parameters, enabling reuse in hybrid configurations without code duplication:

| Component | Interface | Hybrid Usage |
|-----------|-----------|-------------|
| **DP gradient sync** | `allreduce_bucketed_grads(model, world_size, group=None)` | Pass `group=dp_group` |
| **PP 1F1B schedule** | `one_f_one_b_forward_backward(..., prev_rank=None, next_rank=None)` | Pass global rank mapping |
| **TP model** | `TensorParallelGPT2(cfg, tp_size, tp_rank, tp_group=None)` | Pass `tp_group` |
| **ZeRO-1 optimizer** | `ZeROStage1Optimizer(model, ..., group=None)` | Pass `group=dp_group` |

### Unified 3D Plugin (Colossal-AI style)

`MiniColossalPlugin` composes these building blocks using a `ProcessGroupMesh`:

```python
from minicolossal.plugin import MiniColossalPlugin

plugin = MiniColossalPlugin(tp_size=2, pp_size=1, zero_stage=0)
model, optimizer = plugin.configure(cfg, lr=3e-4, device=device)

for batch in dataloader:
    loss = plugin.train_step(model, optimizer, batch, criterion, cfg)
```

The `ProcessGroupMesh` creates an N-dimensional grid of ranks:
- **Axis 0**: Data Parallelism
- **Axis 1**: Pipeline Parallelism
- **Axis 2**: Tensor Parallelism

## Results Summary

| Configuration | Throughput | Peak Mem/GPU | Key Benefit |
|---------------|-----------|--------------|-------------|
| Single GPU | 1,361 tok/s | 7.67 GB | Baseline |
| DP (4-way) | 1,322 tok/s | 7.67 GB | Scale batch size |
| ZeRO-1 (4-way) | 926 tok/s | 9.47 GB | 4x optimizer saving |
| ZeRO-2 (4-way) | 873 tok/s | 6.96 GB | 4x grad+optim saving |
| TP (4-way) | 915 tok/s | 3.39 GB | 56% mem reduction |
| PP 1F1B (4-stage) | 3,040 tok/s | 2.77 GB | Best throughput |
| DP(2) x TP(2) | 2,327 tok/s | 4.80 GB | Balanced hybrid |
| DP(2) x PP(2) | 2,281 tok/s | 4.08 GB | Low bubble + DP |
| PP(2)+ZeRO-1 in DP(2) | 1,742 tok/s | 5.74 GB | PP + optim saving |

See [RESULTS.md](RESULTS.md) for detailed analysis.

## References

- [Colossal-AI](https://github.com/hpcaitech/ColossalAI) — The system this project is inspired by
- Shoeybi et al., "Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism" (2019)
- Narayanan et al., "PipeDream: Generalized Pipeline Parallelism for DNN Training" (2019)
- Rajbhandari et al., "ZeRO: Memory Optimizations Toward Training Trillion Parameter Models" (2020)
