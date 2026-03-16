"""
Data Parallelism Benchmark
============================
Tests our custom data parallelism (ring all-reduce + gradient bucketing)
against the single-GPU baseline.

Compares three methods:
  1. naive      — all-to-all send/recv (slow, O(P) bandwidth)
  2. ring       — ring all-reduce per parameter
  3. ring_bucketed — ring all-reduce with gradient bucketing (our best)

Usage (run from node 0, launcher handles all nodes):
    bash benchmarks/launch_dp.sh <num_nodes> <method>
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist

from minicolossal.gpt2 import GPT2Config, GPT2Model
from minicolossal.data import get_dataloader
from minicolossal.data_parallel import DataParallelEngine
from minicolossal.utils import MetricsTracker, StepTimer, print_metrics, print_model_info


def main():
    # ---- Parse args ----
    method = os.environ.get("DP_METHOD", "ring_bucketed")

    # ---- Initialize distributed ----
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    if rank == 0:
        print(f"\n  Data Parallelism Benchmark")
        print(f"  Method: {method}")
        print(f"  World size: {world_size}")
        print(f"  Backend: nccl")

    # ---- Config ----
    cfg = GPT2Config.medium()
    batch_size = 4    # per GPU
    num_steps = 30
    lr = 3e-4

    # ---- Model ----
    model = GPT2Model(cfg).to(device)
    if rank == 0:
        print_model_info(model, cfg)

    # ---- Data (distributed sampler splits data across workers) ----
    if rank == 0:
        print(f"\n  Loading WikiText-2...")
    dataloader, dataset = get_dataloader(
        split="train", seq_len=cfg.max_seq_len, batch_size=batch_size,
        num_workers=2, distributed=True, rank=rank, world_size=world_size,
    )
    if rank == 0:
        print(f"  Batches per GPU: {len(dataloader)}")
        print(f"  Effective batch size: {batch_size * world_size}")

    # ---- Optimizer ----
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()

    # ---- Wrap model with our DataParallelEngine ----
    engine = DataParallelEngine(model, world_size, rank, method=method)

    # ---- Training ----
    if rank == 0:
        print(f"\n  Training for {num_steps} steps...\n")

    tracker = MetricsTracker(device)
    tracker.start_run()
    model.train()

    step = 0
    for input_ids, target_ids in dataloader:
        if step >= num_steps:
            break

        input_ids = input_ids.to(device, non_blocking=True)
        target_ids = target_ids.to(device, non_blocking=True)

        with StepTimer(device) as timer:
            loss_val = engine.train_step(input_ids, target_ids, optimizer, criterion)

        num_tokens = input_ids.numel()
        tracker.log_step(loss_val, num_tokens, timer.elapsed)
        step += 1

        if step % 10 == 0:
            tok_per_s = num_tokens / timer.elapsed
            peak_mem = torch.cuda.max_memory_allocated(device) / 1e9
            print(f"  [Rank {rank}] Step {step}/{num_steps}  loss={loss_val:.4f}  "
                  f"tok/s={tok_per_s:.0f}  mem={peak_mem:.2f}GB")

    results = tracker.finish_run()

    # Gather throughput from all ranks
    local_throughput = torch.tensor([results["throughput_tokens_per_s"]], device=device)
    all_throughput = [torch.zeros(1, device=device) for _ in range(world_size)]
    dist.all_gather(all_throughput, local_throughput)

    if rank == 0:
        total_throughput = sum(t.item() for t in all_throughput)
        results["total_throughput_tokens_per_s"] = total_throughput
        print_metrics(results, label=f"Data Parallel ({method}, {world_size} GPUs)")
        print(f"  Aggregate throughput: {total_throughput:.0f} tokens/s")
        for r, t in enumerate(all_throughput):
            print(f"  [Rank {r}] throughput: {t.item():.0f} tokens/s")

        # Scaling efficiency vs single GPU baseline (1361 tokens/s from our test)
        single_gpu_baseline = 1361  # tokens/s
        ideal = single_gpu_baseline * world_size
        efficiency = (total_throughput / ideal) * 100
        print(f"\n  Scaling efficiency: {efficiency:.1f}% "
              f"(ideal={ideal:.0f}, actual={total_throughput:.0f})")

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
