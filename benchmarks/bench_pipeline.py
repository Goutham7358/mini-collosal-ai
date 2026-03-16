"""
Pipeline Parallelism Benchmark
================================
Tests naive pipeline vs 1F1B schedule on 4 GPUs.
Uses GPT-2 Medium split across stages (6 blocks per stage with 4 GPUs).

Usage (via launcher):
    bash benchmarks/launch_pipeline.sh <num_nodes> <schedule>
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import time
import socket
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist

from minicolossal.gpt2 import GPT2Config
from minicolossal.pipeline_parallel import (
    create_pipeline_stage,
    split_into_microbatches,
    naive_pipeline_forward_backward,
    one_f_one_b_forward_backward,
)
from minicolossal.data import get_dataloader
from minicolossal.utils import MetricsTracker, StepTimer


def main():
    schedule = os.environ.get("PP_SCHEDULE", "1f1b")

    # ---- Initialize distributed ----
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    # Print hostname for each rank to prove multi-node execution
    hostname = socket.gethostname()
    ip_addr = socket.gethostbyname(hostname)
    print(f"  [Rank {rank}] Running on {hostname} ({ip_addr})")
    dist.barrier()

    if rank == 0:
        print(f"\n  Pipeline Parallelism Benchmark")
        print(f"  Schedule: {schedule}")
        print(f"  World size (stages): {world_size}")

    # ---- Config ----
    cfg = GPT2Config.medium()  # 24 layers -> 6 per stage with 4 GPUs
    batch_size = 8             # total batch size, split into microbatches
    num_microbatches = 8       # more microbatches = less bubble
    num_steps = 20
    lr = 3e-4

    micro_bs = batch_size // num_microbatches
    if micro_bs < 1:
        micro_bs = 1
        num_microbatches = batch_size

    if rank == 0:
        print(f"  Model: GPT-2 Medium ({cfg.n_layers} layers)")
        print(f"  Layers per stage: {cfg.n_layers // world_size}")
        print(f"  Batch size: {batch_size}")
        print(f"  Microbatches: {num_microbatches} (micro_bs={micro_bs})")

    # ---- Create pipeline stage ----
    stage = create_pipeline_stage(cfg, world_size, rank, device)
    n_params = sum(p.numel() for p in stage.parameters()) / 1e6
    print(f"  [Rank {rank}] Stage params: {n_params:.1f}M  "
          f"blocks: {stage.block_indices}  "
          f"first={stage.is_first}  last={stage.is_last}")

    # ---- Data ----
    # All ranks load the same data with the same seed. WikiText-2 is small (~10MB).
    # First stage needs input_ids, last stage needs target_ids for loss.
    # Middle stages only need the microbatch count/shapes.
    torch.manual_seed(42)
    dl, _ = get_dataloader(
        split="train", seq_len=cfg.max_seq_len, batch_size=batch_size,
        num_workers=0, distributed=False,
    )
    dataloader = iter(dl)

    # ---- Optimizer ----
    optimizer = optim.AdamW(stage.parameters(), lr=lr, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()

    # ---- Select schedule function ----
    if schedule == "naive":
        schedule_fn = naive_pipeline_forward_backward
    else:
        schedule_fn = one_f_one_b_forward_backward

    # ---- Training ----
    dist.barrier()
    if rank == 0:
        print(f"\n  Training for {num_steps} steps...\n")

    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    start_time = time.perf_counter()

    stage.train()
    total_loss = 0.0

    for step in range(num_steps):
        try:
            input_ids, target_ids = next(dataloader)
        except StopIteration:
            torch.manual_seed(42 + step)
            dl, _ = get_dataloader(
                split="train", seq_len=cfg.max_seq_len, batch_size=batch_size,
                num_workers=0, distributed=False,
            )
            dataloader = iter(dl)
            input_ids, target_ids = next(dataloader)
        microbatches = split_into_microbatches(input_ids, target_ids, num_microbatches)

        optimizer.zero_grad()

        loss_val = schedule_fn(
            stage, criterion, microbatches, cfg, rank, world_size, device
        )

        torch.nn.utils.clip_grad_norm_(stage.parameters(), 1.0)
        optimizer.step()

        if rank == world_size - 1:
            total_loss += loss_val

        if (step + 1) % 5 == 0:
            peak_mem = torch.cuda.max_memory_allocated(device) / 1e9
            if rank == world_size - 1:
                print(f"  [Rank {rank}] Step {step+1}/{num_steps}  "
                      f"loss={loss_val:.4f}  mem={peak_mem:.2f}GB")
            else:
                print(f"  [Rank {rank}] Step {step+1}/{num_steps}  mem={peak_mem:.2f}GB")

    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start_time

    # ---- Results ----
    dist.barrier()
    peak_mem = torch.cuda.max_memory_allocated(device) / 1e9
    tokens_processed = num_steps * batch_size * cfg.max_seq_len

    # Gather peak memory from all ranks
    local_mem = torch.tensor([peak_mem], device=device)
    all_mem = [torch.zeros(1, device=device) for _ in range(world_size)]
    dist.all_gather(all_mem, local_mem)

    local_time = torch.tensor([elapsed], device=device)
    all_times = [torch.zeros(1, device=device) for _ in range(world_size)]
    dist.all_gather(all_times, local_time)

    if rank == 0:
        max_time = max(t.item() for t in all_times)
        throughput = tokens_processed / max_time

        print(f"\n{'='*60}")
        print(f"  RESULTS: Pipeline Parallel ({schedule}, {world_size} stages)")
        print(f"{'='*60}")
        print(f"  Steps:              {num_steps}")
        print(f"  Total time:         {max_time:.1f}s")
        print(f"  Throughput:         {throughput:.0f} tokens/s")
        if rank == world_size - 1 or total_loss > 0:
            print(f"  Avg loss:           {total_loss/num_steps:.4f}")
        print(f"  Microbatches:       {num_microbatches}")

        # Bubble ratio estimate
        S = world_size
        M = num_microbatches
        if schedule == "naive":
            bubble = (S - 1) / S * 100
        else:
            bubble = (S - 1) / (S - 1 + M) * 100
        print(f"  Theoretical bubble: {bubble:.1f}%")

        print(f"  ---")
        for r in range(world_size):
            print(f"  [Rank {r}] peak_mem={all_mem[r].item():.2f}GB  "
                  f"time={all_times[r].item():.1f}s")
        print(f"{'='*60}\n")

    # Clean shutdown: barrier + graceful destroy
    dist.barrier()
    try:
        dist.destroy_process_group()
    except Exception:
        pass  # NCCL shutdown race condition on remote nodes — harmless


if __name__ == "__main__":
    main()
