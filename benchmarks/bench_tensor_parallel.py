"""
1D Tensor Parallelism Benchmark
=================================
Tests our tensor-parallel GPT-2 against single-GPU baseline.
Shows communication overhead from all-reduces inside each transformer block.

Usage (via launcher):
    bash benchmarks/launch_tp.sh <num_nodes>
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist

from minicolossal.gpt2 import GPT2Config
from minicolossal.tensor_parallel import TensorParallelGPT2
from minicolossal.data import get_dataloader
from minicolossal.utils import MetricsTracker, StepTimer, print_metrics


def main():
    # ---- Initialize distributed ----
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    if rank == 0:
        print(f"\n  1D Tensor Parallelism Benchmark")
        print(f"  World size: {world_size}")

    # ---- Config ----
    cfg = GPT2Config.medium()
    batch_size = 4
    num_steps = 30
    lr = 3e-4

    # ---- Model (tensor-parallel) ----
    # Use the default process group as the TP group
    model = TensorParallelGPT2(cfg, world_size, rank, tp_group=None).to(device)

    # Sync all replicated params (embeddings, layer norms) from rank 0
    for name, param in model.named_parameters():
        dist.broadcast(param.data, src=0)

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    if rank == 0:
        print(f"  Model: GPT-2 ({cfg.n_layers}L, {cfg.hidden_dim}H, {cfg.n_heads}heads)")
        print(f"  Parameters on this GPU: {n_params:.1f}M")
        print(f"  Seq length: {cfg.max_seq_len}")
        print(f"  Batch size: {batch_size}")

    # ---- Data (same data on all GPUs for TP — NOT data-parallel) ----
    if rank == 0:
        print(f"\n  Loading WikiText-2...")
    dataloader, dataset = get_dataloader(
        split="train", seq_len=cfg.max_seq_len, batch_size=batch_size,
        num_workers=2, distributed=False,  # all GPUs see same data for TP
    )

    # ---- Optimizer ----
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()

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
            optimizer.zero_grad()
            logits = model(input_ids)
            loss = criterion(logits.view(-1, cfg.vocab_size), target_ids.view(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        num_tokens = input_ids.numel()
        tracker.log_step(loss.item(), num_tokens, timer.elapsed)
        step += 1

        if step % 10 == 0:
            tok_per_s = num_tokens / timer.elapsed
            peak_mem = torch.cuda.max_memory_allocated(device) / 1e9
            print(f"  [Rank {rank}] Step {step}/{num_steps}  loss={loss.item():.4f}  "
                  f"tok/s={tok_per_s:.0f}  mem={peak_mem:.2f}GB")

    results = tracker.finish_run()

    # Gather memory from all ranks
    local_mem = torch.tensor([results["peak_memory_gb"]], device=device)
    all_mem = [torch.zeros(1, device=device) for _ in range(world_size)]
    dist.all_gather(all_mem, local_mem)

    if rank == 0:
        print_metrics(results, label=f"1D Tensor Parallel ({world_size}-way)")
        for r in range(world_size):
            print(f"  [Rank {r}] peak_mem={all_mem[r].item():.2f}GB")

        # Communication analysis
        all_reduces_per_block = 2  # 1 in attention out_proj, 1 in MLP fc2
        total_all_reduces = all_reduces_per_block * cfg.n_layers
        print(f"\n  All-reduces per forward pass: {total_all_reduces}")
        print(f"  (2 per block x {cfg.n_layers} blocks)")

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
