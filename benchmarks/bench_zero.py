"""
ZeRO Optimizer Sharding Benchmark
===================================
Tests ZeRO Stage 1 and Stage 2 vs standard Adam optimizer.
Shows memory savings from partitioning optimizer states.

Usage (via launcher):
    bash benchmarks/launch_zero.sh <num_nodes> <stage>
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn as nn
import torch.distributed as dist

from minicolossal.gpt2 import GPT2Config, GPT2Model
from minicolossal.data import get_dataloader
from minicolossal.zero_optim import ZeROStage1Optimizer, ZeROStage2Optimizer
from minicolossal.utils import MetricsTracker, StepTimer, print_metrics, print_model_info


def main():
    stage = int(os.environ.get("ZERO_STAGE", "1"))

    # ---- Initialize distributed ----
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    if rank == 0:
        print(f"\n  ZeRO Optimizer Benchmark — Stage {stage}")
        print(f"  World size: {world_size}")

    # ---- Config ----
    cfg = GPT2Config.medium()
    batch_size = 4
    num_steps = 30
    lr = 3e-4

    # ---- Model ----
    model = GPT2Model(cfg).to(device)
    if rank == 0:
        print_model_info(model, cfg)

    # Broadcast model params so all GPUs start identical
    for param in model.parameters():
        dist.broadcast(param.data, src=0)

    # ---- Data ----
    if rank == 0:
        print(f"\n  Loading WikiText-2...")
    dataloader, dataset = get_dataloader(
        split="train", seq_len=cfg.max_seq_len, batch_size=batch_size,
        num_workers=2, distributed=True, rank=rank, world_size=world_size,
    )

    # ---- Optimizer ----
    if stage == 1:
        optimizer = ZeROStage1Optimizer(
            model, lr=lr, weight_decay=0.01, world_size=world_size, rank=rank
        )
    else:
        optimizer = ZeROStage2Optimizer(
            model, lr=lr, weight_decay=0.01, world_size=world_size, rank=rank
        )

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

    # Gather throughput
    local_tp = torch.tensor([results["throughput_tokens_per_s"]], device=device)
    all_tp = [torch.zeros(1, device=device) for _ in range(world_size)]
    dist.all_gather(all_tp, local_tp)

    # Gather peak memory
    local_mem = torch.tensor([results["peak_memory_gb"]], device=device)
    all_mem = [torch.zeros(1, device=device) for _ in range(world_size)]
    dist.all_gather(all_mem, local_mem)

    if rank == 0:
        total_tp = sum(t.item() for t in all_tp)
        print_metrics(results, label=f"ZeRO Stage {stage} ({world_size} GPUs)")
        print(f"  Aggregate throughput: {total_tp:.0f} tokens/s")
        for r in range(world_size):
            print(f"  [Rank {r}] throughput={all_tp[r].item():.0f} tok/s  "
                  f"peak_mem={all_mem[r].item():.2f}GB")

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
