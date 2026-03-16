"""
Unified 3D Parallelism Benchmark — MiniColossal Plugin
========================================================
Uses MiniColossalPlugin to run ANY parallelism configuration with a
single script. Just pass --tp_size, --pp_size, --zero_stage.

    dp_size = world_size // (tp_size * pp_size)

Examples (4 GPUs):
  Pure DP:         --tp_size 1 --pp_size 1 --zero_stage 0
  DP + ZeRO-1:    --tp_size 1 --pp_size 1 --zero_stage 1
  Pure TP:         --tp_size 4 --pp_size 1
  Pure PP:         --tp_size 1 --pp_size 4
  DP(2) x TP(2):  --tp_size 2 --pp_size 1
  DP(2) x PP(2):  --tp_size 1 --pp_size 2
  PP(2) + ZeRO-1: --tp_size 1 --pp_size 2 --zero_stage 1

Usage: torchrun --nnodes=4 --nproc_per_node=1 ... bench_unified.py --tp_size 2 --pp_size 1
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import argparse
import time
import socket
import torch
import torch.nn as nn
import torch.distributed as dist

from minicolossal.gpt2 import GPT2Config
from minicolossal.plugin import MiniColossalPlugin
from minicolossal.data import get_dataloader


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tp_size", type=int, default=1)
    parser.add_argument("--pp_size", type=int, default=1)
    parser.add_argument("--zero_stage", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_microbatches", type=int, default=8)
    parser.add_argument("--num_steps", type=int, default=30)
    args = parser.parse_args()

    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    hostname = socket.gethostname()
    ip_addr = socket.gethostbyname(hostname)

    # ---- Create plugin ----
    plugin = MiniColossalPlugin(
        tp_size=args.tp_size,
        pp_size=args.pp_size,
        zero_stage=args.zero_stage,
        num_microbatches=args.num_microbatches,
    )

    config_name = plugin.info_string()
    print(f"  [Rank {rank}] {hostname} ({ip_addr})  "
          f"dp_rank={plugin.dp_rank}  pp_rank={plugin.pp_rank}  tp_rank={plugin.tp_rank}")
    dist.barrier()

    if rank == 0:
        print(f"\n  MiniColossal Unified Benchmark: {config_name}")
        print(f"  World size: {world_size}")
        print(f"  DP={plugin.dp_size}  PP={plugin.pp_size}  TP={plugin.tp_size}  "
              f"ZeRO={args.zero_stage}")

    # ---- Config ----
    # Use GPT-2 Large for pure PP (4 stages), Medium for everything else
    if args.pp_size == world_size and args.tp_size == 1:
        cfg = GPT2Config.large()
        model_name = "GPT-2 Large"
    else:
        cfg = GPT2Config.medium()
        model_name = "GPT-2 Medium"

    batch_size = args.batch_size
    # For PP, batch_size is the total fed to each pipeline (split into microbatches)
    if args.pp_size > 1:
        batch_size = args.batch_size * args.num_microbatches // args.num_microbatches
        batch_size = max(args.batch_size, args.num_microbatches)

    if rank == 0:
        print(f"  Model: {model_name} ({cfg.n_layers}L)")
        print(f"  Batch size: {batch_size}  Steps: {args.num_steps}")

    # ---- Configure model + optimizer ----
    model, optimizer = plugin.configure(cfg, lr=3e-4, device=device)

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    if rank == 0:
        print(f"  Params per GPU: {n_params:.1f}M")

    criterion = nn.CrossEntropyLoss()

    # ---- Data ----
    torch.manual_seed(42 + plugin.dp_rank)
    dl, _ = get_dataloader(
        split="train", seq_len=cfg.max_seq_len, batch_size=batch_size,
        num_workers=0, distributed=False,
    )
    dataloader = iter(dl)

    # ---- Training ----
    dist.barrier()
    if rank == 0:
        print(f"\n  Training for {args.num_steps} steps...\n")

    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    start_time = time.perf_counter()

    model.train()
    total_loss = 0.0

    for step in range(args.num_steps):
        try:
            input_ids, target_ids = next(dataloader)
        except StopIteration:
            torch.manual_seed(42 + plugin.dp_rank + step * 100)
            dl, _ = get_dataloader(
                split="train", seq_len=cfg.max_seq_len, batch_size=batch_size,
                num_workers=0, distributed=False,
            )
            dataloader = iter(dl)
            input_ids, target_ids = next(dataloader)

        loss_val = plugin.train_step(model, optimizer, (input_ids, target_ids),
                                     criterion, cfg)

        if plugin.is_last_pp_stage():
            total_loss += loss_val

        if (step + 1) % 10 == 0:
            peak_mem = torch.cuda.max_memory_allocated(device) / 1e9
            if plugin.is_last_pp_stage():
                print(f"  [Rank {rank}] Step {step+1}/{args.num_steps}  "
                      f"loss={loss_val:.4f}  mem={peak_mem:.2f}GB")
            else:
                print(f"  [Rank {rank}] Step {step+1}/{args.num_steps}  "
                      f"mem={peak_mem:.2f}GB")

    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start_time

    # ---- Results ----
    dist.barrier()
    peak_mem = torch.cuda.max_memory_allocated(device) / 1e9
    tokens = args.num_steps * batch_size * cfg.max_seq_len
    tps = tokens / elapsed

    local_stats = torch.tensor([tps, peak_mem, elapsed], device=device)
    all_stats = [torch.zeros(3, device=device) for _ in range(world_size)]
    dist.all_gather(all_stats, local_stats)

    if rank == 0:
        max_time = max(s[2].item() for s in all_stats)

        # Aggregate throughput depends on parallelism type
        # DP replicas multiply throughput; PP/TP don't (they share the work)
        if plugin.dp_size > 1:
            agg_tps = plugin.dp_size * tokens / max_time
        else:
            agg_tps = tokens / max_time

        print(f"\n{'='*60}")
        print(f"  RESULTS: {config_name}")
        print(f"{'='*60}")
        print(f"  Steps:              {args.num_steps}")
        print(f"  Total time:         {max_time:.1f}s")
        print(f"  Aggregate throughput: {agg_tps:.0f} tokens/s")
        if plugin.is_last_pp_stage() and total_loss > 0:
            print(f"  Avg loss:           {total_loss / args.num_steps:.4f}")
        if plugin.pp_size > 1:
            S = plugin.pp_size
            M = args.num_microbatches
            bubble = (S - 1) / (S - 1 + M) * 100
            print(f"  Microbatches:       {M}")
            print(f"  Theoretical bubble: {bubble:.1f}%")
        if args.zero_stage > 0:
            print(f"  ZeRO-{args.zero_stage}:            {plugin.dp_size}-way partitioned")
        print(f"  ---")
        for r in range(world_size):
            s = all_stats[r]
            print(f"  [Rank {r}] throughput={s[0].item():.0f} tok/s  "
                  f"peak_mem={s[1].item():.2f}GB  time={s[2].item():.1f}s")
        print(f"{'='*60}\n")

    dist.barrier()
    try:
        plugin.destroy()
        dist.destroy_process_group()
    except Exception:
        pass


if __name__ == "__main__":
    main()
