"""
Single-GPU Baseline Benchmark
==============================
Trains GPT-2 Medium on WikiText-2 on a single GPU.
This establishes the baseline throughput and memory usage that
all distributed methods will be compared against.

Usage:
    python3 benchmarks/bench_single_gpu.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn as nn
import torch.optim as optim

from minicolossal.gpt2 import GPT2Config, GPT2Model
from minicolossal.data import get_dataloader
from minicolossal.utils import MetricsTracker, StepTimer, print_metrics, print_model_info


def main():
    # ---- Device ----
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n  Device: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # ---- Config ----
    cfg = GPT2Config.medium()
    batch_size = 4
    num_steps = 30
    lr = 3e-4

    # ---- Model ----
    model = GPT2Model(cfg).to(device)
    print_model_info(model, cfg)

    # ---- Data ----
    print("\n  Loading WikiText-2...")
    dataloader, dataset = get_dataloader(
        split="train", seq_len=cfg.max_seq_len, batch_size=batch_size,
        num_workers=2, distributed=False,
    )
    print(f"  Batches available: {len(dataloader)}")

    # ---- Optimizer ----
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()

    # ---- Training ----
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
            print(f"  Step {step}/{num_steps}  loss={loss.item():.4f}  "
                  f"tok/s={tok_per_s:.0f}  mem={peak_mem:.2f}GB")

    results = tracker.finish_run()
    print_metrics(results, label="Single GPU Baseline (GPT-2 Medium)")


if __name__ == "__main__":
    main()
