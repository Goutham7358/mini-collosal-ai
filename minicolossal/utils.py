"""
Shared utilities for timing, memory tracking, and metrics collection.
Used across all benchmark scripts.
"""

import time
import torch
import math


class MetricsTracker:
    """
    Collects training metrics per step: loss, throughput, memory, timing.
    Used by all benchmark scripts for consistent reporting.
    """
    def __init__(self, device):
        self.device = device
        self.step_times = []
        self.losses = []
        self.tokens_processed = 0
        self.start_time = None

    def start_run(self):
        """Call once before the training loop begins."""
        torch.cuda.synchronize(self.device)
        torch.cuda.reset_peak_memory_stats(self.device)
        self.start_time = time.perf_counter()

    def log_step(self, loss_val, num_tokens, step_time):
        """Log one training step."""
        self.losses.append(loss_val)
        self.tokens_processed += num_tokens
        self.step_times.append(step_time)

    def finish_run(self):
        """Call once after the training loop ends. Returns summary dict."""
        torch.cuda.synchronize(self.device)
        total_time = time.perf_counter() - self.start_time
        peak_mem_gb = torch.cuda.max_memory_allocated(self.device) / 1e9

        avg_loss = sum(self.losses) / len(self.losses) if self.losses else 0
        throughput = self.tokens_processed / total_time if total_time > 0 else 0

        # Perplexity (capped to avoid overflow)
        ppl = math.exp(min(avg_loss, 20))

        # Compute time breakdown
        avg_step = sum(self.step_times) / len(self.step_times) if self.step_times else 0

        return {
            "total_time_s": total_time,
            "avg_loss": avg_loss,
            "final_loss": self.losses[-1] if self.losses else 0,
            "perplexity": ppl,
            "throughput_tokens_per_s": throughput,
            "peak_memory_gb": peak_mem_gb,
            "avg_step_time_s": avg_step,
            "num_steps": len(self.losses),
            "total_tokens": self.tokens_processed,
        }


class StepTimer:
    """Context manager to time a single training step."""
    def __init__(self, device):
        self.device = device
        self.elapsed = 0

    def __enter__(self):
        torch.cuda.synchronize(self.device)
        self.start = time.perf_counter()
        return self

    def __exit__(self, *args):
        torch.cuda.synchronize(self.device)
        self.elapsed = time.perf_counter() - self.start


def print_metrics(results, label="", rank=0):
    """Pretty-print a metrics summary dict. Only prints from rank 0."""
    if rank != 0:
        return
    print(f"\n{'='*60}")
    print(f"  RESULTS: {label}")
    print(f"{'='*60}")
    print(f"  Steps:              {results['num_steps']}")
    print(f"  Total time:         {results['total_time_s']:.1f}s")
    print(f"  Avg step time:      {results['avg_step_time_s']*1000:.1f}ms")
    print(f"  Throughput:         {results['throughput_tokens_per_s']:.0f} tokens/s")
    print(f"  Final loss:         {results['final_loss']:.4f}")
    print(f"  Avg loss:           {results['avg_loss']:.4f}")
    print(f"  Perplexity:         {results['perplexity']:.1f}")
    print(f"  Peak GPU memory:    {results['peak_memory_gb']:.2f} GB")
    print(f"{'='*60}\n")


def print_model_info(model, cfg, rank=0):
    """Print model config and parameter count. Only from rank 0."""
    if rank != 0:
        return
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  Model: GPT-2 ({cfg.n_layers}L, {cfg.hidden_dim}H, {cfg.n_heads}heads)")
    print(f"  Parameters: {n_params:.1f}M")
    print(f"  Seq length: {cfg.max_seq_len}")
