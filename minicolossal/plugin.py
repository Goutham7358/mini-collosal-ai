"""
Unified 3D Hybrid Parallelism Plugin — Colossal-AI style
==========================================================

A single plugin that composes DP, TP, PP, and ZeRO into a unified
training system. The user just specifies tp_size, pp_size, and
zero_stage — DP size is auto-calculated.

    dp_size = world_size // (tp_size * pp_size)

This follows Colossal-AI's HybridParallelPlugin design:
  - ProcessGroupMesh creates a 3D grid of ranks
  - Process groups are extracted along each axis (DP, PP, TP)
  - Standalone components are composed with group= parameters
  - No code duplication — everything reuses the building blocks

Supported configurations (any combo where tp_size * pp_size divides world_size):
  tp_size=1, pp_size=1, zero=0  → Pure DP (all-reduce)
  tp_size=1, pp_size=1, zero=1  → DP + ZeRO-1
  tp_size=W, pp_size=1, zero=0  → Pure TP
  tp_size=1, pp_size=W, zero=0  → Pure PP
  tp_size=2, pp_size=1, zero=0  → DP(W/2) x TP(2)
  tp_size=1, pp_size=2, zero=0  → DP(W/2) x PP(2)
  tp_size=1, pp_size=2, zero=1  → DP(W/2) x PP(2) + ZeRO-1

Constraint (from Colossal-AI): zero_stage <= 1 when pp_size > 1.
"""

import torch
import torch.nn as nn
import torch.distributed as dist
from typing import List, Tuple, Optional

from minicolossal.data_parallel import allreduce_bucketed_grads
from minicolossal.pipeline_parallel import (
    create_pipeline_stage,
    split_into_microbatches,
    one_f_one_b_forward_backward,
)
from minicolossal.tensor_parallel import TensorParallelGPT2, TensorParallelT5
from minicolossal.zero_optim import ZeROStage1Optimizer
from minicolossal.gpt2 import GPT2Config, GPT2Model
from minicolossal.t5 import T5Config, T5Model, create_t5_pipeline_stage


# ============================================================================
# ProcessGroupMesh — simplified version of Colossal-AI's mesh
# ============================================================================

class ProcessGroupMesh:
    """
    N-dimensional process group mesh.

    Maps world_size ranks onto a grid of shape (dp_size, pp_size, tp_size).
    Creates process groups along each axis so that each dimension's
    communication is isolated.

    Example with 4 GPUs, shape (2, 2, 1):
      Rank 0 → coord (0, 0, 0)  |  Rank 1 → coord (0, 1, 0)
      Rank 2 → coord (1, 0, 0)  |  Rank 3 → coord (1, 1, 0)

      DP groups (axis 0): [0,2], [1,3]  — same pp & tp position
      PP groups (axis 1): [0,1], [2,3]  — same dp & tp position
    """

    def __init__(self, *shape):
        assert dist.is_initialized(), "Call dist.init_process_group first."
        world_size = dist.get_world_size()
        total = 1
        for s in shape:
            total *= s
        assert total == world_size, \
            f"Mesh shape {shape} (product={total}) != world_size ({world_size})"

        self.shape = shape
        self.ndim = len(shape)
        self.rank = dist.get_rank()

        # Convert rank to N-dimensional coordinate
        # Using row-major (C) order: last dimension changes fastest
        self.coord = self._unravel(self.rank, shape)

        # Cache: tuple of ranks → ProcessGroup
        self._groups = {}

    @staticmethod
    def _unravel(rank, shape):
        """Convert flat rank to N-dim coordinate (row-major order)."""
        coord = []
        for dim_size in reversed(shape):
            coord.append(rank % dim_size)
            rank //= dim_size
        return tuple(reversed(coord))

    @staticmethod
    def _ravel(coord, shape):
        """Convert N-dim coordinate to flat rank (row-major order)."""
        rank = 0
        multiplier = 1
        for c, s in zip(reversed(coord), reversed(shape)):
            rank += c * multiplier
            multiplier *= s
        return rank

    def get_group_along_axis(self, axis):
        """
        Create/get the process group along the given axis for this rank.

        "Along axis" means: vary the index on `axis`, keep all other
        dimensions fixed at this rank's coordinate. This gives the set
        of ranks that communicate along that dimension.

        Example: shape=(2,2,1), axis=0 (DP), rank 1 at coord (0,1,0)
          → vary dim 0: coords (0,1,0) and (1,1,0) → ranks [1, 3]
        """
        # Generate all ranks in this group by iterating over the axis dimension
        group_ranks = []
        for i in range(self.shape[axis]):
            coord = list(self.coord)
            coord[axis] = i
            group_ranks.append(self._ravel(tuple(coord), self.shape))
        group_ranks = sorted(group_ranks)

        key = tuple(group_ranks)
        if key not in self._groups:
            # All ranks must call new_group with the same ranks list
            # So we iterate over ALL possible groups along this axis
            self._create_all_groups_along_axis(axis)

        return self._groups[key]

    def _create_all_groups_along_axis(self, axis):
        """Create ALL groups along an axis (required by dist.new_group)."""
        import itertools

        # Generate all base coordinates (with axis dimension = 0)
        other_dims = [range(self.shape[d]) for d in range(self.ndim)]
        other_dims[axis] = [0]  # placeholder

        for base in itertools.product(*other_dims):
            group_ranks = []
            for i in range(self.shape[axis]):
                coord = list(base)
                coord[axis] = i
                group_ranks.append(self._ravel(tuple(coord), self.shape))
            group_ranks = sorted(group_ranks)
            key = tuple(group_ranks)
            if key not in self._groups:
                self._groups[key] = dist.new_group(group_ranks)

    def get_ranks_along_axis(self, axis):
        """Get the list of global ranks along the given axis for this rank."""
        ranks = []
        for i in range(self.shape[axis]):
            coord = list(self.coord)
            coord[axis] = i
            ranks.append(self._ravel(tuple(coord), self.shape))
        return ranks

    def destroy(self):
        """Clean up all process groups."""
        for group in self._groups.values():
            try:
                dist.destroy_process_group(group)
            except Exception:
                pass


# ============================================================================
# Axis constants (for readability)
# ============================================================================

DP_AXIS = 0
PP_AXIS = 1
TP_AXIS = 2


# ============================================================================
# MiniColossalPlugin — unified 3D hybrid parallelism
# ============================================================================

class MiniColossalPlugin:
    """
    Unified 3D parallel training plugin.

    Composes our standalone building blocks (data_parallel, tensor_parallel,
    pipeline_parallel, zero_optim) into a single coherent system using
    process group sub-groups.

    Usage:
        plugin = MiniColossalPlugin(tp_size=2, pp_size=1, zero_stage=0)
        model, optimizer = plugin.configure(cfg, lr=3e-4, device=device)
        for data in dataloader:
            loss = plugin.train_step(model, optimizer, data, criterion)
    """

    def __init__(self, tp_size=1, pp_size=1, zero_stage=0, num_microbatches=8,
                 bad_placement=False, worst_placement=False):
        self.world_size = dist.get_world_size()
        self.global_rank = dist.get_rank()
        self.tp_size = tp_size
        self.pp_size = pp_size
        self.zero_stage = zero_stage
        self.num_microbatches = num_microbatches
        self.bad_placement = bad_placement
        self.worst_placement = worst_placement

        # Validate
        assert self.world_size % (tp_size * pp_size) == 0, \
            f"world_size ({self.world_size}) must be divisible by tp_size*pp_size ({tp_size*pp_size})"
        self.dp_size = self.world_size // (tp_size * pp_size)

        # Colossal-AI constraint: ZeRO-2 forbidden with PP
        if pp_size > 1:
            assert zero_stage <= 1, \
                "ZeRO stage must be 0 or 1 when using pipeline parallelism " \
                "(ZeRO-2 has prohibitive gradient sync costs with PP)"

        # Create 3D mesh.
        # Good placement:  (pp, dp, tp) — PP is first axis (slowest = inter-node)
        #                                 DP+TP are fast axes (intra-node PCIe)
        # Bad placement:   (dp, pp, tp) — DP is first axis (slowest = inter-node)
        #                                 forcing heavier DP traffic over slow TCP
        # Worst placement: (tp, pp, dp) — TP is first axis (slowest = inter-node)
        #                                 TP all-reduces forced over slow TCP
        if worst_placement:
            self.mesh = ProcessGroupMesh(tp_size, pp_size, self.dp_size)
            tp_axis, pp_axis, dp_axis = 0, 1, 2
        elif bad_placement:
            self.mesh = ProcessGroupMesh(self.dp_size, pp_size, tp_size)
            dp_axis, pp_axis, tp_axis = 0, 1, 2
        else:
            self.mesh = ProcessGroupMesh(pp_size, self.dp_size, tp_size)
            dp_axis, pp_axis, tp_axis = 1, 0, 2

        self._dp_axis = dp_axis
        self._pp_axis = pp_axis
        self._tp_axis = tp_axis

        # Extract process groups along each axis
        self.dp_group = self.mesh.get_group_along_axis(dp_axis) if self.dp_size > 1 else None
        self.pp_group = self.mesh.get_group_along_axis(pp_axis) if pp_size > 1 else None
        self.tp_group = self.mesh.get_group_along_axis(tp_axis) if tp_size > 1 else None

        # Per-axis ranks for this process
        self.dp_rank = self.mesh.coord[dp_axis]
        self.pp_rank = self.mesh.coord[pp_axis]
        self.tp_rank = self.mesh.coord[tp_axis]

        # PP neighbor global ranks (for send/recv)
        if pp_size > 1:
            pp_global_ranks = self.mesh.get_ranks_along_axis(self._pp_axis)
            self.pp_prev = pp_global_ranks[self.pp_rank - 1] if self.pp_rank > 0 else None
            self.pp_next = pp_global_ranks[self.pp_rank + 1] if self.pp_rank < pp_size - 1 else None
        else:
            self.pp_prev = None
            self.pp_next = None

    def configure(self, cfg, lr=3e-4, weight_decay=0.01, device=None):
        """
        Build model and optimizer based on the parallelism configuration.

        Returns: (model_or_stage, optimizer)

        The returned model depends on the config:
          - PP enabled → PipelineStage (subset of layers)
          - TP enabled → TensorParallelGPT2 (sharded layers)
          - Neither    → Full GPT2Model
        """
        # --- Build model ---
        is_t5 = getattr(cfg, 'is_t5', False)
        if self.pp_size > 1:
            # Pipeline parallelism: each rank holds a subset of layers
            if is_t5:
                model = create_t5_pipeline_stage(cfg, self.pp_size, self.pp_rank, device)
            else:
                model = create_pipeline_stage(cfg, self.pp_size, self.pp_rank, device)
        elif self.tp_size > 1:
            # Tensor parallelism: each rank holds sharded layers
            if is_t5:
                model = TensorParallelT5(
                    cfg, self.tp_size, self.tp_rank, self.tp_group
                ).to(device)
            else:
                model = TensorParallelGPT2(
                    cfg, self.tp_size, self.tp_rank, self.tp_group
                ).to(device)
        else:
            # Full model (DP only)
            if is_t5:
                model = T5Model(cfg).to(device)
            else:
                model = GPT2Model(cfg).to(device)

        # Sync parameters within DP group so replicas start identical
        if self.dp_size > 1:
            for param in model.parameters():
                # Broadcast from dp_rank=0 within each DP sub-group
                src_coord = list(self.mesh.coord)
                src_coord[self._dp_axis] = 0
                src_global = self.mesh._ravel(tuple(src_coord), self.mesh.shape)
                dist.broadcast(param.data, src=src_global, group=self.dp_group)

        # --- Build optimizer ---
        if self.zero_stage >= 1:
            optimizer = ZeROStage1Optimizer(
                model, lr=lr, weight_decay=weight_decay,
                world_size=self.dp_size, rank=self.dp_rank,
                group=self.dp_group,
            )
        else:
            optimizer = torch.optim.AdamW(
                model.parameters(), lr=lr, weight_decay=weight_decay
            )

        return model, optimizer

    def train_step(self, model, optimizer, batch, criterion, cfg):
        """
        Execute one training step with the configured parallelism.

        Args:
            model: model or pipeline stage from configure()
            optimizer: optimizer from configure()
            batch: (input_ids, target_ids) tensors
            criterion: loss function
            cfg: GPT2Config

        Returns:
            loss_value (float): loss on last PP stage, 0.0 on others
        """
        input_ids, target_ids = batch
        device = next(model.parameters()).device

        if isinstance(optimizer, ZeROStage1Optimizer):
            optimizer.zero_grad()
        else:
            optimizer.zero_grad()

        loss_val = 0.0

        is_t5 = getattr(cfg, 'is_t5', False)

        if self.pp_size > 1:
            # --- Pipeline parallelism path ---
            microbatches = split_into_microbatches(
                input_ids, target_ids, self.num_microbatches
            )
            loss_val = one_f_one_b_forward_backward(
                model, criterion, microbatches, cfg,
                rank=self.pp_rank, world_size=self.pp_size, device=device,
                prev_rank=self.pp_prev, next_rank=self.pp_next,
            )
        else:
            # --- Non-pipeline path (DP, TP, or DP x TP) ---
            input_ids = input_ids.to(device)
            target_ids = target_ids.to(device)
            logits = model(input_ids)
            if is_t5:
                # T5 returns logits for decoder half only: (B, T//2, vocab)
                dec_targets = target_ids[:, target_ids.shape[1] // 2:].contiguous()
                loss = criterion(logits.view(-1, cfg.vocab_size), dec_targets.view(-1))
            else:
                loss = criterion(logits.view(-1, cfg.vocab_size), target_ids.view(-1))
            loss.backward()
            loss_val = loss.item()

        # --- DP gradient sync ---
        if self.dp_size > 1 and self.zero_stage == 0:
            # Plain all-reduce across DP group
            allreduce_bucketed_grads(model, self.dp_size, group=self.dp_group)

        # --- Optimizer step ---
        # ZeRO-1 handles its own all-reduce + all-gather internally
        if isinstance(optimizer, ZeROStage1Optimizer):
            optimizer.step()
        else:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        return loss_val

    def info_string(self):
        """Pretty description of the current configuration."""
        parts = []
        if self.dp_size > 1:
            parts.append(f"DP({self.dp_size})")
        if self.pp_size > 1:
            parts.append(f"PP({self.pp_size})")
        if self.tp_size > 1:
            parts.append(f"TP({self.tp_size})")
        if self.zero_stage > 0:
            parts.append(f"ZeRO-{self.zero_stage}")
        if not parts:
            parts.append("Single GPU")
        return " x ".join(parts)

    def is_last_pp_stage(self):
        """Whether this rank is the last pipeline stage (has the loss)."""
        return self.pp_rank == self.pp_size - 1

    def destroy(self):
        """Clean up process groups."""
        self.mesh.destroy()
