"""
Hybrid Parallelism — Combine multiple parallelism strategies
==============================================================

With 4 GPUs we can combine any two parallelism dimensions:

  1. DP(2) x TP(2):
     - TP groups: [0,1] and [2,3]  (model split within each group)
     - DP groups: [0,2] and [1,3]  (gradient sync across groups)
     Rank layout:
       TP-group-0: rank 0 (TP-slice-0)  rank 1 (TP-slice-1)
       TP-group-1: rank 2 (TP-slice-0)  rank 3 (TP-slice-1)
       DP syncs gradients between rank 0<->2 and rank 1<->3

  2. DP(2) x PP(2):
     - PP pipelines: [0,1] and [2,3]  (stage 0 and stage 1 in each)
     - DP groups: [0,2] and [1,3]  (gradient sync across same-stage ranks)
     Rank layout:
       Pipeline-0: rank 0 (stage 0) -> rank 1 (stage 1)
       Pipeline-1: rank 2 (stage 0) -> rank 3 (stage 1)
       DP syncs gradients between rank 0<->2 (stage 0) and rank 1<->3 (stage 1)

Communication uses torch.distributed process groups to separate
TP/PP traffic from DP traffic.
"""

import torch
import torch.nn as nn
import torch.distributed as dist

# Reuse standalone components instead of duplicating code
# (Following Colossal-AI's pattern: each component exposes a clean interface,
#  hybrid layer just composes them with process sub-groups)
from minicolossal.data_parallel import allreduce_bucketed_grads
from minicolossal.pipeline_parallel import one_f_one_b_forward_backward


# ============================================================================
# Process Group Helpers
# ============================================================================

def create_dp_tp_groups(world_size, tp_size):
    """
    Create TP and DP process groups for DP x TP hybrid.

    With world_size=4, tp_size=2:
      TP groups: [0,1], [2,3]
      DP groups: [0,2], [1,3]

    Returns: (tp_group, dp_group, tp_rank, dp_rank)
    """
    rank = dist.get_rank()
    dp_size = world_size // tp_size

    # TP groups: consecutive ranks
    tp_groups = []
    for i in range(dp_size):
        ranks = list(range(i * tp_size, (i + 1) * tp_size))
        group = dist.new_group(ranks)
        tp_groups.append((ranks, group))

    # DP groups: same position in each TP group
    dp_groups = []
    for i in range(tp_size):
        ranks = [i + j * tp_size for j in range(dp_size)]
        group = dist.new_group(ranks)
        dp_groups.append((ranks, group))

    # Find which groups this rank belongs to
    tp_group_idx = rank // tp_size
    tp_rank = rank % tp_size
    tp_group = tp_groups[tp_group_idx][1]

    dp_group_idx = rank % tp_size
    dp_rank = rank // tp_size
    dp_group = dp_groups[dp_group_idx][1]

    return tp_group, dp_group, tp_rank, dp_rank, tp_size, dp_size


def create_dp_pp_groups(world_size, pp_size):
    """
    Create PP and DP process groups for DP x PP hybrid.

    With world_size=4, pp_size=2:
      PP groups (pipelines): [0,1], [2,3]
      DP groups: [0,2], [1,3]

    Returns: (pp_group, dp_group, pp_rank, dp_rank)
    """
    rank = dist.get_rank()
    dp_size = world_size // pp_size

    # PP groups: consecutive ranks form pipelines
    pp_groups = []
    for i in range(dp_size):
        ranks = list(range(i * pp_size, (i + 1) * pp_size))
        group = dist.new_group(ranks)
        pp_groups.append((ranks, group))

    # DP groups: same stage across pipelines
    dp_groups = []
    for i in range(pp_size):
        ranks = [i + j * pp_size for j in range(dp_size)]
        group = dist.new_group(ranks)
        dp_groups.append((ranks, group))

    # Find which groups this rank belongs to
    pp_group_idx = rank // pp_size
    pp_rank = rank % pp_size  # stage within pipeline
    pp_group = pp_groups[pp_group_idx][1]

    dp_group_idx = rank % pp_size
    dp_rank = rank // pp_size
    dp_group = dp_groups[dp_group_idx][1]

    # Global ranks of PP neighbors in this pipeline
    pp_global_ranks = pp_groups[pp_group_idx][0]

    return pp_group, dp_group, pp_rank, dp_rank, pp_size, dp_size, pp_global_ranks


# ============================================================================
# DP gradient sync for hybrid (reuses standalone data_parallel component)
# ============================================================================

def dp_allreduce_grads(model, dp_group, dp_size, bucket_size_mb=25):
    """
    All-reduce gradients across a DP process group.

    This is a thin wrapper around allreduce_bucketed_grads from data_parallel.py,
    passing the DP sub-group so the SAME bucketing + all-reduce logic is reused.
    No code duplication — follows Colossal-AI's composable design.
    """
    allreduce_bucketed_grads(model, dp_size, bucket_size_mb, group=dp_group)
