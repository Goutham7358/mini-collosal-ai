"""
Pipeline Parallelism — Naive + 1F1B Schedule
==============================================

Splits a model into consecutive stages, one per GPU. Each GPU holds a
subset of the transformer blocks and communicates activations/gradients
with its neighbors via point-to-point send/recv.

Two schedules:
  1. NaivePipelineSchedule: sequential execution, only 1 GPU active at a time.
     Bubble ratio = (S-1)/S where S = number of stages.
     With 4 stages: 75% idle time.

  2. OneFOneBSchedule (1F1B): interleave forward and backward passes.
     Warmup phase:  push microbatches into the pipeline (forward only).
     Steady state:  alternate 1 forward + 1 backward per stage.
     Cooldown:      drain remaining backwards.
     Bubble ratio = (S-1)/(S-1+M) where M = num microbatches.
     With S=4, M=8: ~27% idle (vs 75% naive).

Communication: point-to-point send/recv of activation tensors between
adjacent stages. We use torch.distributed.send() and recv().

Reference: Narayanan et al., "PipeDream: Generalized Pipeline Parallelism
for DNN Training" (2019).
"""

import torch
import torch.nn as nn
import torch.distributed as dist


# ============================================================================
# Pipeline Stage: holds a subset of transformer blocks
# ============================================================================

class PipelineStage(nn.Module):
    """
    Holds a contiguous subset of transformer blocks for one pipeline stage.

    Stage 0 (first stage) also holds the embedding layers.
    Stage S-1 (last stage) also holds the final LayerNorm and lm_head.
    Middle stages only hold transformer blocks.
    """
    def __init__(self, cfg, block_indices, is_first, is_last):
        super().__init__()
        from minicolossal.gpt2 import TransformerBlock

        self.cfg = cfg
        self.is_first = is_first
        self.is_last = is_last
        self.block_indices = block_indices

        # First stage gets embeddings
        if is_first:
            self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.hidden_dim)
            self.pos_emb = nn.Embedding(cfg.max_seq_len, cfg.hidden_dim)
            self.emb_dropout = nn.Dropout(cfg.dropout)

        # Transformer blocks for this stage
        self.blocks = nn.ModuleList([
            TransformerBlock(cfg) for _ in block_indices
        ])

        # Last stage gets final norm + lm_head
        if is_last:
            self.ln_f = nn.LayerNorm(cfg.hidden_dim)
            self.lm_head = nn.Linear(cfg.hidden_dim, cfg.vocab_size, bias=False)
            if is_first:
                # Weight tying (only if first==last, i.e., single stage)
                self.lm_head.weight = self.tok_emb.weight

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x, input_ids=None):
        """
        Args:
            x: hidden states from previous stage, or None if first stage
            input_ids: token ids, only used by first stage
        Returns:
            hidden states (middle stages) or logits (last stage)
        """
        if self.is_first:
            B, T = input_ids.shape
            positions = torch.arange(0, T, device=input_ids.device).unsqueeze(0)
            x = self.tok_emb(input_ids) + self.pos_emb(positions)
            x = self.emb_dropout(x)

        for block in self.blocks:
            x = block(x)

        if self.is_last:
            x = self.ln_f(x)
            x = self.lm_head(x)

        return x


# ============================================================================
# P2P Communication helpers
# ============================================================================

def send_forward(tensor, dst_rank):
    """Send activation tensor to the next stage."""
    dist.send(tensor.contiguous(), dst=dst_rank)


def recv_forward(shape, dtype, device, src_rank):
    """Receive activation tensor from the previous stage."""
    tensor = torch.zeros(shape, dtype=dtype, device=device)
    dist.recv(tensor, src=src_rank)
    return tensor


def send_backward(tensor, dst_rank):
    """Send gradient tensor to the previous stage."""
    dist.send(tensor.contiguous(), dst=dst_rank)


def recv_backward(shape, dtype, device, src_rank):
    """Receive gradient tensor from the next stage."""
    tensor = torch.zeros(shape, dtype=dtype, device=device)
    dist.recv(tensor, src=src_rank)
    return tensor


# ============================================================================
# Helper: split a batch into microbatches
# ============================================================================

def split_into_microbatches(input_ids, target_ids, num_microbatches):
    """Split a batch into equal-sized microbatches."""
    mb_inputs = input_ids.chunk(num_microbatches, dim=0)
    mb_targets = target_ids.chunk(num_microbatches, dim=0)
    return list(zip(mb_inputs, mb_targets))


# ============================================================================
# Helper: create pipeline stages from config
# ============================================================================

def create_pipeline_stage(cfg, num_stages, stage_id, device):
    """
    Partition the model's transformer blocks evenly across stages.
    Returns a PipelineStage for the given stage_id.
    """
    blocks_per_stage = cfg.n_layers // num_stages
    remainder = cfg.n_layers % num_stages

    # Distribute extra blocks to earlier stages
    stage_sizes = []
    for s in range(num_stages):
        size = blocks_per_stage + (1 if s < remainder else 0)
        stage_sizes.append(size)

    # Compute block indices for this stage
    start = sum(stage_sizes[:stage_id])
    end = start + stage_sizes[stage_id]
    block_indices = list(range(start, end))

    is_first = (stage_id == 0)
    is_last = (stage_id == num_stages - 1)

    stage = PipelineStage(cfg, block_indices, is_first, is_last).to(device)
    return stage


# ============================================================================
# Naive Pipeline Schedule
# ============================================================================

def naive_pipeline_forward_backward(
    stage, criterion, microbatches, cfg, rank, world_size, device
):
    """
    Naive (sequential) pipeline schedule.

    All microbatches go through all stages forward, then all backward.
    Only 1 GPU is active at a time. Bubble ratio = (S-1)/S.

    Returns: total loss, list of step times for bubble analysis
    """
    num_mb = len(microbatches)
    is_first = (rank == 0)
    is_last = (rank == world_size - 1)
    hidden_shape = None  # determined at runtime
    total_loss = 0.0

    # Storage for activations needed in backward
    saved_inputs = []   # inputs that require grad for backward
    saved_outputs = []  # outputs for loss computation (last stage only)

    # ---- Forward pass: all microbatches through all stages ----
    for mb_idx in range(num_mb):
        if is_first:
            mb_input, mb_target = microbatches[mb_idx]
            mb_input = mb_input.to(device)
            mb_target = mb_target.to(device)
            output = stage(None, input_ids=mb_input)
        else:
            # Receive activation from previous stage
            if hidden_shape is None:
                # First microbatch: need to figure out the shape
                # Shape is (micro_batch_size, seq_len, hidden_dim)
                micro_bs = microbatches[0][0].shape[0] // 1  # already split
                hidden_shape = (microbatches[0][0].shape[0], cfg.max_seq_len, cfg.hidden_dim)
            recv_act = recv_forward(hidden_shape, torch.float32, device, rank - 1)
            recv_act.requires_grad_(True)
            saved_inputs.append(recv_act)
            output = stage(recv_act)

        # Send to next stage (unless we're last)
        if not is_last:
            send_forward(output, rank + 1)
            saved_outputs.append(output)
        else:
            # Last stage: compute loss
            mb_target = microbatches[mb_idx][1].to(device)
            loss = criterion(output.view(-1, cfg.vocab_size), mb_target.view(-1))
            total_loss += loss.item()
            saved_outputs.append((output, loss))

    # ---- Backward pass: all microbatches in reverse ----
    for mb_idx in range(num_mb):
        if is_last:
            output, loss = saved_outputs[mb_idx]
            loss.backward()
        else:
            output = saved_outputs[mb_idx]
            # Receive gradient from next stage
            grad = recv_backward(output.shape, torch.float32, device, rank + 1)
            output.backward(grad)

        if not is_first and mb_idx < len(saved_inputs):
            # Send gradient to previous stage
            send_backward(saved_inputs[mb_idx].grad, rank - 1)

    return total_loss / num_mb


# ============================================================================
# 1F1B Pipeline Schedule
# ============================================================================

def one_f_one_b_forward_backward(
    stage, criterion, microbatches, cfg, rank, world_size, device,
    prev_rank=None, next_rank=None
):
    """
    One-Forward-One-Backward (1F1B) pipeline schedule.

    Three phases:
      1. Warmup: push microbatches into the pipeline (forward only).
         Number of warmup steps = (world_size - 1 - rank) for each stage.
      2. Steady state: alternate 1 forward + 1 backward.
      3. Cooldown: drain remaining backward passes.

    This keeps more GPUs busy simultaneously than the naive schedule.
    Bubble ratio ≈ (S-1) / (S-1+M) where S = stages, M = microbatches.

    Args:
        stage: PipelineStage module for this rank
        criterion: loss function (used by last stage)
        microbatches: list of (input_ids, target_ids) tuples
        cfg: GPT2Config
        rank: this stage's index within the pipeline (0-indexed)
        world_size: number of stages in the pipeline
        device: CUDA device
        prev_rank: global rank of previous stage (default: rank-1).
                   Set this for hybrid parallelism where pipeline ranks
                   don't map to contiguous global ranks.
        next_rank: global rank of next stage (default: rank+1).
                   Same purpose as prev_rank.
    """
    num_mb = len(microbatches)
    is_first = (rank == 0)
    is_last = (rank == world_size - 1)
    total_loss = 0.0

    # Resolve global ranks for P2P communication
    # In standalone mode: prev=rank-1, next=rank+1
    # In hybrid mode: caller passes the actual global ranks
    _prev = prev_rank if prev_rank is not None else (rank - 1 if not is_first else None)
    _next = next_rank if next_rank is not None else (rank + 1 if not is_last else None)

    # Number of warmup microbatches for this stage
    # Earlier stages (lower rank) do more warmup forwards
    num_warmup = min(world_size - 1 - rank, num_mb)
    num_steady = num_mb - num_warmup

    # Queues for saved tensors
    input_queue = []    # received activations (need grad for backward)
    output_queue = []   # computed outputs (for backward)

    forward_idx = 0     # next microbatch to forward
    backward_idx = 0    # next microbatch to backward

    def do_forward():
        nonlocal forward_idx, total_loss
        mb_idx = forward_idx
        forward_idx += 1

        if is_first:
            mb_input = microbatches[mb_idx][0].to(device)
            output = stage(None, input_ids=mb_input)
        else:
            hidden_shape = (microbatches[0][0].shape[0], cfg.max_seq_len, cfg.hidden_dim)
            recv_act = recv_forward(hidden_shape, torch.float32, device, _prev)
            recv_act.requires_grad_(True)
            input_queue.append(recv_act)
            output = stage(recv_act)

        if not is_last:
            send_forward(output, _next)
            output_queue.append(output)
        else:
            mb_target = microbatches[mb_idx][1].to(device)
            loss = criterion(output.view(-1, cfg.vocab_size), mb_target.view(-1))
            total_loss += loss.item()
            output_queue.append((output, loss))

    def do_backward():
        nonlocal backward_idx

        if is_last:
            output, loss = output_queue.pop(0)
            loss.backward()
        else:
            output = output_queue.pop(0)
            grad = recv_backward(output.shape, torch.float32, device, _next)
            output.backward(grad)

        if not is_first and input_queue:
            inp = input_queue.pop(0)
            if inp.grad is not None:
                send_backward(inp.grad, _prev)

        backward_idx += 1

    # Phase 1: Warmup — only forward passes
    for _ in range(num_warmup):
        do_forward()

    # Phase 2: Steady state — 1 forward + 1 backward
    for _ in range(num_steady):
        do_forward()
        do_backward()

    # Phase 3: Cooldown — drain remaining backwards
    remaining = num_mb - backward_idx
    for _ in range(remaining):
        do_backward()

    return total_loss / max(num_mb, 1) if is_last else 0.0
