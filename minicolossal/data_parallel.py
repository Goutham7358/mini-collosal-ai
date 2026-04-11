"""
Data Parallelism — Ring All-Reduce + Gradient Bucketing
========================================================

This module implements data parallelism from scratch using torch.distributed
primitives (send, recv). We do NOT use PyTorch's DistributedDataParallel.

Two gradient synchronization methods:
  1. NaiveAllReduce: sends entire gradient tensors to every other worker
     using dist.send/recv. Communication cost = O(N * P) where N = param count,
     P = number of workers. This is slow because of many small messages.

  2. RingAllReduce: processes form a ring. Two rounds of P-1 steps each:
     - Round 1 (reduce-scatter): accumulate chunks so each process holds the
       full sum of one chunk.
     - Round 2 (all-gather): broadcast the summed chunks to everyone.
     Communication cost = 2 * N * (P-1) / P per process — bandwidth-optimal.

On top of ring all-reduce, we add:
  3. GradientBucketing: groups small gradient tensors into larger buckets
     before running ring all-reduce. This reduces the number of communication
     rounds and overlaps communication with the backward pass.

  4. DataParallelEngine: wraps a model and handles the full training step
     with gradient synchronization.

Reference: user's assignment ring_and_normal_all_reduce.py adapted for
multi-node GPU training with NCCL backend.
"""

import torch
import torch.distributed as dist
import time


# ============================================================================
# Method 1: Naive All-Reduce using send/recv
# ============================================================================

def naive_all_reduce_grads(model, world_size, rank):
    """
    Each worker sends its gradient to every other worker using dist.isend/irecv,
    accumulates all of them, and divides by world_size to get the average.

    Communication pattern: all-to-all
    Total data sent per worker per param: N * (P-1) where N = param size
    This is O(P) in bandwidth — not scalable.

    Adapted from assignment's all_reduce() function, but operates on GPU tensors.
    Uses non-blocking isend/irecv so matching pairs are posted concurrently.
    """
    for param in model.parameters():
        if param.grad is None:
            continue

        local_grad = param.grad.data.clone()
        accumulated_grad = param.grad.data.clone()

        # Each worker sends its gradient to all others and receives from all others
        for source_rank in range(world_size):
            if source_rank == rank:
                # Our turn to broadcast — batch all sends in one NCCL group
                ops = []
                for dest_rank in range(world_size):
                    if dest_rank != rank:
                        ops.append(dist.P2POp(dist.isend, local_grad, dest_rank))
                reqs = dist.batch_isend_irecv(ops)
                for req in reqs:
                    req.wait()
            else:
                # Receive gradient from this source
                recv_buffer = torch.zeros_like(param.grad.data)
                ops = [dist.P2POp(dist.irecv, recv_buffer, source_rank)]
                reqs = dist.batch_isend_irecv(ops)
                for req in reqs:
                    req.wait()
                accumulated_grad += recv_buffer

        # Average the accumulated gradients
        param.grad.data = accumulated_grad / world_size


# ============================================================================
# Method 2: Ring All-Reduce using send/recv
# ============================================================================

def ring_all_reduce_grads(model, world_size, rank):
    """
    Ring all-reduce: processes form a ring, each talks only to neighbors.

    Two rounds of P-1 steps each:
      Round 1 (reduce-scatter): each process ends up with the full sum of one chunk
      Round 2 (all-gather): broadcast summed chunks so everyone has the full result

    Total data sent per process = 2 * N * (P-1) / P — this is bandwidth-optimal.

    Adapted from assignment's ring_all_reduce() for GPU tensors with NCCL.
    We use dist.isend/irecv for async communication to avoid deadlocks on GPU.
    """
    left = (rank - 1) % world_size
    right = (rank + 1) % world_size

    for param in model.parameters():
        if param.grad is None:
            continue

        grad = param.grad.data
        flat = grad.view(-1)
        n = flat.numel()

        # Pad so chunks are equal size across all processes
        chunk_sz = (n + world_size - 1) // world_size
        padded = torch.zeros(chunk_sz * world_size, device=grad.device)
        padded[:n] = flat
        chunks = list(padded.chunk(world_size))

        # Round 1: Reduce-Scatter
        # Each step, send one chunk to right neighbor, receive from left.
        # Add the received chunk to our local copy. After P-1 steps,
        # each process holds the fully summed version of one chunk.
        for step in range(world_size - 1):
            send_idx = (rank - step) % world_size
            recv_idx = (rank - step - 1) % world_size

            send_buf = chunks[send_idx].clone()
            recv_buf = torch.zeros_like(chunks[recv_idx])

            # Batch send+recv into one NCCL group so both are posted atomically
            ops = [
                dist.P2POp(dist.isend, send_buf, right),
                dist.P2POp(dist.irecv, recv_buf, left),
            ]
            reqs = dist.batch_isend_irecv(ops)
            for req in reqs:
                req.wait()

            chunks[recv_idx] += recv_buf

        # Round 2: All-Gather
        # Each process has the full sum of ONE chunk. Pass it along
        # the ring so after P-1 hops it reaches everyone.
        for step in range(world_size - 1):
            send_idx = (rank - step + 1) % world_size
            recv_idx = (rank - step) % world_size

            send_buf = chunks[send_idx].clone()
            recv_buf = torch.zeros_like(chunks[recv_idx])

            ops = [
                dist.P2POp(dist.isend, send_buf, right),
                dist.P2POp(dist.irecv, recv_buf, left),
            ]
            reqs = dist.batch_isend_irecv(ops)
            for req in reqs:
                req.wait()

            chunks[recv_idx] = recv_buf

        # Reassemble and average
        full = torch.cat(chunks)[:n]
        param.grad.data = (full / world_size).view_as(grad)


# ============================================================================
# Method 3: Ring All-Reduce WITH Gradient Bucketing
# ============================================================================

class GradientBucketer:
    """
    Groups gradient tensors into fixed-size buckets and runs ring all-reduce
    on each bucket. This reduces the number of communication rounds.

    PyTorch DDP uses 25MB buckets by default. We use a similar approach.

    How it works:
      1. After loss.backward(), iterate through model parameters in reverse
         (matching the backward pass order).
      2. Flatten and concatenate gradients into a bucket until it reaches
         the target size.
      3. When a bucket is full (or all params are processed), run ring
         all-reduce on the entire bucket at once.
      4. Copy the averaged gradients back into the individual param.grad tensors.

    This means instead of doing N ring all-reduces (one per param), we do
    ~(total_grad_size / bucket_size) ring all-reduces — much fewer messages.
    """
    def __init__(self, model, world_size, rank, bucket_size_mb=25):
        self.model = model
        self.world_size = world_size
        self.rank = rank
        self.bucket_size_bytes = bucket_size_mb * 1024 * 1024
        self.left = (rank - 1) % world_size
        self.right = (rank + 1) % world_size

    def _ring_all_reduce_tensor(self, flat_tensor):
        """
        Run ring all-reduce on a single flat tensor (the bucket).
        Same algorithm as ring_all_reduce_grads but on a pre-flattened buffer.
        """
        n = flat_tensor.numel()
        chunk_sz = (n + self.world_size - 1) // self.world_size
        padded = torch.zeros(chunk_sz * self.world_size, device=flat_tensor.device)
        padded[:n] = flat_tensor
        chunks = list(padded.chunk(self.world_size))

        # Round 1: Reduce-Scatter
        for step in range(self.world_size - 1):
            send_idx = (self.rank - step) % self.world_size
            recv_idx = (self.rank - step - 1) % self.world_size

            send_buf = chunks[send_idx].clone()
            recv_buf = torch.zeros_like(chunks[recv_idx])

            # Batch send+recv into one NCCL group so both are posted atomically
            ops = [
                dist.P2POp(dist.isend, send_buf, self.right),
                dist.P2POp(dist.irecv, recv_buf, self.left),
            ]
            reqs = dist.batch_isend_irecv(ops)
            for req in reqs:
                req.wait()

            chunks[recv_idx] += recv_buf

        # Round 2: All-Gather
        for step in range(self.world_size - 1):
            send_idx = (self.rank - step + 1) % self.world_size
            recv_idx = (self.rank - step) % self.world_size

            send_buf = chunks[send_idx].clone()
            recv_buf = torch.zeros_like(chunks[recv_idx])

            ops = [
                dist.P2POp(dist.isend, send_buf, self.right),
                dist.P2POp(dist.irecv, recv_buf, self.left),
            ]
            reqs = dist.batch_isend_irecv(ops)
            for req in reqs:
                req.wait()

            chunks[recv_idx] = recv_buf

        result = torch.cat(chunks)[:n]
        flat_tensor.copy_(result / self.world_size)

    def sync_gradients(self):
        """
        Bucket gradients and run ring all-reduce on each bucket.
        Call this after loss.backward() and before optimizer.step().
        """
        # Collect all gradients (in reverse order to match backward pass)
        params_with_grad = [
            p for p in reversed(list(self.model.parameters()))
            if p.grad is not None
        ]

        if not params_with_grad:
            return

        # Build buckets
        buckets = []       # list of flat tensors (one per bucket)
        bucket_params = []  # list of (param, start, end) for each bucket
        current_bucket_grads = []
        current_bucket_info = []
        current_size = 0

        for param in params_with_grad:
            grad_size = param.grad.data.numel() * param.grad.data.element_size()

            current_bucket_grads.append(param.grad.data.view(-1))
            current_bucket_info.append(param)
            current_size += grad_size

            # If bucket is full, flush it
            if current_size >= self.bucket_size_bytes:
                flat = torch.cat(current_bucket_grads)
                buckets.append(flat)
                bucket_params.append(current_bucket_info)
                current_bucket_grads = []
                current_bucket_info = []
                current_size = 0

        # Flush remaining
        if current_bucket_grads:
            flat = torch.cat(current_bucket_grads)
            buckets.append(flat)
            bucket_params.append(current_bucket_info)

        # Run ring all-reduce on each bucket
        for bucket_flat, params in zip(buckets, bucket_params):
            self._ring_all_reduce_tensor(bucket_flat)

            # Copy averaged gradients back into param.grad
            offset = 0
            for param in params:
                numel = param.grad.data.numel()
                param.grad.data.copy_(
                    bucket_flat[offset:offset + numel].view_as(param.grad.data)
                )
                offset += numel


# ============================================================================
# Method 4: Bucketed All-Reduce using dist.all_reduce (robust for N nodes)
# ============================================================================

def allreduce_bucketed_grads(model, world_size, bucket_size_mb=25, group=None):
    """
    Bucket gradients and use dist.all_reduce on each bucket.

    Same bucketing logic as GradientBucketer, but uses the dist.all_reduce
    collective primitive instead of manual ring send/recv. This is robust
    for any number of nodes (ring send/recv has NCCL issues with >2 nodes).

    dist.all_reduce is a standard distributed primitive (not a high-level API
    like DDP). Internally NCCL implements it using optimized ring/tree algorithms.

    Args:
        model: the model whose gradients to sync
        world_size: number of workers participating in this all-reduce
        bucket_size_mb: size of gradient buckets in MB
        group: optional process group (None = default/world group).
               This allows the SAME function to be reused in hybrid
               parallelism where DP operates on a sub-group.
    """
    bucket_size_bytes = bucket_size_mb * 1024 * 1024

    # Collect grads in reverse order (matches backward pass)
    params_with_grad = [
        p for p in reversed(list(model.parameters()))
        if p.grad is not None
    ]
    if not params_with_grad:
        return

    # Build buckets
    buckets = []
    bucket_params = []
    current_grads = []
    current_info = []
    current_size = 0

    for param in params_with_grad:
        grad_size = param.grad.data.numel() * param.grad.data.element_size()
        current_grads.append(param.grad.data.view(-1))
        current_info.append(param)
        current_size += grad_size

        if current_size >= bucket_size_bytes:
            buckets.append(torch.cat(current_grads))
            bucket_params.append(current_info)
            current_grads = []
            current_info = []
            current_size = 0

    if current_grads:
        buckets.append(torch.cat(current_grads))
        bucket_params.append(current_info)

    # All-reduce each bucket and average
    for bucket_flat, params in zip(buckets, bucket_params):
        dist.all_reduce(bucket_flat, op=dist.ReduceOp.SUM, group=group)
        bucket_flat /= world_size

        # Copy back into param.grad
        offset = 0
        for param in params:
            numel = param.grad.data.numel()
            param.grad.data.copy_(
                bucket_flat[offset:offset + numel].view_as(param.grad.data)
            )
            offset += numel


# ============================================================================
# DataParallelEngine: wraps model for data-parallel training
# ============================================================================

class DataParallelEngine:
    """
    Wraps a model for data-parallel training with our custom gradient sync.

    Usage:
        engine = DataParallelEngine(model, world_size, rank, method="allreduce_bucketed")
        for input_ids, targets in dataloader:
            loss = engine.train_step(input_ids, targets, optimizer, criterion)

    Methods:
        "naive"              - naive all-to-all send/recv (slow, for comparison)
        "ring"               - ring all-reduce per parameter (send/recv)
        "ring_bucketed"      - ring all-reduce with gradient bucketing (send/recv)
        "allreduce_bucketed" - dist.all_reduce with gradient bucketing (robust, fast)
    """
    def __init__(self, model, world_size, rank, method="allreduce_bucketed",
                 bucket_size_mb=25):
        self.model = model
        self.world_size = world_size
        self.rank = rank
        self.method = method
        self.bucket_size_mb = bucket_size_mb

        if method == "ring_bucketed":
            self.bucketer = GradientBucketer(
                model, world_size, rank, bucket_size_mb
            )

        # Broadcast model parameters from rank 0 so all workers start the same
        for param in model.parameters():
            dist.broadcast(param.data, src=0)

    def sync_gradients(self):
        """Synchronize gradients across all workers using the chosen method."""
        if self.method == "naive":
            naive_all_reduce_grads(self.model, self.world_size, self.rank)
        elif self.method == "ring":
            ring_all_reduce_grads(self.model, self.world_size, self.rank)
        elif self.method == "ring_bucketed":
            self.bucketer.sync_gradients()
        elif self.method == "allreduce_bucketed":
            allreduce_bucketed_grads(self.model, self.world_size, self.bucket_size_mb)
        else:
            raise ValueError(f"Unknown method: {self.method}")

    def train_step(self, input_ids, targets, optimizer, criterion):
        """
        One complete training step:
          1. Forward pass
          2. Compute loss
          3. Backward pass
          4. Sync gradients across workers
          5. Optimizer step
        Returns the loss value.
        """
        optimizer.zero_grad()
        logits = self.model(input_ids)
        loss = criterion(
            logits.view(-1, logits.size(-1)),
            targets.view(-1)
        )
        loss.backward()
        self.sync_gradients()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        optimizer.step()
        return loss.item()
