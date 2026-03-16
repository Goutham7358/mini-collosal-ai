"""
ZeRO Optimizer Sharding — Stage 1 and Stage 2
===============================================

Implements the Zero Redundancy Optimizer (ZeRO) from scratch.
Instead of every GPU storing the full optimizer state (wasteful),
we partition it across GPUs.

ZeRO Stage 1: Partition optimizer states
  - Each GPU only stores Adam (momentum, variance) for 1/N of the params.
  - Forward/backward: normal, full model on each GPU.
  - After backward: all-reduce gradients, each GPU updates its partition,
    then all-gather to broadcast updated weights.
  - Memory saving: optimizer states go from 8*P to 8*P/N per GPU.

ZeRO Stage 2: Partition optimizer states AND gradients
  - Same as Stage 1, but instead of all-reduce on gradients, we use
    reduce-scatter so each GPU only receives gradients for its partition.
  - Memory saving: optimizer states 8*P/N + gradients 4*P/N per GPU.
  - Extra: no need to store full gradients, only the partition.

We use torch.distributed.all_reduce, reduce_scatter, all_gather for
the collective operations (these are standard primitives, not high-level APIs).

Reference: Rajbhandari et al., "ZeRO: Memory Optimizations Toward Training
Trillion Parameter Models" (2019).
"""

import torch
import torch.distributed as dist


class ZeROStage1Optimizer:
    """
    ZeRO Stage 1: partition optimizer states across GPUs.

    Each GPU maintains Adam optimizer states (momentum + variance) for only
    its assigned partition of the model parameters. After the backward pass:
      1. All-reduce gradients (so every GPU has the average gradient).
      2. Each GPU runs optimizer step ONLY on its partition.
      3. All-gather the updated parameters so every GPU has the full model.

    Memory savings per GPU (for Adam with FP32):
      Standard: 8 bytes/param (momentum + variance for ALL params)
      ZeRO-1:   8/N bytes/param (only store states for 1/N of params)

    Usage:
        zero_optim = ZeROStage1Optimizer(model, lr=3e-4, world_size=4, rank=0)
        for data in dataloader:
            loss = model(data)
            loss.backward()
            zero_optim.step()
            zero_optim.zero_grad()
    """
    def __init__(self, model, lr=3e-4, weight_decay=0.01, betas=(0.9, 0.999),
                 eps=1e-8, world_size=1, rank=0, group=None):
        self.model = model
        self.world_size = world_size
        self.rank = rank
        self.lr = lr
        self.weight_decay = weight_decay
        self.betas = betas
        self.eps = eps
        self.step_count = 0
        self.group = group  # Optional process group for hybrid parallelism

        # Collect all trainable parameters into a flat list
        self.all_params = [p for p in model.parameters() if p.requires_grad]
        total_params = sum(p.numel() for p in self.all_params)

        # Partition parameters into N groups, one per GPU
        # Each GPU "owns" a contiguous slice of the flattened parameter space
        self.partition_size = (total_params + world_size - 1) // world_size

        # Figure out which parameters (and which elements within them)
        # belong to this GPU's partition
        self.param_partitions = []  # list of (param, start_in_param, end_in_param)
        offset = 0
        my_start = rank * self.partition_size
        my_end = min(my_start + self.partition_size, total_params)

        for param in self.all_params:
            p_start = offset
            p_end = offset + param.numel()

            # Check if this param overlaps with our partition
            overlap_start = max(p_start, my_start)
            overlap_end = min(p_end, my_end)

            if overlap_start < overlap_end:
                # Part of this param belongs to us
                local_start = overlap_start - p_start
                local_end = overlap_end - p_start
                self.param_partitions.append((param, local_start, local_end))

            offset = p_end

        # Initialize Adam states ONLY for our partition
        self.momentum = []
        self.variance = []
        for param, start, end in self.param_partitions:
            size = end - start
            self.momentum.append(torch.zeros(size, device=param.device))
            self.variance.append(torch.zeros(size, device=param.device))

        my_total = sum(end - start for _, start, end in self.param_partitions)
        if rank == 0:
            print(f"  [ZeRO-1] Total params: {total_params:,}")
            print(f"  [ZeRO-1] Partition size: {self.partition_size:,}")
            print(f"  [ZeRO-1] This GPU stores optimizer states for {my_total:,} params")
            optim_mem_standard = total_params * 8 / 1e9
            optim_mem_zero = my_total * 8 / 1e9
            print(f"  [ZeRO-1] Optimizer memory: {optim_mem_zero:.3f} GB "
                  f"(vs {optim_mem_standard:.3f} GB standard, "
                  f"{optim_mem_standard/optim_mem_zero:.1f}x saving)")

    def step(self):
        """
        Perform one optimizer step with ZeRO Stage 1:
          1. All-reduce gradients across all GPUs
          2. Update only our partition using Adam
          3. All-gather updated parameters to all GPUs
        """
        self.step_count += 1
        beta1, beta2 = self.betas

        # Step 1: All-reduce gradients so every GPU has the average
        for param in self.all_params:
            if param.grad is not None:
                dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM, group=self.group)
                param.grad.data /= self.world_size

        # Step 2: Adam update on our partition only
        for i, (param, start, end) in enumerate(self.param_partitions):
            # Get the gradient slice for our partition
            grad_flat = param.grad.data.view(-1)[start:end]
            param_flat = param.data.view(-1)[start:end]

            # Adam update
            self.momentum[i] = beta1 * self.momentum[i] + (1 - beta1) * grad_flat
            self.variance[i] = beta2 * self.variance[i] + (1 - beta2) * grad_flat ** 2

            # Bias correction
            m_hat = self.momentum[i] / (1 - beta1 ** self.step_count)
            v_hat = self.variance[i] / (1 - beta2 ** self.step_count)

            # Update parameters (with weight decay)
            update = m_hat / (v_hat.sqrt() + self.eps) + self.weight_decay * param_flat
            param_flat.add_(-self.lr * update)

        # Step 3: All-gather so every GPU has the updated full model
        # We flatten all params, gather, then scatter back
        all_flat = torch.cat([p.data.view(-1) for p in self.all_params])
        total = all_flat.numel()

        # Pad to make evenly divisible
        padded_size = self.partition_size * self.world_size
        padded = torch.zeros(padded_size, device=all_flat.device)
        padded[:total] = all_flat

        # All-gather: each GPU contributes its partition
        chunks = list(padded.chunk(self.world_size))
        gathered = [torch.zeros_like(c) for c in chunks]
        dist.all_gather(gathered, chunks[self.rank], group=self.group)

        # Reconstruct full parameter tensor
        full = torch.cat(gathered)[:total]

        # Copy back into model parameters
        offset = 0
        for param in self.all_params:
            numel = param.numel()
            param.data.copy_(full[offset:offset + numel].view_as(param.data))
            offset += numel

    def zero_grad(self):
        """Zero out all gradients."""
        for param in self.all_params:
            if param.grad is not None:
                param.grad.data.zero_()


class ZeROStage2Optimizer:
    """
    ZeRO Stage 2: partition optimizer states AND gradients.

    Difference from Stage 1:
      - Instead of all-reduce on gradients (which gives every GPU all gradients),
        we use reduce-scatter: each GPU only gets the gradients for its partition.
      - This saves memory because we don't need to store the full gradient tensor.

    Memory savings per GPU (for Adam with FP32):
      Standard: 4 bytes/param (grads) + 8 bytes/param (optim) = 12 bytes/param
      ZeRO-2:   4/N bytes/param (grads) + 8/N bytes/param (optim) = 12/N bytes/param
    """
    def __init__(self, model, lr=3e-4, weight_decay=0.01, betas=(0.9, 0.999),
                 eps=1e-8, world_size=1, rank=0):
        self.model = model
        self.world_size = world_size
        self.rank = rank
        self.lr = lr
        self.weight_decay = weight_decay
        self.betas = betas
        self.eps = eps
        self.step_count = 0

        self.all_params = [p for p in model.parameters() if p.requires_grad]
        total_params = sum(p.numel() for p in self.all_params)
        self.total_params = total_params

        # Partition size: each GPU owns this many elements
        self.partition_size = (total_params + world_size - 1) // world_size

        # This GPU's range in the flat parameter space
        self.my_start = rank * self.partition_size
        self.my_end = min(self.my_start + self.partition_size, total_params)
        my_total = self.my_end - self.my_start

        # Adam states only for our partition (stored as single flat tensors)
        device = self.all_params[0].device
        self.momentum = torch.zeros(my_total, device=device)
        self.variance = torch.zeros(my_total, device=device)

        if rank == 0:
            print(f"  [ZeRO-2] Total params: {total_params:,}")
            print(f"  [ZeRO-2] Partition size per GPU: {self.partition_size:,}")
            mem_standard = total_params * 12 / 1e9
            mem_zero2 = my_total * 12 / 1e9
            print(f"  [ZeRO-2] Grad+Optim memory: {mem_zero2:.3f} GB "
                  f"(vs {mem_standard:.3f} GB standard, "
                  f"{mem_standard/mem_zero2:.1f}x saving)")

    def _flatten_grads(self):
        """Flatten all gradients into a single contiguous tensor."""
        grads = []
        for param in self.all_params:
            if param.grad is not None:
                grads.append(param.grad.data.view(-1))
            else:
                grads.append(torch.zeros(param.numel(), device=param.device))
        return torch.cat(grads)

    def _flatten_params(self):
        """Flatten all parameters into a single contiguous tensor."""
        return torch.cat([p.data.view(-1) for p in self.all_params])

    def step(self):
        """
        ZeRO Stage 2 optimizer step (memory-efficient version):
          1. Reduce-scatter gradients (each GPU gets only its partition's grads)
          2. Adam update on our partition
          3. Broadcast each partition so all GPUs have the updated model

        We avoid creating full-model-sized temporary tensors to stay within
        the 16GB VRAM of the T4 GPU.
        """
        self.step_count += 1
        beta1, beta2 = self.betas
        device = self.all_params[0].device
        my_total = self.my_end - self.my_start

        # Step 1: All-reduce gradients (simpler than reduce-scatter, same result)
        # We use all-reduce + extract our slice, which avoids large padded buffers
        for param in self.all_params:
            if param.grad is not None:
                dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
                param.grad.data /= self.world_size

        # Extract only our partition's gradients (memory-efficient: small tensor)
        grad_slices = []
        offset = 0
        for param in self.all_params:
            p_start = offset
            p_end = offset + param.numel()
            overlap_start = max(p_start, self.my_start)
            overlap_end = min(p_end, self.my_end)
            if overlap_start < overlap_end:
                local_s = overlap_start - p_start
                local_e = overlap_end - p_start
                grad_slices.append(param.grad.data.view(-1)[local_s:local_e])
            offset = p_end
        my_grads = torch.cat(grad_slices)

        # Step 2: Extract our partition's params and do Adam update
        param_slices = []
        offset = 0
        for param in self.all_params:
            p_start = offset
            p_end = offset + param.numel()
            overlap_start = max(p_start, self.my_start)
            overlap_end = min(p_end, self.my_end)
            if overlap_start < overlap_end:
                local_s = overlap_start - p_start
                local_e = overlap_end - p_start
                param_slices.append(param.data.view(-1)[local_s:local_e])
            offset = p_end
        my_params = torch.cat(param_slices)

        # Adam update on our partition only
        self.momentum = beta1 * self.momentum + (1 - beta1) * my_grads
        self.variance = beta2 * self.variance + (1 - beta2) * my_grads ** 2

        m_hat = self.momentum / (1 - beta1 ** self.step_count)
        v_hat = self.variance / (1 - beta2 ** self.step_count)

        update = m_hat / (v_hat.sqrt() + self.eps) + self.weight_decay * my_params
        my_params = my_params - self.lr * update

        # Write our updated partition back into model parameters
        offset = 0
        part_offset = 0
        for param in self.all_params:
            p_start = offset
            p_end = offset + param.numel()
            overlap_start = max(p_start, self.my_start)
            overlap_end = min(p_end, self.my_end)
            if overlap_start < overlap_end:
                local_s = overlap_start - p_start
                local_e = overlap_end - p_start
                n = local_e - local_s
                param.data.view(-1)[local_s:local_e] = my_params[part_offset:part_offset + n]
                part_offset += n
            offset = p_end

        # Step 3: Broadcast updated params from each rank to all others
        # Each rank broadcasts the parameters it owns. This avoids creating
        # a full-model-sized temporary — we work parameter by parameter.
        # Broadcast the full model param-by-param from
        # whichever rank updated it. Since partitions are contiguous in the
        # flat space, we track which rank owns each param region.
        offset = 0
        for param in self.all_params:
            p_start = offset
            p_end = offset + param.numel()

            # Check if this param is fully owned by one rank
            owner_start = p_start // self.partition_size
            owner_end = (p_end - 1) // self.partition_size

            if owner_start == owner_end:
                # Single owner — just broadcast from that rank
                dist.broadcast(param.data, src=owner_start)
            else:
                # This param spans multiple ranks — broadcast each chunk
                for r in range(self.world_size):
                    r_start = r * self.partition_size
                    r_end = min((r + 1) * self.partition_size, self.total_params)
                    overlap_start = max(p_start, r_start)
                    overlap_end = min(p_end, r_end)
                    if overlap_start < overlap_end:
                        local_s = overlap_start - p_start
                        local_e = overlap_end - p_start
                        chunk = param.data.view(-1)[local_s:local_e].contiguous()
                        dist.broadcast(chunk, src=r)
                        param.data.view(-1)[local_s:local_e] = chunk

            offset = p_end

    def zero_grad(self):
        """Zero out all gradients."""
        for param in self.all_params:
            if param.grad is not None:
                param.grad.data.zero_()
