"""
1D Tensor Parallelism — Megatron-LM style
==========================================

Splits individual layers across GPUs so each GPU computes a portion of
the layer's output. This reduces per-GPU memory for large layers.

Two fundamental building blocks (from Megatron-LM paper):

  ColumnParallelLinear:
    - Splits weight columns: each GPU holds W[:, local_cols]
    - Forward: Y_local = X @ W_local  (no comm needed)
    - Backward: all-reduce dX across GPUs
    - Used for: Q/K/V projections, first MLP linear

  RowParallelLinear:
    - Splits weight rows: each GPU holds W[local_rows, :]
    - Forward: Y_local = X_local @ W_local, then all-reduce to sum Y
    - Backward: no all-reduce needed for dX
    - Used for: attention output projection, second MLP linear

Net communication: 1 all-reduce per attention block + 1 per MLP block.

We also provide:
  - ParallelMultiHeadAttention: splits attention heads across GPUs
  - ParallelMLP: column-parallel first linear, row-parallel second
  - TensorParallelGPT2: full GPT-2 with tensor-parallel transformer blocks

Reference: Shoeybi et al., "Megatron-LM: Training Multi-Billion Parameter
Language Models Using Model Parallelism" (2019).
"""

import torch
import torch.nn as nn
import torch.distributed as dist
import math


# ============================================================================
# Custom autograd functions for communication in forward/backward
# ============================================================================

class _CopyToParallelRegion(torch.autograd.Function):
    """
    Forward: identity (input is replicated, just pass through).
    Backward: all-reduce gradients across the tensor parallel group.

    Used before ColumnParallelLinear — the input X is the same on all GPUs,
    so forward is a no-op. But in backward, dX needs to be summed because
    each GPU computed partial dX from its column slice.
    """
    @staticmethod
    def forward(ctx, x, group):
        ctx.group = group
        return x

    @staticmethod
    def backward(ctx, grad_output):
        dist.all_reduce(grad_output, op=dist.ReduceOp.SUM, group=ctx.group)
        return grad_output, None


class _ReduceFromParallelRegion(torch.autograd.Function):
    """
    Forward: all-reduce input across the tensor parallel group.
    Backward: identity (just pass gradient through).

    Used after RowParallelLinear — each GPU has a partial result that
    needs to be summed. In backward, gradients are already split correctly.
    """
    @staticmethod
    def forward(ctx, x, group):
        dist.all_reduce(x, op=dist.ReduceOp.SUM, group=group)
        return x

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None


# ============================================================================
# Parallel Linear Layers
# ============================================================================

class ColumnParallelLinear(nn.Module):
    """
    Linear layer with weight columns split across GPUs.

    Standard Linear:  Y = X @ W + b       where W is [in, out]
    Column Parallel:  Y_i = X @ W_i + b_i  where W_i is [in, out/N]

    Each GPU stores only out/N columns of W. The output Y_i is a partial
    result — the full output would be concat(Y_0, Y_1, ..., Y_{N-1}).

    For attention QKV: we don't need to concat — each GPU processes its
    own subset of attention heads independently.
    """
    def __init__(self, in_features, out_features, world_size, rank,
                 tp_group=None, bias=True):
        super().__init__()
        self.world_size = world_size
        self.rank = rank
        self.tp_group = tp_group

        # Each GPU gets out_features / world_size columns
        assert out_features % world_size == 0, \
            f"out_features ({out_features}) must be divisible by world_size ({world_size})"
        self.local_out = out_features // world_size

        self.weight = nn.Parameter(torch.empty(self.local_out, in_features))
        if bias:
            self.bias = nn.Parameter(torch.zeros(self.local_out))
        else:
            self.bias = None

        # Initialize weights (same seed offset per rank for reproducibility)
        nn.init.normal_(self.weight, mean=0.0, std=0.02)

    def forward(self, x):
        # Copy input to parallel region (no-op forward, all-reduce in backward)
        x = _CopyToParallelRegion.apply(x, self.tp_group)
        # Local matmul: each GPU computes its slice of the output
        out = nn.functional.linear(x, self.weight, self.bias)
        return out


class RowParallelLinear(nn.Module):
    """
    Linear layer with weight rows split across GPUs.

    Standard Linear:  Y = X @ W + b       where W is [in, out]
    Row Parallel:     Y = sum(X_i @ W_i) + b

    Each GPU holds W_i which is [in/N, out]. The input X_i is the local
    slice (output from a preceding ColumnParallelLinear).
    After the local matmul, we all-reduce to sum the partial results.
    """
    def __init__(self, in_features, out_features, world_size, rank,
                 tp_group=None, bias=True):
        super().__init__()
        self.world_size = world_size
        self.rank = rank
        self.tp_group = tp_group

        # Each GPU gets in_features / world_size rows
        assert in_features % world_size == 0, \
            f"in_features ({in_features}) must be divisible by world_size ({world_size})"
        self.local_in = in_features // world_size

        self.weight = nn.Parameter(torch.empty(out_features, self.local_in))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.bias = None

        nn.init.normal_(self.weight, mean=0.0, std=0.02)

    def forward(self, x):
        # Local matmul with our row slice
        out = nn.functional.linear(x, self.weight)
        # All-reduce to sum partial results across GPUs
        out = _ReduceFromParallelRegion.apply(out, self.tp_group)
        if self.bias is not None:
            out = out + self.bias
        return out


# ============================================================================
# Parallel Attention and MLP
# ============================================================================

class ParallelMultiHeadAttention(nn.Module):
    """
    Multi-head self-attention with heads split across GPUs.

    Each GPU handles n_heads/N attention heads. The QKV projection is
    column-parallel (splits output dim = 3*hidden across GPUs), and the
    output projection is row-parallel (each GPU's head outputs are partial).

    Communication: 1 all-reduce in the output projection's forward pass.
    """
    def __init__(self, cfg, world_size, rank, tp_group=None):
        super().__init__()
        self.hidden_dim = cfg.hidden_dim
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.head_dim
        self.world_size = world_size
        self.tp_group = tp_group

        assert cfg.n_heads % world_size == 0, \
            f"n_heads ({cfg.n_heads}) must be divisible by world_size ({world_size})"
        self.local_n_heads = cfg.n_heads // world_size

        # QKV projection: column-parallel (output = 3 * hidden_dim, split across GPUs)
        self.qkv_proj = ColumnParallelLinear(
            cfg.hidden_dim, 3 * cfg.hidden_dim, world_size, rank, tp_group
        )
        # Output projection: row-parallel (input = hidden_dim, split across GPUs)
        self.out_proj = RowParallelLinear(
            cfg.hidden_dim, cfg.hidden_dim, world_size, rank, tp_group
        )
        self.attn_dropout = nn.Dropout(cfg.dropout)
        self.resid_dropout = nn.Dropout(cfg.dropout)

    def forward(self, x):
        B, T, C = x.shape

        # QKV: each GPU gets 3 * hidden_dim / N outputs
        qkv = self.qkv_proj(x)  # (B, T, 3 * hidden/N)

        # Reshape into Q, K, V for our local heads
        local_dim = self.local_n_heads * self.head_dim
        qkv = qkv.reshape(B, T, 3, self.local_n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, local_heads, T, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Scaled dot-product attention with causal mask
        scale = 1.0 / math.sqrt(self.head_dim)
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale
        causal_mask = torch.tril(torch.ones(T, T, device=x.device, dtype=torch.bool))
        attn = attn.masked_fill(~causal_mask, float('-inf'))
        attn = torch.softmax(attn, dim=-1)
        attn = self.attn_dropout(attn)

        out = torch.matmul(attn, v)  # (B, local_heads, T, head_dim)
        out = out.transpose(1, 2).reshape(B, T, local_dim)

        # Output projection: row-parallel (all-reduce happens here)
        out = self.out_proj(out)
        return self.resid_dropout(out)


class ParallelMLP(nn.Module):
    """
    Feed-forward network with tensor parallelism.
    First linear is column-parallel, second is row-parallel.

    Communication: 1 all-reduce in the second linear's forward pass.
    Total per transformer block: 2 all-reduces (1 attention + 1 MLP).
    """
    def __init__(self, cfg, world_size, rank, tp_group=None):
        super().__init__()
        # fc1: column-parallel (split ff_dim across GPUs)
        self.fc1 = ColumnParallelLinear(
            cfg.hidden_dim, cfg.ff_dim, world_size, rank, tp_group
        )
        # fc2: row-parallel (each GPU has ff_dim/N input rows)
        self.fc2 = RowParallelLinear(
            cfg.ff_dim, cfg.hidden_dim, world_size, rank, tp_group
        )
        self.act = nn.GELU()
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return self.dropout(x)


# ============================================================================
# Tensor-Parallel Transformer Block and Full Model
# ============================================================================

class TensorParallelTransformerBlock(nn.Module):
    """
    Transformer block with tensor-parallel attention and MLP.
    Pre-norm (LayerNorm before attention/MLP), same as standard GPT-2.
    """
    def __init__(self, cfg, world_size, rank, tp_group=None):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.hidden_dim)
        self.attn = ParallelMultiHeadAttention(cfg, world_size, rank, tp_group)
        self.ln2 = nn.LayerNorm(cfg.hidden_dim)
        self.mlp = ParallelMLP(cfg, world_size, rank, tp_group)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


# ============================================================================
# Tensor-Parallel T5 Components
# ============================================================================

class ParallelSelfAttention(nn.Module):
    """
    Multi-head self-attention with heads split across GPUs.
    Supports both bidirectional (encoder) and causal (decoder) modes.
    Communication: 1 all-reduce in forward pass.
    """
    def __init__(self, cfg, world_size, rank, tp_group=None, causal=False):
        super().__init__()
        self.hidden_dim = cfg.hidden_dim
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.head_dim
        self.world_size = world_size
        self.causal = causal

        assert cfg.n_heads % world_size == 0
        self.local_n_heads = cfg.n_heads // world_size

        self.qkv_proj = ColumnParallelLinear(
            cfg.hidden_dim, 3 * cfg.hidden_dim, world_size, rank, tp_group
        )
        self.out_proj = RowParallelLinear(
            cfg.hidden_dim, cfg.hidden_dim, world_size, rank, tp_group
        )
        self.attn_dropout = nn.Dropout(cfg.dropout)
        self.resid_dropout = nn.Dropout(cfg.dropout)

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.qkv_proj(x)
        local_dim = self.local_n_heads * self.head_dim
        qkv = qkv.reshape(B, T, 3, self.local_n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        scale = 1.0 / math.sqrt(self.head_dim)
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale
        if self.causal:
            causal_mask = torch.tril(torch.ones(T, T, device=x.device, dtype=torch.bool))
            attn = attn.masked_fill(~causal_mask, float('-inf'))
        attn = torch.softmax(attn, dim=-1)
        attn = self.attn_dropout(attn)

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).reshape(B, T, local_dim)
        out = self.out_proj(out)
        return self.resid_dropout(out)


class ParallelCrossAttention(nn.Module):
    """
    Cross-attention with heads split across GPUs.
    Q from decoder, K/V from encoder — each split column-parallel.
    Communication: 1 all-reduce in forward pass (from out_proj).
    This is the EXTRA all-reduce that T5 decoder blocks have over GPT-2.
    """
    def __init__(self, cfg, world_size, rank, tp_group=None):
        super().__init__()
        self.hidden_dim = cfg.hidden_dim
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.head_dim
        self.world_size = world_size

        assert cfg.n_heads % world_size == 0
        self.local_n_heads = cfg.n_heads // world_size

        self.q_proj = ColumnParallelLinear(
            cfg.hidden_dim, cfg.hidden_dim, world_size, rank, tp_group
        )
        self.kv_proj = ColumnParallelLinear(
            cfg.hidden_dim, 2 * cfg.hidden_dim, world_size, rank, tp_group
        )
        self.out_proj = RowParallelLinear(
            cfg.hidden_dim, cfg.hidden_dim, world_size, rank, tp_group
        )
        self.attn_dropout = nn.Dropout(cfg.dropout)
        self.resid_dropout = nn.Dropout(cfg.dropout)

    def forward(self, x, enc_hidden):
        B, T_dec, C = x.shape
        T_enc = enc_hidden.shape[1]
        local_dim = self.local_n_heads * self.head_dim

        q = self.q_proj(x).reshape(B, T_dec, self.local_n_heads, self.head_dim).transpose(1, 2)

        kv = self.kv_proj(enc_hidden)
        kv = kv.reshape(B, T_enc, 2, self.local_n_heads, self.head_dim)
        kv = kv.permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        scale = 1.0 / math.sqrt(self.head_dim)
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale
        attn = torch.softmax(attn, dim=-1)
        attn = self.attn_dropout(attn)

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).reshape(B, T_dec, local_dim)
        out = self.out_proj(out)
        return self.resid_dropout(out)


class TensorParallelT5EncoderBlock(nn.Module):
    """
    T5 encoder block with tensor-parallel attention and MLP.
    Communication: 2 all-reduces per block (same as GPT-2).
    """
    def __init__(self, cfg, world_size, rank, tp_group=None):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.hidden_dim)
        self.attn = ParallelSelfAttention(cfg, world_size, rank, tp_group, causal=False)
        self.ln2 = nn.LayerNorm(cfg.hidden_dim)
        self.mlp = ParallelMLP(cfg, world_size, rank, tp_group)

    def forward(self, x, enc_hidden=None):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class TensorParallelT5DecoderBlock(nn.Module):
    """
    T5 decoder block with tensor-parallel self-attn, cross-attn, and MLP.
    Communication: 3 all-reduces per block (1 extra vs GPT-2 for cross-attn).
    """
    def __init__(self, cfg, world_size, rank, tp_group=None):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.hidden_dim)
        self.self_attn = ParallelSelfAttention(cfg, world_size, rank, tp_group, causal=True)
        self.ln2 = nn.LayerNorm(cfg.hidden_dim)
        self.cross_attn = ParallelCrossAttention(cfg, world_size, rank, tp_group)
        self.ln3 = nn.LayerNorm(cfg.hidden_dim)
        self.mlp = ParallelMLP(cfg, world_size, rank, tp_group)

    def forward(self, x, enc_hidden):
        x = x + self.self_attn(self.ln1(x))
        x = x + self.cross_attn(self.ln2(x), enc_hidden)
        x = x + self.mlp(self.ln3(x))
        return x


class TensorParallelT5(nn.Module):
    """
    Full T5 with 1D tensor parallelism.

    Embeddings are replicated. Encoder blocks get 2 all-reduces each,
    decoder blocks get 3 all-reduces each (extra for cross-attention).

    For T5-base (12+12 blocks):
      Encoder: 12 × 2 = 24 all-reduces
      Decoder: 12 × 3 = 36 all-reduces
      Total: 60 all-reduces per forward pass (vs GPT-2 Medium's 48)
    """
    def __init__(self, cfg, world_size, rank, tp_group=None):
        super().__init__()
        self.cfg = cfg

        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.hidden_dim)
        self.pos_emb = nn.Embedding(cfg.max_seq_len, cfg.hidden_dim)
        self.emb_dropout = nn.Dropout(cfg.dropout)

        self.encoder_blocks = nn.ModuleList([
            TensorParallelT5EncoderBlock(cfg, world_size, rank, tp_group)
            for _ in range(cfg.n_enc_layers)
        ])
        self.enc_ln_f = nn.LayerNorm(cfg.hidden_dim)

        self.decoder_blocks = nn.ModuleList([
            TensorParallelT5DecoderBlock(cfg, world_size, rank, tp_group)
            for _ in range(cfg.n_dec_layers)
        ])
        self.dec_ln_f = nn.LayerNorm(cfg.hidden_dim)

        self.lm_head = nn.Linear(cfg.hidden_dim, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, ColumnParallelLinear, RowParallelLinear)):
            if hasattr(module, 'weight'):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if hasattr(module, 'bias') and module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _embed(self, ids):
        B, T = ids.shape
        positions = torch.arange(0, T, device=ids.device).unsqueeze(0)
        return self.emb_dropout(self.tok_emb(ids) + self.pos_emb(positions))

    def forward(self, input_ids):
        B, T = input_ids.shape
        T_half = T // 2

        enc_ids = input_ids[:, :T_half]
        dec_ids = input_ids[:, T_half:]

        x = self._embed(enc_ids)
        for block in self.encoder_blocks:
            x = block(x)
        enc_hidden = self.enc_ln_f(x)

        x = self._embed(dec_ids)
        for block in self.decoder_blocks:
            x = block(x, enc_hidden)
        x = self.dec_ln_f(x)

        logits = self.lm_head(x)
        return logits


class TensorParallelGPT2(nn.Module):
    """
    Full GPT-2 with 1D tensor parallelism.

    Embeddings and final LayerNorm are replicated on all GPUs.
    Transformer blocks use tensor-parallel attention and MLP.
    The lm_head is replicated (tied to token embedding).

    Communication: 2 all-reduces per transformer block per forward pass.
    Total for GPT-2 Medium (24 blocks): 48 all-reduces per forward pass.
    """
    def __init__(self, cfg, world_size, rank, tp_group=None):
        super().__init__()
        self.cfg = cfg

        # Embeddings (replicated on all GPUs)
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.hidden_dim)
        self.pos_emb = nn.Embedding(cfg.max_seq_len, cfg.hidden_dim)
        self.emb_dropout = nn.Dropout(cfg.dropout)

        # Tensor-parallel transformer blocks
        self.blocks = nn.ModuleList([
            TensorParallelTransformerBlock(cfg, world_size, rank, tp_group)
            for _ in range(cfg.n_layers)
        ])

        # Final norm + lm_head (replicated)
        self.ln_f = nn.LayerNorm(cfg.hidden_dim)
        self.lm_head = nn.Linear(cfg.hidden_dim, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, ColumnParallelLinear, RowParallelLinear)):
            if hasattr(module, 'weight'):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if hasattr(module, 'bias') and module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids):
        B, T = input_ids.shape
        positions = torch.arange(0, T, device=input_ids.device).unsqueeze(0)
        x = self.tok_emb(input_ids) + self.pos_emb(positions)
        x = self.emb_dropout(x)

        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)
        logits = self.lm_head(x)
        return logits
