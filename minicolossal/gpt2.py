"""
GPT-2 Model — built from scratch for distributed training experiments.

We implement GPT-2 as a clean sequence of TransformerBlocks so that:
  - Pipeline parallelism can split at block boundaries
  - Tensor parallelism can replace Linear layers inside blocks
  - Data parallelism and ZeRO work on the full model

Four configs provided:
  - GPT2Small:  12 layers, hidden=768,  12 heads (~117M params)
  - GPT2Medium: 24 layers, hidden=1024, 16 heads (~354M params)
  - GPT2Large:  36 layers, hidden=1280, 20 heads (~774M params)
  - GPT2XL:     48 layers, hidden=1600, 25 heads (~1.5B params)
"""

import torch
import torch.nn as nn
import math


class GPT2Config:
    """Configuration for GPT-2 model variants."""
    def __init__(
        self,
        vocab_size=50257,
        max_seq_len=256,
        n_layers=24,
        n_heads=16,
        hidden_dim=1024,
        dropout=0.1,
    ):
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.hidden_dim = hidden_dim
        self.ff_dim = 4 * hidden_dim  # standard GPT-2 ratio
        self.dropout = dropout
        self.head_dim = hidden_dim // n_heads

    @staticmethod
    def small():
        """GPT-2 Small: ~117M params."""
        return GPT2Config(n_layers=12, n_heads=12, hidden_dim=768)

    @staticmethod
    def medium():
        """GPT-2 Medium: ~354M params."""
        return GPT2Config(n_layers=24, n_heads=16, hidden_dim=1024)

    @staticmethod
    def large():
        """GPT-2 Large: ~774M params."""
        return GPT2Config(n_layers=36, n_heads=20, hidden_dim=1280)

    @staticmethod
    def xl():
        """GPT-2 XL: ~1.5B params."""
        return GPT2Config(n_layers=48, n_heads=32, hidden_dim=1600)


class MultiHeadSelfAttention(nn.Module):
    """
    Multi-head self-attention with causal mask.
    Q, K, V are computed from a single fused linear projection for efficiency.
    """
    def __init__(self, cfg):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.head_dim
        self.hidden_dim = cfg.hidden_dim

        # Fused QKV projection: one big matrix multiply instead of three
        self.qkv_proj = nn.Linear(cfg.hidden_dim, 3 * cfg.hidden_dim)
        self.out_proj = nn.Linear(cfg.hidden_dim, cfg.hidden_dim)
        self.attn_dropout = nn.Dropout(cfg.dropout)
        self.resid_dropout = nn.Dropout(cfg.dropout)

    def forward(self, x):
        B, T, C = x.shape

        # Compute Q, K, V in one shot
        qkv = self.qkv_proj(x)  # (B, T, 3*C)
        qkv = qkv.reshape(B, T, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, n_heads, T, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Scaled dot-product attention with causal mask
        scale = 1.0 / math.sqrt(self.head_dim)
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * scale

        # Causal mask: prevent attending to future tokens
        causal_mask = torch.tril(torch.ones(T, T, device=x.device, dtype=torch.bool))
        attn_weights = attn_weights.masked_fill(~causal_mask, float('-inf'))

        attn_weights = torch.softmax(attn_weights, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        # Weighted sum of values
        out = torch.matmul(attn_weights, v)  # (B, n_heads, T, head_dim)
        out = out.transpose(1, 2).reshape(B, T, C)  # (B, T, C)

        return self.resid_dropout(self.out_proj(out))


class MLP(nn.Module):
    """
    Feed-forward network: Linear -> GELU -> Linear -> Dropout.
    The intermediate dimension is 4x the hidden dimension (standard GPT-2).
    """
    def __init__(self, cfg):
        super().__init__()
        self.fc1 = nn.Linear(cfg.hidden_dim, cfg.ff_dim)
        self.fc2 = nn.Linear(cfg.ff_dim, cfg.hidden_dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class TransformerBlock(nn.Module):
    """
    One transformer block: LayerNorm -> Attention -> residual -> LayerNorm -> MLP -> residual.
    Uses pre-norm (LayerNorm before attention/MLP) which is the GPT-2 convention.
    """
    def __init__(self, cfg):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.hidden_dim)
        self.attn = MultiHeadSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.hidden_dim)
        self.mlp = MLP(cfg)

    def forward(self, x):
        # Pre-norm + residual connections
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class GPT2Model(nn.Module):
    """
    Full GPT-2 language model.

    Architecture:
        token_embedding + position_embedding
        -> dropout
        -> N x TransformerBlock
        -> LayerNorm
        -> lm_head (tied with token_embedding weights)

    The model is structured so that self.blocks is a nn.ModuleList,
    making it easy to split at block boundaries for pipeline parallelism.
    """
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        # Embeddings
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.hidden_dim)
        self.pos_emb = nn.Embedding(cfg.max_seq_len, cfg.hidden_dim)
        self.emb_dropout = nn.Dropout(cfg.dropout)

        # Transformer blocks (stored as ModuleList for easy slicing)
        self.blocks = nn.ModuleList([
            TransformerBlock(cfg) for _ in range(cfg.n_layers)
        ])

        # Final layer norm + language model head
        self.ln_f = nn.LayerNorm(cfg.hidden_dim)
        self.lm_head = nn.Linear(cfg.hidden_dim, cfg.vocab_size, bias=False)

        # Weight tying: share embedding weights with output head
        self.lm_head.weight = self.tok_emb.weight

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize weights using normal distribution (GPT-2 style)."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids):
        """
        Args:
            input_ids: (batch_size, seq_len) tensor of token ids
        Returns:
            logits: (batch_size, seq_len, vocab_size) raw predictions
        """
        B, T = input_ids.shape

        # Token + positional embeddings
        positions = torch.arange(0, T, device=input_ids.device).unsqueeze(0)
        x = self.tok_emb(input_ids) + self.pos_emb(positions)
        x = self.emb_dropout(x)

        # Pass through all transformer blocks
        for block in self.blocks:
            x = block(x)

        # Final norm + project to vocab
        x = self.ln_f(x)
        logits = self.lm_head(x)
        return logits

    def get_num_params(self):
        """Return total number of parameters (in millions)."""
        return sum(p.numel() for p in self.parameters()) / 1e6
