"""
T5 Model — encoder-decoder transformer for distributed training experiments.

T5 differs from GPT-2 in three ways that affect distributed training:
  1. Cross-attention in decoder blocks adds 50% more TP communication
     (3 all-reduces per decoder block vs GPT-2's 2 per block)
  2. Encoder uses bidirectional attention (no causal mask)
  3. Asymmetric encoder/decoder creates pipeline imbalance

For distributed training (TP=2, 12+12 blocks):
  - Encoder stage: 12 blocks × 2 all-reduces = 24 all-reduces
  - Decoder stage: 12 blocks × 3 all-reduces = 36 all-reduces
  - Total: 60 all-reduces vs GPT-2's 48

Configs provided:
  - T5Small: 6+6 layers, hidden=512,  8 heads (~60M params)
  - T5Base:  12+12 layers, hidden=768, 12 heads (~220M params)
"""

import torch
import torch.nn as nn
import math


class T5Config:
    """Configuration for T5 model variants."""
    def __init__(
        self,
        vocab_size=50257,
        max_seq_len=256,
        n_enc_layers=12,
        n_dec_layers=12,
        n_heads=12,
        hidden_dim=768,
        dropout=0.1,
    ):
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.n_enc_layers = n_enc_layers
        self.n_dec_layers = n_dec_layers
        self.n_layers = n_enc_layers + n_dec_layers
        self.n_heads = n_heads
        self.hidden_dim = hidden_dim
        self.ff_dim = 4 * hidden_dim
        self.dropout = dropout
        self.head_dim = hidden_dim // n_heads
        self.is_t5 = True

    @staticmethod
    def small():
        """T5-small: ~60M params."""
        return T5Config(n_enc_layers=6, n_dec_layers=6, n_heads=8, hidden_dim=512)

    @staticmethod
    def base():
        """T5-base: ~220M params."""
        return T5Config(n_enc_layers=12, n_dec_layers=12, n_heads=12, hidden_dim=768)


# ============================================================================
# Attention modules
# ============================================================================

class SelfAttention(nn.Module):
    """
    Multi-head self-attention, supports both bidirectional (encoder)
    and causal (decoder) modes via the `causal` flag.
    """
    def __init__(self, cfg, causal=False):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.head_dim
        self.hidden_dim = cfg.hidden_dim
        self.causal = causal

        self.qkv_proj = nn.Linear(cfg.hidden_dim, 3 * cfg.hidden_dim)
        self.out_proj = nn.Linear(cfg.hidden_dim, cfg.hidden_dim)
        self.attn_dropout = nn.Dropout(cfg.dropout)
        self.resid_dropout = nn.Dropout(cfg.dropout)

    def forward(self, x):
        B, T, C = x.shape

        qkv = self.qkv_proj(x)
        qkv = qkv.reshape(B, T, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        scale = 1.0 / math.sqrt(self.head_dim)
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * scale

        if self.causal:
            causal_mask = torch.tril(torch.ones(T, T, device=x.device, dtype=torch.bool))
            attn_weights = attn_weights.masked_fill(~causal_mask, float('-inf'))

        attn_weights = torch.softmax(attn_weights, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        out = torch.matmul(attn_weights, v)
        out = out.transpose(1, 2).reshape(B, T, C)
        return self.resid_dropout(self.out_proj(out))


class CrossAttention(nn.Module):
    """
    Cross-attention: queries from decoder hidden states,
    keys/values from encoder hidden states.
    This is the extra communication overhead that T5 has over GPT-2.
    """
    def __init__(self, cfg):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.head_dim
        self.hidden_dim = cfg.hidden_dim

        self.q_proj = nn.Linear(cfg.hidden_dim, cfg.hidden_dim)
        self.kv_proj = nn.Linear(cfg.hidden_dim, 2 * cfg.hidden_dim)
        self.out_proj = nn.Linear(cfg.hidden_dim, cfg.hidden_dim)
        self.attn_dropout = nn.Dropout(cfg.dropout)
        self.resid_dropout = nn.Dropout(cfg.dropout)

    def forward(self, x, enc_hidden):
        B, T_dec, C = x.shape
        T_enc = enc_hidden.shape[1]

        q = self.q_proj(x).reshape(B, T_dec, self.n_heads, self.head_dim).transpose(1, 2)

        kv = self.kv_proj(enc_hidden).reshape(B, T_enc, 2, self.n_heads, self.head_dim)
        kv = kv.permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        scale = 1.0 / math.sqrt(self.head_dim)
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * scale
        attn_weights = torch.softmax(attn_weights, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        out = torch.matmul(attn_weights, v)
        out = out.transpose(1, 2).reshape(B, T_dec, C)
        return self.resid_dropout(self.out_proj(out))


class MLP(nn.Module):
    """Feed-forward: Linear -> GELU -> Linear -> Dropout."""
    def __init__(self, cfg):
        super().__init__()
        self.fc1 = nn.Linear(cfg.hidden_dim, cfg.ff_dim)
        self.fc2 = nn.Linear(cfg.ff_dim, cfg.hidden_dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x):
        return self.dropout(self.fc2(self.act(self.fc1(x))))


# ============================================================================
# Encoder and Decoder blocks
# ============================================================================

class T5EncoderBlock(nn.Module):
    """
    Encoder block: pre-norm bidirectional self-attention + MLP.
    Communication in TP: 2 all-reduces (same as GPT-2 block).
    """
    def __init__(self, cfg):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.hidden_dim)
        self.attn = SelfAttention(cfg, causal=False)
        self.ln2 = nn.LayerNorm(cfg.hidden_dim)
        self.mlp = MLP(cfg)

    def forward(self, x, enc_hidden=None):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class T5DecoderBlock(nn.Module):
    """
    Decoder block: causal self-attention + cross-attention + MLP.
    Communication in TP: 3 all-reduces (1 extra vs GPT-2 for cross-attn).
    """
    def __init__(self, cfg):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.hidden_dim)
        self.self_attn = SelfAttention(cfg, causal=True)
        self.ln2 = nn.LayerNorm(cfg.hidden_dim)
        self.cross_attn = CrossAttention(cfg)
        self.ln3 = nn.LayerNorm(cfg.hidden_dim)
        self.mlp = MLP(cfg)

    def forward(self, x, enc_hidden):
        x = x + self.self_attn(self.ln1(x))
        x = x + self.cross_attn(self.ln2(x), enc_hidden)
        x = x + self.mlp(self.ln3(x))
        return x


# ============================================================================
# Full T5 Model
# ============================================================================

class T5Model(nn.Module):
    """
    Full T5 encoder-decoder language model.

    Architecture:
        Encoder: shared_embedding + positional → N encoder blocks → LayerNorm
        Decoder: shared_embedding + positional → N decoder blocks (with cross-attn) → LayerNorm → lm_head

    For benchmarking, input_ids is split in half:
      - First half → encoder input
      - Second half → decoder input
    This way total tokens processed = batch_size × seq_len (same as GPT-2).
    """
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        # Shared embedding (T5 shares embeddings between encoder and decoder)
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.hidden_dim)
        self.pos_emb = nn.Embedding(cfg.max_seq_len, cfg.hidden_dim)
        self.emb_dropout = nn.Dropout(cfg.dropout)

        # Encoder
        self.encoder_blocks = nn.ModuleList([
            T5EncoderBlock(cfg) for _ in range(cfg.n_enc_layers)
        ])
        self.enc_ln_f = nn.LayerNorm(cfg.hidden_dim)

        # Decoder
        self.decoder_blocks = nn.ModuleList([
            T5DecoderBlock(cfg) for _ in range(cfg.n_dec_layers)
        ])
        self.dec_ln_f = nn.LayerNorm(cfg.hidden_dim)

        # LM head (tied with shared embedding)
        self.lm_head = nn.Linear(cfg.hidden_dim, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _embed(self, ids):
        """Shared embedding + positional encoding."""
        B, T = ids.shape
        positions = torch.arange(0, T, device=ids.device).unsqueeze(0)
        return self.emb_dropout(self.tok_emb(ids) + self.pos_emb(positions))

    def forward(self, input_ids):
        """
        Args:
            input_ids: (B, T) — split internally into encoder/decoder halves
        Returns:
            logits: (B, T//2, vocab_size) — decoder predictions
        """
        B, T = input_ids.shape
        T_half = T // 2

        enc_ids = input_ids[:, :T_half]
        dec_ids = input_ids[:, T_half:]

        # Encode
        x = self._embed(enc_ids)
        for block in self.encoder_blocks:
            x = block(x)
        enc_hidden = self.enc_ln_f(x)

        # Decode with cross-attention to encoder output
        x = self._embed(dec_ids)
        for block in self.decoder_blocks:
            x = block(x, enc_hidden)
        x = self.dec_ln_f(x)

        logits = self.lm_head(x)
        return logits

    def get_num_params(self):
        """Return total number of parameters (in millions)."""
        return sum(p.numel() for p in self.parameters()) / 1e6


# ============================================================================
# Pipeline-Parallel T5 Stage
# ============================================================================

class T5PipelineStage(nn.Module):
    """
    Pipeline stage for T5 encoder-decoder model.

    With PP(2):
      Stage 0 (encoder): embeddings + all encoder blocks + enc LayerNorm
        forward(None, input_ids) → enc_hidden (B, T//2, hidden_dim)
      Stage 1 (decoder): embeddings + all decoder blocks + dec LayerNorm + lm_head
        forward(enc_hidden, input_ids) → logits (B, T//2, vocab_size)

    The encoder stage finishes faster (2 all-reduces/block, no cross-attn)
    and idles waiting for the heavier decoder stage (3 all-reduces/block).
    This demonstrates pipeline imbalance from architectural asymmetry.
    """
    def __init__(self, cfg, stage_type, is_first, is_last):
        super().__init__()
        self.cfg = cfg
        self.stage_type = stage_type  # 'encoder' or 'decoder'
        self.is_first = is_first
        self.is_last = is_last

        # Both stages need embeddings (encoder for its input, decoder for its input)
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.hidden_dim)
        self.pos_emb = nn.Embedding(cfg.max_seq_len, cfg.hidden_dim)
        self.emb_dropout = nn.Dropout(cfg.dropout)

        if stage_type == 'encoder':
            self.blocks = nn.ModuleList([
                T5EncoderBlock(cfg) for _ in range(cfg.n_enc_layers)
            ])
            self.ln_f = nn.LayerNorm(cfg.hidden_dim)
        else:  # decoder
            self.blocks = nn.ModuleList([
                T5DecoderBlock(cfg) for _ in range(cfg.n_dec_layers)
            ])
            self.ln_f = nn.LayerNorm(cfg.hidden_dim)
            self.lm_head = nn.Linear(cfg.hidden_dim, cfg.vocab_size, bias=False)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _embed(self, ids):
        B, T = ids.shape
        positions = torch.arange(0, T, device=ids.device).unsqueeze(0)
        return self.emb_dropout(self.tok_emb(ids) + self.pos_emb(positions))

    def forward(self, x, input_ids=None):
        """
        Args:
            x: hidden states from previous stage (None for encoder stage)
            input_ids: (B, T) full token sequence — stage picks its half
        Returns:
            enc_hidden (encoder stage) or logits (decoder stage)
        """
        T_half = self.cfg.max_seq_len // 2

        if self.stage_type == 'encoder':
            # Encoder: embed first half, run encoder blocks
            enc_ids = input_ids[:, :T_half]
            x = self._embed(enc_ids)
            for block in self.blocks:
                x = block(x)
            x = self.ln_f(x)
            return x  # enc_hidden: (B, T//2, hidden_dim)

        else:  # decoder
            # x = enc_hidden from encoder stage
            enc_hidden = x
            dec_ids = input_ids[:, T_half:]
            x = self._embed(dec_ids)
            for block in self.blocks:
                x = block(x, enc_hidden)
            x = self.ln_f(x)
            x = self.lm_head(x)
            return x  # logits: (B, T//2, vocab_size)


def create_t5_pipeline_stage(cfg, num_stages, stage_id, device):
    """
    Create a T5 pipeline stage. Only PP(2) is supported:
      Stage 0 = full encoder, Stage 1 = full decoder.
    """
    assert num_stages == 2, \
        f"T5 PP only supports 2 stages (got {num_stages}). " \
        f"Encoder=stage0, Decoder=stage1."

    is_first = (stage_id == 0)
    is_last = (stage_id == num_stages - 1)
    stage_type = 'encoder' if stage_id == 0 else 'decoder'

    stage = T5PipelineStage(cfg, stage_type, is_first, is_last).to(device)
    return stage


# ============================================================================
# Tensor-Parallel T5 Pipeline Stage (for PP + TP combined / true 3D)
# ============================================================================

class TensorParallelT5PipelineStage(nn.Module):
    """
    T5 pipeline stage with tensor-parallel blocks.

    Same as T5PipelineStage but uses TensorParallelT5EncoderBlock and
    TensorParallelT5DecoderBlock. This enables true 3D parallelism where
    each pipeline stage's layers are also sharded across TP ranks.
    """
    def __init__(self, cfg, stage_type, is_first, is_last,
                 tp_size, tp_rank, tp_group):
        super().__init__()
        from minicolossal.tensor_parallel import (
            TensorParallelT5EncoderBlock,
            TensorParallelT5DecoderBlock,
            ColumnParallelLinear,
            RowParallelLinear,
        )

        self.cfg = cfg
        self.stage_type = stage_type
        self.is_first = is_first
        self.is_last = is_last

        # Both stages need embeddings (replicated across TP ranks)
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.hidden_dim)
        self.pos_emb = nn.Embedding(cfg.max_seq_len, cfg.hidden_dim)
        self.emb_dropout = nn.Dropout(cfg.dropout)

        if stage_type == 'encoder':
            self.blocks = nn.ModuleList([
                TensorParallelT5EncoderBlock(cfg, tp_size, tp_rank, tp_group)
                for _ in range(cfg.n_enc_layers)
            ])
            self.ln_f = nn.LayerNorm(cfg.hidden_dim)
        else:  # decoder
            self.blocks = nn.ModuleList([
                TensorParallelT5DecoderBlock(cfg, tp_size, tp_rank, tp_group)
                for _ in range(cfg.n_dec_layers)
            ])
            self.ln_f = nn.LayerNorm(cfg.hidden_dim)
            self.lm_head = nn.Linear(cfg.hidden_dim, cfg.vocab_size, bias=False)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        from minicolossal.tensor_parallel import ColumnParallelLinear, RowParallelLinear
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

    def forward(self, x, input_ids=None):
        """Same interface as T5PipelineStage.forward()."""
        T_half = self.cfg.max_seq_len // 2

        if self.stage_type == 'encoder':
            enc_ids = input_ids[:, :T_half]
            x = self._embed(enc_ids)
            for block in self.blocks:
                x = block(x)
            x = self.ln_f(x)
            return x

        else:  # decoder
            enc_hidden = x
            dec_ids = input_ids[:, T_half:]
            x = self._embed(dec_ids)
            for block in self.blocks:
                x = block(x, enc_hidden)
            x = self.ln_f(x)
            x = self.lm_head(x)
            return x


def create_tp_t5_pipeline_stage(cfg, num_stages, stage_id, tp_size, tp_rank,
                                tp_group, device):
    """
    Create a tensor-parallel T5 pipeline stage for true 3D parallelism.
    """
    assert num_stages == 2, \
        f"T5 PP only supports 2 stages (got {num_stages}). " \
        f"Encoder=stage0, Decoder=stage1."

    is_first = (stage_id == 0)
    is_last = (stage_id == num_stages - 1)
    stage_type = 'encoder' if stage_id == 0 else 'decoder'

    stage = TensorParallelT5PipelineStage(
        cfg, stage_type, is_first, is_last,
        tp_size, tp_rank, tp_group
    ).to(device)
    return stage
