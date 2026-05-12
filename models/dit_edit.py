"""DIT backbone extended with source-text cross-attention for text editing.

Based on models/dit.py from https://github.com/david3684/flm, adapted to
condition on a source sequence via cross-attention in each block.
"""
import math
import typing

import einops
import flash_attn
import flash_attn.layers.rotary
import huggingface_hub
import omegaconf
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from torch.nn.attention import SDPBackend, sdpa_kernel

# Reuse all low-level utilities from the original dit.py
from models.dit import (
    USE_COMPILE,
    torch_compile_deco,
    jit_deco,
    bias_dropout_add_scale,
    get_bias_dropout_add_scale,
    modulate,
    bias_dropout_add_scale_fused_train,
    bias_dropout_add_scale_fused_inference,
    modulate_fused,
    Rotary,
    rotate_half,
    split_and_apply_rotary_pos_emb,
    apply_rotary_pos_emb,
    regular_attention_multi_headed,
    LearnableLossWeighting,
    LayerNorm,
    residual_linear,
    TimestepEmbedder,
    SquaredReLU,
    LabelEmbedder,
    DDiTBlock,
    EmbeddingLayer,
    DDiTFinalLayer,
)


class DDiTEditBlock(nn.Module):
    """DDiTBlock extended with a gated cross-attention layer over source tokens."""

    def __init__(self, dim, n_heads, cond_dim, mlp_ratio=4, dropout=0.1):
        super().__init__()
        self.n_heads = n_heads
        self.softcap = 50
        self.dropout = dropout

        # --- self-attention (identical to DDiTBlock with adaLN=True) ---
        self.norm1 = LayerNorm(dim)
        self.attn_qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.attn_out = nn.Linear(dim, dim, bias=False)
        self.dropout1 = nn.Dropout(dropout)

        self.norm2 = LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_ratio * dim, bias=True),
            nn.GELU(approximate='tanh'),
            nn.Linear(mlp_ratio * dim, dim, bias=True))
        self.dropout2 = nn.Dropout(dropout)

        self.adaLN_modulation = nn.Linear(cond_dim, 6 * dim)
        self.adaLN_modulation.weight.data.zero_()
        self.adaLN_modulation.bias.data.zero_()

        # --- cross-attention over source tokens (NEW) ---
        self.cross_norm = LayerNorm(dim)
        self.cross_attn_q = nn.Linear(dim, dim, bias=False)
        self.cross_attn_kv = nn.Linear(dim, 2 * dim, bias=False)
        self.cross_attn_out = nn.Linear(dim, dim, bias=False)
        # Zero-init gate: cross-attention starts as identity (stable init)
        self.cross_gate = nn.Parameter(torch.zeros(dim))

    def _get_bias_dropout_scale(self):
        if self.training:
            return bias_dropout_add_scale_fused_train
        else:
            return bias_dropout_add_scale_fused_inference

    def custom_sdpa(self, q, k, v, softcap=-1.0):
        B, H, S, D = q.shape
        q = q / (D ** 0.5)
        attn_weights = torch.einsum('bhid,bhjd->bhij', q, k)
        if softcap > 0.0:
            attn_weights = softcap * torch.tanh(attn_weights / softcap)
        attn_probs = torch.softmax(attn_weights, dim=-1)
        output = torch.einsum('bhij,bhjd->bhid', attn_probs, v)
        return output

    def forward(self, x, src_enc, rotary_cos_sin=None, c=None,
                seqlens=None, use_jvp_attn=False):
        bias_dropout_scale_fn = self._get_bias_dropout_scale()

        # --- 1. adaLN-modulated self-attention (copied from DDiTBlock) ---
        x_skip = x
        x = self.norm1(x)

        (shift_msa, scale_msa, gate_msa, shift_mlp,
         scale_mlp, gate_mlp) = self.adaLN_modulation(c)[:, None].chunk(6, dim=2)
        x = modulate_fused(x, shift_msa, scale_msa)

        qkv = self.attn_qkv(x)
        qkv = einops.rearrange(
            qkv, 'b s (three h d) -> b s three h d', three=3, h=self.n_heads)
        with torch.cuda.amp.autocast(enabled=False):
            cos, sin = rotary_cos_sin
            qkv = apply_rotary_pos_emb(
                qkv, cos.to(qkv.dtype), sin.to(qkv.dtype),
                use_flash=not use_jvp_attn)

        if use_jvp_attn:
            q, k, v = qkv.unbind(dim=2)
            q = q.transpose(1, 2)
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)
            x = self.custom_sdpa(q, k, v, softcap=self.softcap)
            x = x.transpose(1, 2)
        else:
            x = flash_attn.flash_attn_qkvpacked_func(
                qkv, 0.0, causal=False, softcap=self.softcap)

        x = einops.rearrange(x, 'b s h d -> b s (h d)')
        x = bias_dropout_scale_fn(
            self.attn_out(x), None, gate_msa, x_skip, self.dropout)

        # --- 2. Cross-attention over source tokens (NEW) ---
        x_normed = self.cross_norm(x)
        q_cross = self.cross_attn_q(x_normed)               # (B, L_tgt, dim)
        kv_cross = self.cross_attn_kv(src_enc)               # (B, L_src, 2*dim)
        k_cross, v_cross = kv_cross.chunk(2, dim=-1)

        # Standard SDPA (not flash) so JVP works through this path
        scale = q_cross.shape[-1] ** -0.5
        attn_w = (q_cross @ k_cross.transpose(-2, -1)) * scale  # (B, L_tgt, L_src)
        attn_w = attn_w.softmax(dim=-1)
        cross_out = attn_w @ v_cross                          # (B, L_tgt, dim)
        cross_out = self.cross_attn_out(cross_out)
        # Gated residual: gate starts near zero → smooth incorporation during training
        x = x + torch.tanh(self.cross_gate) * cross_out

        # --- 3. adaLN-modulated MLP (copied from DDiTBlock) ---
        x = bias_dropout_scale_fn(
            self.mlp(modulate_fused(self.norm2(x), shift_mlp, scale_mlp)),
            None, gate_mlp, x, self.dropout)

        return x


class SourceEncoder(nn.Module):
    """Bidirectional transformer encoder for the source tokens.

    Produces a sequence of context vectors (B, L, dim) that DITEdit
    attends to via cross-attention. Uses the same vocab size as the
    main model but maintains independent weights.
    """

    def __init__(self, config, vocab_size: int):
        super().__init__()
        dim = config.model.hidden_size
        n_heads = config.model.n_heads
        dropout = config.model.dropout
        n_enc = getattr(config.model, 'n_src_encoder_blocks', 4)

        self.embed = EmbeddingLayer(dim, vocab_size)
        self.rotary_emb = Rotary(dim // n_heads)

        # Plain DDiTBlock without adaLN (no time conditioning needed)
        self.blocks = nn.ModuleList([
            DDiTBlock(dim=dim, n_heads=n_heads, adaLN=False,
                      cond_dim=dim, dropout=dropout)
            for _ in range(n_enc)
        ])
        self.norm = LayerNorm(dim)

    def forward(self, src_ids):
        """
        Args:
            src_ids: (B, L) integer token IDs
        Returns:
            (B, L, dim) contextual source encodings
        """
        x = self.embed(src_ids)                   # (B, L, dim)
        rotary_cos_sin = self.rotary_emb(x)
        for block in self.blocks:
            x = block(x, rotary_cos_sin, c=None, use_jvp_attn=False)
        return self.norm(x)


class DITEdit(nn.Module, huggingface_hub.PyTorchModelHubMixin):
    """DIT backbone conditioned on source tokens via cross-attention.

    Replaces DIT for the text-editing setting. The noisy input x_noisy
    is the source→target interpolant; x_src_ids is the source text.
    """

    def __init__(self, config, vocab_size: int):
        super().__init__()
        if isinstance(config, dict):
            config = omegaconf.OmegaConf.create(config)
        self.config = config
        self.vocab_size = vocab_size

        dim = config.model.hidden_size
        cond_dim = config.model.cond_dim
        n_heads = config.model.n_heads
        dropout = config.model.dropout

        # Token + time embeddings (same as upstream DIT)
        self.vocab_embed = EmbeddingLayer(dim, vocab_size)
        self.sigma_map = TimestepEmbedder(cond_dim)
        if getattr(config.algo, 'double_temb', False):
            self.sigma_map_prime = TimestepEmbedder(cond_dim)
        else:
            self.sigma_map_prime = None

        self.rotary_emb = Rotary(dim // n_heads)

        # Transformer blocks with source cross-attention
        self.blocks = nn.ModuleList([
            DDiTEditBlock(dim=dim, n_heads=n_heads, cond_dim=cond_dim,
                          dropout=dropout)
            for _ in range(config.model.n_blocks)
        ])

        self.output_layer = DDiTFinalLayer(
            hidden_size=dim,
            out_channels=vocab_size,
            cond_dim=cond_dim,
            adaLN=True)

        # Source text encoder
        self.src_encoder = SourceEncoder(config, vocab_size)

        # Optional dialogue/document context encoder (independent weights)
        if getattr(getattr(config, 'data', None), 'use_context', False):
            self.ctx_encoder = SourceEncoder(config, vocab_size)
        else:
            self.ctx_encoder = None

        self.scale_by_sigma = config.model.scale_by_sigma
        self.is_di4c = False
        self.is_di4c_deterministic = False

    def forward(self, x_noisy, x_src_ids, sigma, sigma_prime=None,
                use_jvp_attn=False, context_ids=None):
        """
        Args:
            x_noisy:     (B, L, V) interpolated continuous source+target embedding
            x_src_ids:   (B, L) integer source token IDs
            sigma:       (B,) flow time τ_s
            sigma_prime: (B,) flow time τ_t (two-time denoiser only)
            context_ids: (B, L_ctx) integer document context token IDs, or None
        Returns:
            logits: (B, L, V) pre-softmax output
        """
        # Encode source once; reused by all cross-attention layers
        src_enc = self.src_encoder(x_src_ids)          # (B, L_src, dim)
        if context_ids is not None and self.ctx_encoder is not None:
            ctx_enc = self.ctx_encoder(context_ids)    # (B, L_ctx, dim)
            src_enc = torch.cat([src_enc, ctx_enc], dim=1)  # (B, L_src+L_ctx, dim)

        # Time conditioning
        t_emb = self.sigma_map(sigma)
        if sigma_prime is not None and self.sigma_map_prime is not None:
            t_emb = t_emb + self.sigma_map_prime(sigma_prime)
        t_cond = F.silu(t_emb)                         # (B, cond_dim)

        # Token embedding of noisy input
        x = self.vocab_embed(x_noisy)                  # (B, L, dim)

        rotary_cos_sin = self.rotary_emb(x)

        with torch.amp.autocast(device_type=x.device.type, dtype=torch.bfloat16):
            for block in self.blocks:
                x = block(x, src_enc, rotary_cos_sin, c=t_cond,
                          use_jvp_attn=use_jvp_attn)
            x = self.output_layer(x, c=t_cond)

        return x
