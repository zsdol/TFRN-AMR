"""Feature fusion mechanisms (Section 4.4, Fig. 2).

CAD: element-wise addition  F = X_tf + GAP(X_IQ)
CST: self-attention over [X_tf ; X_IQ], take the TF-token position
CCT: cross-attention, Query = X_tf (1 token), Key/Value = X_IQ (16 tokens)
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn


class AdditionFusion(nn.Module):
    """(a) CAD -- element-wise addition with global average pooling."""

    def forward(self, tf: torch.Tensor, iq: torch.Tensor,
                return_attn: bool = False):
        # tf: (B,1,d), iq: (B,L,d)
        f = tf + iq.mean(dim=1, keepdim=True)
        return (f, None) if return_attn else f


class SelfAttentionFusion(nn.Module):
    """(b) CST -- symmetric self-attention over the concatenated sequence."""

    def __init__(self, d_model: int, nhead: int = 8, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout,
                                          batch_first=True)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, tf: torch.Tensor, iq: torch.Tensor,
                return_attn: bool = False):
        seq = torch.cat([tf, iq], dim=1)                  # (B,1+L,d)
        out, w = self.attn(seq, seq, seq, need_weights=return_attn,
                           average_attn_weights=False)
        f = self.norm(seq + out)
        return (f[:, :1], w) if return_attn else f[:, :1]


class CrossAttentionFusion(nn.Module):
    """(c) CCT -- adopted asymmetric cross-attention fusion.

    The single global TF token queries the L=16 IQ temporal tokens, so the
    softmax is taken over the 16 IQ time positions.  With d=128 and h_f=8,
    the per-head dimension d_f = 16.
    """

    def __init__(self, d_model: int, nhead: int = 8, dropout: float = 0.1):
        super().__init__()
        if d_model % nhead != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by nhead ({nhead})")
        self.attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout,
                                          batch_first=True)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, tf: torch.Tensor, iq: torch.Tensor,
                return_attn: bool = False) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        out, w = self.attn(tf, iq, iq, need_weights=return_attn,
                           average_attn_weights=False)
        f = self.norm(tf + out)
        return (f, w) if return_attn else (f, None)


def build_fusion(name: str, d_model: int, nhead: int, dropout: float = 0.1):
    name = name.lower()
    if name == "cad":
        return AdditionFusion()
    if name == "cst":
        return SelfAttentionFusion(d_model, nhead, dropout)
    if name == "cct":
        return CrossAttentionFusion(d_model, nhead, dropout)
    raise ValueError(f"Unknown fusion mechanism: {name}. Choose from cct/cst/cad")
