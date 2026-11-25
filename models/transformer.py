"""Pre-LN Transformer encoder used by the TFRN-TF branch (Section 4.3).

Each layer: LN -> MSA -> residual -> LN -> FFN -> residual.
FFN: ReLU(W1 x + b1) W2 + b2, with d_ff = 256 and d = 128.
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn


class PreLNTransformerLayer(nn.Module):
    def __init__(self, d_model: int, nhead: int, dim_ff: int, dropout: float = 0.1):
        super().__init__()
        if d_model % nhead != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by nhead ({nhead})")
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_ff), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(dim_ff, d_model), nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, need_weights: bool = False
                ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        y = self.norm1(x)
        attn_out, w = self.attn(y, y, y, need_weights=need_weights,
                                average_attn_weights=False)
        x = x + attn_out
        x = x + self.ffn(self.norm2(x))
        return x, w


class PreLNTransformerEncoder(nn.Module):
    def __init__(self, d_model: int, nhead: int, dim_ff: int,
                 num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            PreLNTransformerLayer(d_model, nhead, dim_ff, dropout)
            for _ in range(num_layers)
        ])

    def forward(self, x: torch.Tensor, need_weights: bool = False
                ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        weights = []
        for layer in self.layers:
            x, w = layer(x, need_weights=need_weights)
            if w is not None:
                weights.append(w)
        return x, weights
