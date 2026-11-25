"""Full TFRN model and model factory (Section 4.1, Fig. 1).

TFRN = IQ branch (MSCR stack) + TF branch (CNN+Transformer) + fusion + classifier.
Variants controlled by CFG.mode:
    mode='both' -> full multi-domain TFRN (with CFG.fusion)
    mode='iq'   -> TFRN-IQ alone (GAP + classifier)
    mode='tf'   -> TFRN-TF alone (CLS + classifier)
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from .mscr import IQBranch
from .tf_branch import TFBranch
from .fusion import build_fusion


class TFRN(nn.Module):
    def __init__(self, num_classes: int, d_model: int = 128, num_mscr: int = 6,
                 no_multiscale: bool = False, tf_heads: int = 8, tf_ff: int = 256,
                 tf_layers: int = 2, use_transformer: bool = True,
                 fusion: str = "cct", fusion_heads: int = 8,
                 mode: str = "both", dropout: float = 0.1,
                 img_h: int = 40, img_w: int = 50):
        super().__init__()
        self.mode = mode
        self.d_model = d_model

        self.iq_branch = IQBranch(in_ch=2, num_blocks=num_mscr, out_dim=d_model,
                                  no_multiscale=no_multiscale)
        self.tf_branch = TFBranch(img_h=img_h, img_w=img_w, d_model=d_model,
                                  nhead=tf_heads, dim_ff=tf_ff,
                                  num_layers=tf_layers,
                                  use_transformer=use_transformer,
                                  dropout=dropout)
        if mode == "both":
            self.fusion = build_fusion(fusion, d_model, fusion_heads, dropout)
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, iq: torch.Tensor, tfi: torch.Tensor,
                return_attn: bool = False):
        """
        iq:  (B, 2, 1024) raw IQ samples
        tfi: (B, 1, H, W) STFT magnitude image
        returns logits (B, M), optional attention weights
        """
        attn = None
        if self.mode == "iq":
            tokens = self.iq_branch(iq)                       # (B,L,d)
            fused = tokens.mean(dim=1, keepdim=True)           # GAP -> (B,1,d)
        elif self.mode == "tf":
            xtf, _ = self.tf_branch(tfi, need_weights=return_attn)
            fused = xtf.unsqueeze(1)                           # (B,1,d)
        else:
            x_iq = self.iq_branch(iq)                          # (B,16,d)
            x_tf, _ = self.tf_branch(tfi, need_weights=False)  # (B,d)
            x_tf = x_tf.unsqueeze(1)                           # (B,1,d)
            out = self.fusion(x_tf, x_iq, return_attn=return_attn)
            if isinstance(out, tuple):
                fused, attn = out
            else:
                fused, attn = out, None
        logits = self.classifier(fused.squeeze(1))
        return (logits, attn) if return_attn else logits


def build_model(cfg, num_classes: Optional[int] = None):
    """Factory used by train.py / evaluate.py."""
    n_cls = num_classes or cfg.num_classes
    return TFRN(
        num_classes=n_cls,
        d_model=cfg.d_model,
        num_mscr=cfg.num_mscr,
        no_multiscale=cfg.mscr_no_multiscale,
        tf_heads=cfg.tf_heads,
        tf_ff=cfg.tf_ff,
        tf_layers=cfg.tf_layers,
        use_transformer=cfg.use_transformer,
        fusion=cfg.fusion,
        fusion_heads=cfg.fusion_heads,
        mode=cfg.mode,
        dropout=cfg.dropout,
    )
