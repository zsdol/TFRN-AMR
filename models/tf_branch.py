"""TFRN-TF branch: 2D CNN + Transformer over the STFT magnitude image.

Paper reference: Section 4.3, Table 2, Eqs. (7)-(12).

For a 40x50 TF image, three 2x2 max-pools give 5x6 -> P = 30 patch tokens;
prepending the learnable CLS token gives 31 tokens fed to two Transformer
layers. The CLS output is the global TF feature X_tf in R^(1 x d), d = 128.
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn

from .transformer import PreLNTransformerEncoder


def conv2d_bn_relu(in_ch: int, out_ch: int, k: int = 3, s: int = 1, p: int = 1) -> nn.Module:
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=p, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class TFBranch(nn.Module):
    def __init__(self, img_h: int = 40, img_w: int = 50, d_model: int = 128,
                 nhead: int = 8, dim_ff: int = 256, num_layers: int = 2,
                 use_transformer: bool = True, dropout: float = 0.1):
        super().__init__()
        self.cnn = nn.Sequential(
            conv2d_bn_relu(1, 32), nn.MaxPool2d(2, 2),
            conv2d_bn_relu(32, 64), nn.MaxPool2d(2, 2),
            conv2d_bn_relu(64, 128), nn.MaxPool2d(2, 2),
        )
        # infer P = H' * W' from the actual CNN output (40x50 -> 5x6 -> 30)
        with torch.no_grad():
            dummy = torch.zeros(1, 1, img_h, img_w)
            feat = self.cnn(dummy)
            self.num_patches = int(feat.shape[2] * feat.shape[3])

        self.patch_embed = nn.Linear(128, d_model)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        self.use_transformer = use_transformer
        if use_transformer:
            self.encoder = PreLNTransformerEncoder(d_model, nhead, dim_ff,
                                                    num_layers, dropout)
        self.d_model = d_model

    def forward(self, x: torch.Tensor, need_weights: bool = False
                ) -> Tuple[torch.Tensor, Optional[List[torch.Tensor]]]:
        """x: (B, 1, H, W) TF image -> X_tf: (B, d)."""
        feat = self.cnn(x)                                  # (B,128,H',W')

        if not self.use_transformer:
            # "No Trans" ablation: global-average-pool CNN output into X_tf
            return self.patch_embed(feat.mean(dim=(2, 3))), None

        tokens = feat.flatten(2).transpose(1, 2)            # (B,P,128)
        tokens = self.patch_embed(tokens)                   # (B,P,d)
        B = tokens.size(0)
        tokens = torch.cat([self.cls_token.expand(B, -1, -1), tokens], dim=1)
        tokens = tokens + self.pos_embed
        tokens, weights = self.encoder(tokens, need_weights=need_weights)
        return tokens[:, 0], weights                        # CLS token -> (B,d)
