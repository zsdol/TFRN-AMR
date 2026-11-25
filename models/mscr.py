"""TFRN-IQ branch: multi-scale convolutional residual (MSCR) blocks.

Paper reference: Section 4.2, Table 1, Eqs. (4)-(6).

Shape contract (L0 = 1024, six blocks, default settings):
    input (B, 2, 1024)
    preconv -> (B, 32, 1024)
    block 1/2 -> (B, 32, 512) / (B, 32, 256)
    block 3/4 -> (B, 64, 128) / (B, 64, 64)
    block 5/6 -> (B, 128, 32) / (B, 128, 16)
    output tokens -> (B, 16, 128)

Every one of the three parallel paths halves the time dimension, so the
shortcut (max-pool) and the concatenated features have identical lengths and
can be added element-wise.
"""
from __future__ import annotations

from typing import List

import torch
import torch.nn as nn


def conv1d_bn_relu(in_ch: int, out_ch: int, k: int, s: int = 1, p: int = 0) -> nn.Module:
    return nn.Sequential(
        nn.Conv1d(in_ch, out_ch, kernel_size=k, stride=s, padding=p, bias=False),
        nn.BatchNorm1d(out_ch),
        nn.ReLU(inplace=True),
    )


class MSCRBlock(nn.Module):
    """One multi-scale convolutional residual block (Table 1, rows 2-7).

    Main path:
        Conv1(k=3,s=1,p=1) -> { Conv2(k=3,s=2,p=1) || [Conv3(k=5,s=1,p=2)
        -> MaxPool(3,2,1)] }, concatenated along channels.
    Shortcut:
        MaxPool(3,2,1) (+ 1x1 Conv1d projection when channel count changes).
    Output:
        shortcut + concatenated multi-scale features.
    """

    def __init__(self, in_ch: int, mid_ch: int, branch_ch: int,
                 pool_k: int = 3, pool_s: int = 2, pool_p: int = 1):
        super().__init__()
        out_ch = 2 * branch_ch  # two branches of branch_ch each

        self.conv1 = conv1d_bn_relu(in_ch, mid_ch, 3, 1, 1)
        self.conv2 = conv1d_bn_relu(mid_ch, branch_ch, 3, 2, 1)   # stride-2 short branch
        self.conv3 = conv1d_bn_relu(mid_ch, branch_ch, 5, 1, 2)   # large-kernel branch
        self.pool = nn.MaxPool1d(pool_k, pool_s, pool_p)

        shortcut_layers: List[nn.Module] = [nn.MaxPool1d(pool_k, pool_s, pool_p)]
        if in_ch != out_ch:
            # Table 1 footnote: 1x1 projection when channels change between stages
            shortcut_layers += [
                nn.Conv1d(in_ch, out_ch, kernel_size=1, bias=False),
                nn.BatchNorm1d(out_ch),
            ]
        self.shortcut = nn.Sequential(*shortcut_layers)
        self.out_ch = out_ch

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sc = self.shortcut(x)
        c1 = self.conv1(x)
        c2 = self.conv2(c1)
        c3 = self.pool(self.conv3(c1))
        feat = torch.cat([c3, c2], dim=1)
        return feat + sc


class PlainResBlock(nn.Module):
    """Simplified block used by the 'No MultiScale' ablation (equal depth)."""

    def __init__(self, in_ch: int, mid_ch: int, out_ch: int,
                 pool_k: int = 3, pool_s: int = 2, pool_p: int = 1):
        super().__init__()
        self.conv1 = conv1d_bn_relu(in_ch, mid_ch, 3, 1, 1)
        self.conv2 = conv1d_bn_relu(mid_ch, out_ch, 3, 2, 1)
        shortcut: List[nn.Module] = [nn.MaxPool1d(pool_k, pool_s, pool_p)]
        if in_ch != out_ch:
            shortcut += [nn.Conv1d(in_ch, out_ch, 1, bias=False), nn.BatchNorm1d(out_ch)]
        self.shortcut = nn.Sequential(*shortcut)

    def forward(self, x):
        return self.conv2(self.conv1(x)) + self.shortcut(x)


class IQBranch(nn.Module):
    """Stack of MSCR blocks producing the temporal IQ token sequence."""

    # (mid_ch, branch_ch) for each two-block stage; out_ch = 2 * branch_ch
    STAGE_SPECS = [(32, 16), (64, 32), (128, 64), (128, 64)]

    def __init__(self, in_ch: int = 2, num_blocks: int = 6, out_dim: int = 128,
                 no_multiscale: bool = False):
        super().__init__()
        self.preconv = conv1d_bn_relu(in_ch, 32, 3, 1, 1)

        blocks: List[nn.Module] = []
        cur = 32
        for b in range(num_blocks):
            mid, branch = self.STAGE_SPECS[min(b // 2, len(self.STAGE_SPECS) - 1)]
            if no_multiscale:
                blocks.append(PlainResBlock(cur, mid, 2 * branch))
            else:
                blocks.append(MSCRBlock(cur, mid, branch))
            cur = 2 * branch
        self.blocks = nn.ModuleList(blocks)

        # For the default 6-block setting cur == 128 == d, so this is a no-op,
        # matching the paper ("no extra projection is needed"). It keeps the
        # ablation variants (2/4/8 blocks) compatible with the d=128 fusion.
        self.proj = nn.Conv1d(cur, out_dim, 1) if cur != out_dim else nn.Identity()
        self.out_dim = out_dim
        self.num_blocks = num_blocks
        self.final_channels = cur

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 2, L0) -> tokens: (B, L6, d)."""
        h = self.preconv(x)
        for block in self.blocks:
            h = block(h)
        h = self.proj(h)
        return h.transpose(1, 2).contiguous()   # (B, L', d)
