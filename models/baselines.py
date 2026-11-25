"""Baseline models used in Section 5.4.

These are faithful-in-spirit re-implementations from the cited papers' descriptions
(ResNet, MCNet, LSTM2, MCLDNN, FEA-T, VT-MCNet). They all accept the same (B,2,1024)
IQ input and emit (B,num_classes) logits. Exact replication of published numbers
requires the authors' official code; see README for the honest caveat.

For TF-only baselines the TFI tensor is accepted but ignored, so every model can
be trained with the exact same data pipeline and protocol.
"""
from __future__ import annotations

import torch
import torch.nn as nn


def conv1d_bn_relu(in_ch, out_ch, k=3, s=1, p=1):
    return nn.Sequential(
        nn.Conv1d(in_ch, out_ch, k, s, p, bias=False),
        nn.BatchNorm1d(out_ch), nn.ReLU(inplace=True),
    )


def conv2d_bn_relu(in_ch, out_ch, k=3, s=1, p=1):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, k, s, p, bias=False),
        nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
    )


# ---------------------------------------------------------------------------
# ResNet (O'Shea et al., RadioML2018.01a) -- 1D residual stack on raw IQ
# ---------------------------------------------------------------------------
class BasicBlock1D(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = conv1d_bn_relu(in_ch, out_ch, 3, stride, 1)
        self.conv2 = nn.Sequential(
            nn.Conv1d(out_ch, out_ch, 3, 1, 1, bias=False), nn.BatchNorm1d(out_ch))
        self.shortcut = (nn.Sequential(nn.Conv1d(in_ch, out_ch, 1, stride, bias=False),
                                       nn.BatchNorm1d(out_ch))
                         if (in_ch != out_ch or stride != 1) else nn.Identity())

    def forward(self, x):
        return nn.ReLU(inplace=True)(self.conv2(self.conv1(x)) + self.shortcut(x))


class ResNet1D(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.stem = conv1d_bn_relu(2, 64, 3, 1, 1)
        chans, strides = [64, 64, 128, 128, 256, 256], [1, 2, 2, 2, 2, 2]
        layers, cur = [], 64
        for c, s in zip(chans, strides):
            layers.append(BasicBlock1D(cur, c, s))
            cur = c
        self.layers = nn.Sequential(*layers)
        self.fc = nn.Linear(256, num_classes)

    def forward(self, iq, tfi=None):
        h = self.stem(iq)
        h = self.layers(h)
        return self.fc(h.mean(dim=2))


# ---------------------------------------------------------------------------
# MCNet (Huynh-The et al.) -- asymmetric convolutions on the 2 x L IQ image
# ---------------------------------------------------------------------------
class AsymResBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.a = conv2d_bn_relu(ch, ch, (3, 1), 1, (1, 0))
        self.b = conv2d_bn_relu(ch, ch, (1, 3), 1, (0, 1))

    def forward(self, x):
        return nn.ReLU(inplace=True)(self.a(x) + self.b(x) + x)


class MCNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.stem = conv2d_bn_relu(1, 32, (2, 3), 1, 0)
        self.stage1 = nn.Sequential(AsymResBlock(32), nn.MaxPool2d((1, 2)))
        self.stage2 = nn.Sequential(conv2d_bn_relu(32, 64, 1), AsymResBlock(64),
                                    nn.MaxPool2d((1, 2)))
        self.stage3 = nn.Sequential(conv2d_bn_relu(64, 128, 1), AsymResBlock(128),
                                    nn.MaxPool2d((1, 2)))
        self.fc = nn.Linear(128, num_classes)

    def forward(self, iq, tfi=None):
        x = iq.unsqueeze(1)                     # (B,1,2,1024)
        h = self.stage3(self.stage2(self.stage1(self.stem(x))))
        return self.fc(h.mean(dim=(2, 3)))


# ---------------------------------------------------------------------------
# LSTM2 (Rajendran et al.) -- stacked LSTM, option for amplitude/phase input
# ---------------------------------------------------------------------------
class LSTM2(nn.Module):
    def __init__(self, num_classes, hidden=128, layers=2, input_repr="iq"):
        super().__init__()
        self.input_repr = input_repr
        self.lstm = nn.LSTM(input_size=2, hidden_size=hidden, num_layers=layers,
                            batch_first=True, dropout=0.2 if layers > 1 else 0.0)
        self.fc = nn.Linear(hidden, num_classes)

    def forward(self, iq, tfi=None):
        if self.input_repr == "ap":             # amplitude/phase representation
            amp = torch.sqrt(iq[:, 0, :] ** 2 + iq[:, 1, :] ** 2 + 1e-8)
            phase = torch.atan2(iq[:, 1, :], iq[:, 0, :])
            x = torch.stack([amp, phase], dim=-1)         # (B,1024,2)
        else:
            x = iq.transpose(1, 2)                        # (B,1024,2)
        out, _ = self.lstm(x)
        return self.fc(out[:, -1])


# ---------------------------------------------------------------------------
# MCLDNN (Xu et al.) -- parallel I/Q/IQ conv streams merged with an LSTM
# ---------------------------------------------------------------------------
class MCLDNN(nn.Module):
    def __init__(self, num_classes, hidden=128):
        super().__init__()
        self.conv_i = conv1d_bn_relu(1, 32, 3)
        self.conv_q = conv1d_bn_relu(1, 32, 3)
        self.conv_iq = conv1d_bn_relu(2, 32, 3)
        self.lstm = nn.LSTM(96, hidden, num_layers=2, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden, num_classes)

    def forward(self, iq, tfi=None):
        i, q = iq[:, 0:1], iq[:, 1:2]
        h = torch.cat([self.conv_i(i), self.conv_q(q), self.conv_iq(iq)], dim=1)
        h = h.transpose(1, 2)                   # (B,L,96)
        out, _ = self.lstm(h)
        return self.fc(out[:, -1])


# ---------------------------------------------------------------------------
# FEA-T (Chen et al.) -- frame-wise embedding + Transformer encoder
# ---------------------------------------------------------------------------
class FEAT(nn.Module):
    def __init__(self, num_classes, frames=8, d_model=128, nhead=8,
                 dim_ff=256, num_layers=2):
        super().__init__()
        self.frames = frames
        frame_len = 1024 // frames
        self.embed = nn.Linear(2 * frame_len, d_model)
        self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos = nn.Parameter(torch.zeros(1, frames + 1, d_model))
        layer = nn.TransformerEncoderLayer(d_model, nhead, dim_ff, 0.1,
                                           batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, iq, tfi=None):
        B = iq.size(0)
        fl = iq.size(2) // self.frames
        x = iq[:, :, :self.frames * fl].view(B, 2, self.frames, fl)
        x = x.permute(0, 2, 1, 3).reshape(B, self.frames, 2 * fl)
        x = self.embed(x)
        x = torch.cat([self.cls.expand(B, -1, -1), x], dim=1) + self.pos
        x = self.norm(self.encoder(x))
        return self.fc(x[:, 0])


# ---------------------------------------------------------------------------
# VT-MCNet (Dao et al.) -- parallel multi-kernel 2D conv + Vision Transformer
# ---------------------------------------------------------------------------
class VTMCNET(nn.Module):
    def __init__(self, num_classes, d_model=128, nhead=8, dim_ff=256, num_layers=2):
        super().__init__()
        self.kernels = nn.ModuleList([
            conv2d_bn_relu(1, 64, (8, 2), 1, "same"),
            conv2d_bn_relu(1, 64, (16, 2), 1, "same"),
            conv2d_bn_relu(1, 64, (64, 2), 1, "same"),
        ])
        self.pool = nn.MaxPool2d((1, 8))
        self.proj = nn.Linear(192 * 2, d_model)          # 2 rows x 192 channels
        self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos = nn.Parameter(torch.zeros(1, 1 + 128, d_model))
        layer = nn.TransformerEncoderLayer(d_model, nhead, dim_ff, 0.1,
                                           batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.mlp_a = nn.Linear(d_model, d_model)
        self.mlp_b = nn.Linear(d_model, d_model)
        self.fc = nn.Linear(2 * d_model, num_classes)

    def forward(self, iq, tfi=None):
        x = iq.unsqueeze(1)                              # (B,1,2,1024)
        feats = [k(x) for k in self.kernels]
        h = self.pool(torch.cat(feats, dim=1))           # (B,192,2,128)
        tokens = h.permute(0, 3, 1, 2).reshape(h.size(0), h.size(3), -1)
        tokens = self.proj(tokens)
        B = tokens.size(0)
        tokens = torch.cat([self.cls.expand(B, -1, -1), tokens], dim=1) + self.pos
        x = self.norm(self.encoder(tokens))[:, 0]
        return self.fc(torch.cat([self.mlp_a(x), self.mlp_b(x)], dim=-1))


BASELINE_REGISTRY = {
    "resnet": ResNet1D,
    "mcnet": MCNet,
    "lstm2": LSTM2,
    "mcldnn": MCLDNN,
    "feat": FEAT,
    "vtmcnet": VTMCNET,
}


def build_baseline(name: str, num_classes: int):
    name = name.lower()
    if name not in BASELINE_REGISTRY:
        raise ValueError(f"Unknown baseline {name}. Available: {list(BASELINE_REGISTRY)}")
    return BASELINE_REGISTRY[name](num_classes)
