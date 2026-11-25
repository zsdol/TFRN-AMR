"""Parameter / FLOP counter (Table: Params & FLOPs for one 2x1024 frame).

FLOPs are estimated analytically by forwarding one dummy sample through the
model on CPU with forward hooks over Conv1d/Conv2d/Linear/MultiheadAttention.
The number is an approximation (MACs x 2, attention estimated as
4BLd^2 + 2BL^2d) -- adequate for relative complexity comparisons.
"""
from __future__ import annotations

import torch
import torch.nn as nn


def count_params(model: nn.Module):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return dict(params_M=total / 1e6, trainable_M=trainable / 1e6, params=total)


def count_flops(model: nn.Module, iq, tfi) -> float:
    stats = {"macs": 0}

    def hook_conv(m, inp, out):
        if isinstance(m, (nn.Conv1d, nn.Conv2d)):
            k = m.kernel_size
            k_prod = k[0] if isinstance(k, int) else int(np_prod(k))
            spatial = out.shape[2:]
            out_per_ch = int(np_prod(spatial))
            stats["macs"] += out.shape[0] * out_per_ch * (m.in_channels // m.groups) * k_prod

    def hook_linear(m, inp, out):
        if isinstance(m, nn.Linear):
            stats["macs"] += out.numel() // max(out.shape[-1], 1) * in_features_safe(m, inp)

    def hook_mha(m, inp, out):
        if isinstance(m, nn.MultiheadAttention):
            q = inp[0]
            if q.dim() == 3:
                B, L, d = q.shape
                stats["macs"] += B * (4 * L * d * d + 2 * L * L * d)

    def np_prod(t):
        v = 1
        for x in t:
            v *= int(x)
        return v

    def in_features_safe(m, inp):
        x = inp[0]
        return int(x.shape[-1]) if isinstance(x, torch.Tensor) else m.in_features

    handles = []
    for mod in model.modules():
        if isinstance(mod, (nn.Conv1d, nn.Conv2d)):
            handles.append(mod.register_forward_hook(hook_conv))
        elif isinstance(mod, nn.Linear):
            handles.append(mod.register_forward_hook(hook_linear))
        elif isinstance(mod, nn.MultiheadAttention):
            handles.append(mod.register_forward_hook(hook_mha))
    was_training = model.training
    model.eval()
    with torch.no_grad():
        model(iq, tfi)
    for h in handles:
        h.remove()
    model.train(was_training)
    return stats["macs"] * 2 / 1e6     # FLOPs in millions


def model_complexity(model: nn.Module, device="cpu", iq_length=1024,
                     img_h=40, img_w=50):
    info = count_params(model)
    iq = torch.randn(1, 2, iq_length, device=device)
    tfi = torch.randn(1, 1, img_h, img_w, device=device)
    info["flops_M"] = round(count_flops(model, iq, tfi), 3)
    return info
