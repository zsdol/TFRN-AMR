#!/usr/bin/env python
"""Shape/forward smoke test + one tiny training loop on synthetic data.

Run this first after installation:
    python smoke_test.py
"""
from __future__ import annotations

import numpy as np
import torch

from config import CFG
from data import iq_to_tfi, make_synthetic, stratified_split, ModulationDataset
from models import IQBranch, TFBranch, build_fusion, TFRN, build_baseline
from utils import per_snr_accuracy, summarize
from torch.utils.data import DataLoader
import torch.nn as nn


def check(name, cond):
    print(("  OK  " if cond else " FAIL ") + name)
    if not cond:
        raise AssertionError(name)


def main():
    print("== 1. STFT shape ==")
    iq0 = np.random.randn(2, 1024).astype(np.float32)
    tfi0 = iq_to_tfi(iq0)
    check("TFI is (40, 50)", tfi0.shape == (40, 50))

    print("== 2. IQ branch shapes (L0=1024 -> L6=16, d=128) ==")
    x = torch.randn(2, 2, 1024)
    for n in (2, 4, 6, 8):
        tok = IQBranch(num_blocks=n, out_dim=128)(x)
        exp_len = 1024 // (2 ** n)
        check(f"num_mscr={n}: tokens (2,{exp_len},128)", tuple(tok.shape) == (2, exp_len, 128))

    print("== 3. TF branch + fusion ==")
    img = torch.randn(2, 1, 40, 50)
    for heads in (4, 8, 16):
        tf = TFBranch(nhead=heads)(img)[0]
        check(f"TF heads={heads}: X_tf (2,128)", tuple(tf.shape) == (2, 128))
    xiq = torch.randn(2, 16, 128)
    xtf = torch.randn(2, 1, 128)
    for f in ("cad", "cst", "cct"):
        mod = build_fusion(f, 128, 8)
        out = mod(xtf, xiq)
        fout = out[0] if isinstance(out, tuple) else out
        check(f"fusion {f}: (2,1,128)", tuple(fout.shape) == (2, 1, 128))

    print("== 4. Full TFRN forward/backward ==")
    cfg = CFG(num_classes=6, dataset="synthetic")
    model = TFRN(num_classes=6)
    logits, attn = model(x, img, return_attn=True)
    check("logits (2,6)", tuple(logits.shape) == (2, 6))
    check("CCT attention (2,8,1,16)", attn is not None and tuple(attn.shape) == (2, 8, 1, 16))
    loss = nn.CrossEntropyLoss()(logits, torch.tensor([0, 1]))
    loss.backward()
    check("backward produces grads", model.classifier.weight.grad is not None)
    for mode in ("iq", "tf"):
        m = TFRN(num_classes=6, mode=mode)
        check(f"single-domain mode={mode}", tuple(m(x, img).shape) == (2, 6))

    print("== 5. Baselines forward ==")
    for name in ("resnet", "mcnet", "lstm2", "mcldnn", "feat", "vtmcnet"):
        m = build_baseline(name, 6)
        out = m(x, img)
        check(f"baseline {name}: (2,6)", tuple(out.shape) == (2, 6))

    print("== 6. Split + metrics ==")
    iqs, labs, snrs = make_synthetic(per_class_snr=8, seed=0)
    tr, va, te = stratified_split(labs, snrs, seed=42)
    check("split sizes positive", len(tr) > 0 and len(va) > 0 and len(te) > 0)
    ds = ModulationDataset(iqs, labs, snrs, indices=te)
    iq_b, tfi_b, y_b, s_b = ds[0]
    check("dataset item shapes", tuple(iq_b.shape) == (2, 1024) and
          tuple(tfi_b.shape) == (1, 40, 50))
    per = per_snr_accuracy([0, 1, 1], [0, 1, 0], [-5, -5, 10])
    s = summarize(per)
    check("per-SNR metrics", abs(s["avg"] - (1.0 + 0.0) / 2) < 1e-6)

    print("== 7. Mini training loop (1 epoch, synthetic) ==")
    from train import run, parse_args, build_cfg
    import argparse
    args = argparse.Namespace(
        dataset="synthetic", data_path="", csv_path="", num_classes=6, max_samples=None,
        split_seed=42, cache_tfi=False, model="tfrn", mode="both", fusion="cct",
        num_mscr=6, no_multiscale=False, tf_heads=8, no_transformer=False, d_model=128,
        epochs=1, batch_size=8, lr=1e-4, weight_decay=1e-4, early_stop_patience=2,
        scheduler_patience=1, seeds=[0], num_workers=0, device="cpu", amp=False,
        grad_clip=5.0, tfi_log=False, tfi_norm="max", out_dir="./runs",
        exp_name="smoke", no_save_ckpt=True, complexity=False)
    # cfg.log_every default 50 means no per-step logs for this short run
    cfg = build_cfg(args)
    cfg.log_every = 0
    res = run(args, cfg)
    check("mini training produced avg accuracy", 0.0 <= res["avg_mean"] <= 1.0)

    print("\nALL SMOKE TESTS PASSED")


if __name__ == "__main__":
    main()
