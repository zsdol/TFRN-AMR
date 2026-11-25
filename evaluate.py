#!/usr/bin/env python
"""Evaluate a saved checkpoint once on the shared test split.

    python evaluate.py --ckpt runs/<exp>/seed0.pt --data-path ... --dataset 5gradio
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from config import CFG, DATASET_SPECS
from data import load_dataset, stratified_split
from models import build_model, build_baseline
from utils import per_snr_accuracy, summarize


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--dataset", default=None)
    p.add_argument("--data-path", default=None)
    p.add_argument("--csv-path", default=None)
    p.add_argument("--num-classes", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--split-seed", type=int, default=42)
    p.add_argument("--device", default="auto")
    p.add_argument("--plot", default="", help="optional output path for per-SNR curve PNG")
    return p.parse_args()


def main():
    args = parse_args()
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = CFG.from_dict(ckpt["cfg"])
    if args.dataset:
        cfg.dataset = args.dataset
    if args.data_path:
        cfg.data_path = args.data_path
    if args.csv_path:
        cfg.csv_path = args.csv_path
    if args.num_classes:
        cfg.num_classes = args.num_classes
    cfg.batch_size = args.batch_size
    cfg.num_workers = args.num_workers
    cfg.split_seed = args.split_seed
    if not cfg.snr_levels:
        cfg.snr_levels = DATASET_SPECS[cfg.dataset]["snr_levels"]

    device = torch.device("cuda" if torch.cuda.is_available() and
                          args.device in ("auto", "cuda") else "cpu")
    _, iq, labels, snrs = load_dataset(cfg)
    tr, va, te = stratified_split(labels, snrs, cfg.split_ratio, cfg.split_seed)
    ds, _, _, _ = load_dataset(cfg, split_indices=(tr, va, te),
                               arrays=(iq, labels, snrs))
    loader = DataLoader(ds[2], batch_size=cfg.batch_size, shuffle=False,
                        num_workers=cfg.num_workers)

    name = ckpt["model"]
    if name == "tfrn":
        model = build_model(cfg, cfg.num_classes)
    else:
        model = build_baseline(name, cfg.num_classes)
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()

    preds, ys, ss = [], [], []
    with torch.no_grad():
        for iq_b, tfi_b, y, snr in loader:
            logits = model(iq_b.to(device), tfi_b.to(device))
            preds.append(logits.argmax(1).cpu().numpy())
            ys.append(y.numpy())
            ss.append(snr.numpy())
    per = per_snr_accuracy(np.concatenate(ys), np.concatenate(preds), np.concatenate(ss))
    summary = summarize(per)
    print(json.dumps(dict(model=name, per_snr={str(k): round(v, 4) for k, v in per.items()},
                          summary=summary), indent=2))

    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        xs = sorted(per)
        plt.figure(figsize=(7, 4))
        plt.plot(xs, [per[x] * 100 for x in xs], marker="o")
        plt.xlabel("SNR (dB)")
        plt.ylabel("Accuracy (%)")
        plt.title(f"{name} per-SNR accuracy")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(args.plot, dpi=150)
        print(f"[plot] -> {args.plot}")


if __name__ == "__main__":
    main()
