#!/usr/bin/env python
"""Main training / evaluation driver for TFRN.

Usage examples (see README for the full matrix):
    python train.py --dataset synthetic --model tfrn --epochs 3 --seeds 0 --num-workers 0
    python train.py --dataset radioml2018 --data-path GOLD_XYZ_OSC.0001_16.hdf5 \
        --model tfrn --fusion cct --seeds 0 1 2
    python train.py --dataset 5gradio --data-path 5GRadio.hdf5 \
        --model lstm2                       # baseline, identical protocol
    python train.py --dataset radioml2018 --data-path ... --mode iq   # single-domain ablation
    python train.py --dataset radioml2018 --data-path ... --num-mscr 2 --no-multiscale
"""
from __future__ import annotations

import argparse
import json
import os
import time
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from config import CFG, DATASET_SPECS
from data import load_dataset, stratified_split
from models import build_model, build_baseline
from utils import set_seed, per_snr_accuracy, summarize, mean_std, paired_tests
from utils.params import model_complexity


def parse_args():
    p = argparse.ArgumentParser("TFRN reproduction")
    # data
    p.add_argument("--dataset", default="radioml2018", choices=list(DATASET_SPECS))
    p.add_argument("--data-path", default="")
    p.add_argument("--csv-path", default="")
    p.add_argument("--num-classes", type=int, default=None)
    p.add_argument("--max-samples", type=int, default=None,
                   help="debug: cap number of frames loaded from HDF5")
    p.add_argument("--split-seed", type=int, default=42)
    p.add_argument("--cache-tfi", action="store_true")
    # model
    p.add_argument("--model", default="tfrn",
                   choices=["tfrn", "resnet", "mcnet", "lstm2", "mcldnn", "feat", "vtmcnet"])
    p.add_argument("--mode", default="both", choices=["both", "iq", "tf"])
    p.add_argument("--fusion", default="cct", choices=["cct", "cst", "cad"])
    p.add_argument("--num-mscr", type=int, default=6)
    p.add_argument("--no-multiscale", action="store_true")
    p.add_argument("--tf-heads", type=int, default=8)
    p.add_argument("--no-transformer", action="store_true")
    p.add_argument("--d-model", type=int, default=128)
    # training
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--early-stop-patience", type=int, default=20)
    p.add_argument("--scheduler-patience", type=int, default=10)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--device", default="auto")
    p.add_argument("--amp", action="store_true")
    p.add_argument("--grad-clip", type=float, default=5.0)
    p.add_argument("--tfi-log", action="store_true")
    p.add_argument("--tfi-norm", default="max", choices=["max", "none", "zscore"])
    # output
    p.add_argument("--out-dir", default="./runs")
    p.add_argument("--exp-name", default=None)
    p.add_argument("--no-save-ckpt", action="store_true")
    p.add_argument("--complexity", action="store_true",
                   help="print param/FLOP estimate and exit")
    return p.parse_args()


def build_cfg(args) -> CFG:
    spec = DATASET_SPECS[args.dataset]
    cfg = CFG(
        dataset=args.dataset, data_path=args.data_path, csv_path=args.csv_path,
        num_classes=args.num_classes or spec["num_classes"],
        snr_levels=list(spec["snr_levels"]),
        split_seed=args.split_seed,
        d_model=args.d_model, num_mscr=args.num_mscr,
        mscr_no_multiscale=args.no_multiscale,
        tf_heads=args.tf_heads, use_transformer=not args.no_transformer,
        fusion=args.fusion, mode=args.mode,
        epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
        weight_decay=args.weight_decay,
        early_stop_patience=args.early_stop_patience,
        scheduler_patience=args.scheduler_patience,
        seeds=list(args.seeds), num_workers=args.num_workers,
        device=args.device, amp=args.amp, grad_clip=args.grad_clip,
        tfi_log=args.tfi_log, tfi_norm=args.tfi_norm,
        out_dir=args.out_dir, save_ckpt=not args.no_save_ckpt,
        exp_name=args.exp_name or f"{args.dataset}_{args.model}_"
                                  f"{args.fusion if args.model == 'tfrn' else 'base'}",
    )
    return cfg


def make_model(args, cfg, device):
    if args.model == "tfrn":
        model = build_model(cfg, cfg.num_classes)
    else:
        model = build_baseline(args.model, cfg.num_classes)
    return model.to(device)


@torch.no_grad()
def evaluate(model, loader, criterion, device, amp=False):
    model.eval()
    preds_all, labels_all, snrs_all = [], [], []
    loss_sum, n_batches = 0.0, 0
    for iq, tfi, y, snr in loader:
        iq, tfi, y = iq.to(device), tfi.to(device), y.to(device)
        with torch.amp.autocast(device_type=device.type, enabled=amp):
            logits = model(iq, tfi)
            loss = criterion(logits, y)
        loss_sum += float(loss.item())
        n_batches += 1
        preds_all.append(logits.argmax(1).cpu().numpy())
        labels_all.append(y.cpu().numpy())
        snrs_all.append(snr.numpy())
    preds = np.concatenate(preds_all)
    labels = np.concatenate(labels_all)
    snrs = np.concatenate(snrs_all)
    per = per_snr_accuracy(labels, preds, snrs, snr_levels=None)
    return per, summarize(per), loss_sum / max(n_batches, 1)


def train_one_epoch(model, loader, optimizer, criterion, scaler, device, cfg, epoch):
    model.train()
    running, nb = 0.0, 0
    for step, (iq, tfi, y, _) in enumerate(loader):
        iq, tfi, y = iq.to(device), tfi.to(device), y.to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=device.type, enabled=cfg.amp):
            logits = model(iq, tfi)
            loss = criterion(logits, y)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        running += float(loss.item())
        nb += 1
        if cfg.log_every and (step + 1) % cfg.log_every == 0:
            print(f"  epoch {epoch} step {step+1}/{len(loader)} loss={running/nb:.4f}",
                  flush=True)
    return running / max(nb, 1)


def run(args, cfg):
    device = torch.device(
        "cuda" if torch.cuda.is_available() and args.device in ("auto", "cuda") else "cpu")
    print(f"[setup] device={device}  dataset={cfg.dataset}  model={args.model}  "
          f"classes={cfg.num_classes}  seeds={cfg.seeds}")

    # one fixed, class/SNR-stratified split shared by every model and seed
    _, iq_all, labels_all, snrs_all = load_dataset(cfg, max_samples=args.max_samples)
    tr, va, te = stratified_split(labels_all, snrs_all, cfg.split_ratio, cfg.split_seed)
    print(f"[setup] split train/val/test = {len(tr)}/{len(va)}/{len(te)}")

    datasets, _, _, _ = load_dataset(cfg, split_indices=(tr, va, te),
                                     cache_tfi=args.cache_tfi,
                                     arrays=(iq_all, labels_all, snrs_all))

    def loader(ds, shuffle, seed):
        return DataLoader(ds, batch_size=cfg.batch_size, shuffle=shuffle,
                          num_workers=cfg.num_workers,
                          generator=torch.Generator().manual_seed(seed) if shuffle else None,
                          pin_memory=(device.type == "cuda"), drop_last=False)

    os.makedirs(cfg.out_dir, exist_ok=True)
    exp_dir = os.path.join(cfg.out_dir, cfg.exp_name)
    os.makedirs(exp_dir, exist_ok=True)

    # complexity estimate (once)
    if args.complexity:
        m = make_model(args, cfg, device)
        print("[complexity]", model_complexity(m, device, cfg.iq_length,
                                               cfg.stft_n, 50))
        return

    results = {"config": cfg.to_dict(), "model": args.model, "seeds": {}}
    for seed in cfg.seeds:
        t0 = time.time()
        set_seed(seed)
        model = make_model(args, cfg, device)
        train_loader = loader(datasets[0], True, seed)
        val_loader = loader(datasets[1], False, seed)
        test_loader = loader(datasets[2], False, seed)

        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr,
                                      weight_decay=cfg.weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=cfg.scheduler_factor,
            patience=cfg.scheduler_patience)
        scaler = torch.amp.GradScaler(enabled=cfg.amp)
        criterion = nn.CrossEntropyLoss()

        best_val, bad_epochs, best_state = float("inf"), 0, None
        history = []
        for ep in range(1, cfg.epochs + 1):
            tr_loss = train_one_epoch(model, train_loader, optimizer, criterion,
                                      scaler, device, cfg, ep)
            _, val_sum, val_loss = evaluate(model, val_loader, criterion, device, cfg.amp)
            scheduler.step(val_loss)
            history.append(dict(epoch=ep, train_loss=tr_loss, val_loss=val_loss,
                                val_avg=val_sum["avg"]))
            if val_loss < best_val - 1e-5:
                best_val = val_loss
                bad_epochs = 0
                best_state = deepcopy(model.state_dict())
            else:
                bad_epochs += 1
            if ep % 10 == 0 or ep == 1:
                print(f"[seed {seed}] ep {ep:3d} train {tr_loss:.4f} "
                      f"val {val_loss:.4f} valAvg {val_sum['avg']*100:.2f}%", flush=True)
            if bad_epochs >= cfg.early_stop_patience:
                print(f"[seed {seed}] early stop at epoch {ep}")
                break

        if best_state is not None:
            model.load_state_dict(best_state)
        per, summary, test_loss = evaluate(model, test_loader, criterion, device, cfg.amp)
        results["seeds"][str(seed)] = dict(per_snr=per, **summary,
                                           test_loss=test_loss,
                                           epochs_run=len(history), history=history)
        print(f"[seed {seed}] TEST avg={summary['avg']*100:.2f}% "
              f"low={summary['low']*100:.2f}% peak={summary['peak']*100:.2f}% "
              f"({time.time()-t0:.0f}s)")

        if cfg.save_ckpt:
            torch.save(dict(state_dict=model.state_dict(), cfg=cfg.to_dict(),
                            model=args.model, seed=seed),
                       os.path.join(exp_dir, f"seed{seed}.pt"))

    # aggregate over seeds (paper: mean +/- std of average accuracy)
    for metric in ("avg", "low", "peak"):
        vals = [results["seeds"][str(s)][metric] for s in cfg.seeds]
        mu, sd = mean_std(vals)
        results[f"{metric}_mean"] = mu
        results[f"{metric}_std"] = sd
    print(f"[summary] {cfg.exp_name}: Avg {results['avg_mean']*100:.2f}+/-"
          f"{results['avg_std']*100:.2f}%  Low {results['low_mean']*100:.2f}%  "
          f"Peak {results['peak_mean']*100:.2f}%")

    with open(os.path.join(exp_dir, "results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"[done] results -> {os.path.join(exp_dir, 'results.json')}")
    return results


if __name__ == "__main__":
    args = parse_args()
    run(args, build_cfg(args))
