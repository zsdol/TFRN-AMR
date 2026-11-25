"""Evaluation metrics (Section 5.2).

per-SNR accuracy : accuracy on test frames at a given SNR
average accuracy : mean of per-SNR accuracies over all SNR levels (26 / 11)
peak accuracy    : max per-SNR accuracy
low-SNR average  : mean over SNR levels in [-20, 0] dB
"""
from __future__ import annotations

from typing import Dict, Iterable

import numpy as np


def per_snr_accuracy(y_true: Iterable[int], y_pred: Iterable[int],
                     snrs: Iterable[float], snr_levels: Iterable[float] = None) -> Dict[float, float]:
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    snrs = np.asarray(snrs).reshape(-1)
    levels = list(snr_levels) if snr_levels is not None else sorted(set(snrs.tolist()))
    result = {}
    for s in levels:
        mask = snrs == s
        if mask.any():
            result[float(s)] = float((y_true[mask] == y_pred[mask]).mean())
    return result


def summarize(per_snr: Dict[float, float]) -> Dict[str, float]:
    vals = list(per_snr.values())
    if not vals:
        return dict(avg=0.0, low=0.0, peak=0.0)
    low = [v for s, v in per_snr.items() if -20 <= s <= 0]
    return dict(
        avg=float(np.mean(vals)),
        low=float(np.mean(low)) if low else 0.0,
        peak=float(np.max(vals)),
    )


def accuracy_from_logits(logits, labels):
    preds = logits.argmax(dim=1)
    return float((preds == labels).float().mean()), preds
