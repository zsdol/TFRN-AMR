"""Stratified 6:2:2 train/val/test split shared by all models (Section 5.2)."""
from __future__ import annotations

from collections import defaultdict

import numpy as np


def stratified_split(labels: np.ndarray, snrs: np.ndarray,
                     ratio=(0.6, 0.2, 0.2), seed: int = 42):
    """Split indices stratified jointly by (class, SNR).

    Returns three numpy index arrays (train, val, test). Every (class, SNR)
    group is shuffled with the same RNG and split independently, so the split
    is deterministic given `seed` and is identical across all trained models.
    """
    if abs(sum(ratio) - 1.0) > 1e-6:
        raise ValueError(f"split ratio must sum to 1, got {ratio}")
    rng = np.random.default_rng(seed)
    groups = defaultdict(list)
    for i, (lab, snr) in enumerate(zip(labels, snrs)):
        groups[(int(lab), float(snr))].append(i)

    train_parts, val_parts, test_parts = [], [], []
    for key in sorted(groups):
        idx = np.asarray(groups[key], dtype=np.int64)
        rng.shuffle(idx)
        n = len(idx)
        n_tr = int(round(n * ratio[0]))
        n_va = int(round(n * ratio[1]))
        # guard against tiny groups
        n_tr = min(n_tr, n)
        n_va = min(n_va, n - n_tr)
        train_parts.append(idx[:n_tr])
        val_parts.append(idx[n_tr:n_tr + n_va])
        test_parts.append(idx[n_tr + n_va:])

    train = np.concatenate(train_parts) if train_parts else np.array([], dtype=np.int64)
    val = np.concatenate(val_parts) if val_parts else np.array([], dtype=np.int64)
    test = np.concatenate(test_parts) if test_parts else np.array([], dtype=np.int64)
    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)
    return train, val, test
