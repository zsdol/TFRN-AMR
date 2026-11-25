"""Dataset loaders for RadioML2018.01a, 5GRadio and a synthetic smoke-test set.

All loaders produce three aligned numpy arrays:
    iq     (N, 2, 1024) float32  -- I and Q channels
    labels (N,)         int64    -- class index
    snrs   (N,)         float32  -- SNR in dB

HDF5 convention (DeepSig RadioML2018.01a):
    X: (N, 1024, 2) complex/real frames; Y: (N, M) one-hot; Z: (N, 1) SNR.
The same X/Y/Z convention is assumed for the released 5GRadio HDF5 file.
Common alternative key names are also accepted (see _read_hdf5).
"""
from __future__ import annotations

import os
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from .stft import iq_to_tfi


def _stratified_subset_indices(labels: np.ndarray, snrs: np.ndarray,
                               max_samples: int, seed: int = 42) -> np.ndarray:
    """Pick max_samples rows evenly across (class, SNR) groups.

    RadioML2018.01a is stored ordered by modulation and SNR, so taking the
    first N rows would yield a single class. We instead draw evenly from
    every (class, SNR) group and return sorted row indices (h5py requires
    monotonically increasing indices for fancy indexing).
    """
    rng = np.random.default_rng(seed)
    groups = {}
    for i, (lab, snr) in enumerate(zip(labels, snrs)):
        groups.setdefault((int(lab), float(snr)), []).append(i)
    groups = {k: np.asarray(v, dtype=np.int64) for k, v in groups.items()}
    n_groups, total = len(groups), len(labels)
    per_group = max(1, max_samples // n_groups)
    chosen = []
    for arr in groups.values():
        k = min(per_group, len(arr))
        chosen.append(rng.choice(arr, size=k, replace=False))
    sel = np.concatenate(chosen) if chosen else np.array([], dtype=np.int64)
    if len(sel) > max_samples:
        sel = rng.choice(sel, size=max_samples, replace=False)
    return np.sort(sel)


def _read_hdf5(path: str, max_samples: Optional[int] = None):
    """Read HDF5 frames; with max_samples, subsample evenly by class and SNR.

    When max_samples is None the full X array is loaded, which for
    RadioML2018.01a means ~21 GB RAM (2,555,904 x 1024 x 2 x float32).
    Use --max-samples on machines without enough memory.
    """
    import h5py
    with h5py.File(path, "r") as f:
        keys = list(f.keys())
        xkey = next((k for k in ["X", "X_data", "iq", "data", "samples"] if k in keys), keys[0])
        ykey = next((k for k in ["Y", "Y_data", "label", "labels", "y"] if k in keys), None)
        zkey = next((k for k in ["Z", "Z_data", "snr", "snrs", "z"] if k in keys), None)
        X = f[xkey]
        n_total = X.shape[0]
        y_all = f[ykey][:] if ykey else np.zeros(n_total, dtype=np.int64)
        z_all = f[zkey][:] if zkey else np.zeros(n_total, dtype=np.float32)
        y_flat = np.argmax(y_all, axis=1) if y_all.ndim == 2 else y_all.reshape(-1)

        if max_samples is not None and max_samples < n_total:
            idx = _stratified_subset_indices(y_flat, z_all.reshape(-1),
                                             int(max_samples))
            if n_total * 1024 * 2 * 4 > 8 * 1024 ** 3:   # >8GB: read in chunks
                iq = np.empty((len(idx),) + X.shape[1:], dtype=np.float32)
                chunk = 8192
                for start in range(0, n_total, chunk):
                    end = min(n_total, start + chunk)
                    hit = np.flatnonzero((idx >= start) & (idx < end))
                    if hit.size:
                        iq[hit] = X[start:end][idx[hit] - start]
            else:
                iq = X[idx]
            labels, snrs = y_flat[idx], z_all.reshape(-1)[idx]
        else:
            iq, labels, snrs = X[:], y_flat, z_all.reshape(-1)
    return iq, labels, snrs


def _normalize_arrays(iq, labels, snrs, num_classes: Optional[int] = None):
    iq = np.asarray(iq)
    # accept (N,1024,2), (N,2,1024), or complex (N,1024)
    if np.iscomplexobj(iq):
        iq = np.stack([iq.real, iq.imag], axis=1)
    elif iq.ndim == 3 and iq.shape[-1] == 2:
        iq = np.transpose(iq, (0, 2, 1))
    elif iq.ndim == 3 and iq.shape[1] == 2:
        pass
    else:
        raise ValueError(f"Unsupported IQ shape: {iq.shape}")
    iq = iq.astype(np.float32)

    labels = np.asarray(labels)
    if labels.ndim == 2 and labels.shape[1] > 1:      # one-hot -> index
        labels = np.argmax(labels, axis=1)
    labels = labels.reshape(-1).astype(np.int64)

    snrs = np.asarray(snrs).reshape(-1).astype(np.float32)
    if len(snrs) == 1:
        snrs = np.full(len(iq), snrs[0], dtype=np.float32)
    if not (len(iq) == len(labels) == len(snrs)):
        raise ValueError(f"Length mismatch: iq={len(iq)}, labels={len(labels)}, snrs={len(snrs)}")
    if num_classes is not None:
        assert labels.max() < num_classes, "label index exceeds num_classes"
    return iq, labels, snrs


class ModulationDataset(Dataset):
    """Returns (iq, tfi, label, snr); TFI computed on the fly."""

    def __init__(self, iq: np.ndarray, labels: np.ndarray, snrs: np.ndarray,
                 n_fft: int = 40, hop: int = 20, log: bool = False, norm: str = "max",
                 cache_tfi: bool = False, indices: Optional[np.ndarray] = None):
        # Keep a single copy of the full arrays and only store row indices.
        # Fancy indexing (iq[indices]) would duplicate ~21 GB for RadioML and
        # blow up RAM on a 32 GB cloud VM.
        self.iq, self.labels, self.snrs = iq, labels, snrs
        self.indices = indices
        self.n_fft, self.hop = n_fft, hop
        self.log, self.norm = log, norm
        self._cache = [None] * len(self) if cache_tfi else None

    def __len__(self):
        return len(self.indices) if self.indices is not None else len(self.iq)

    def __getitem__(self, i: int):
        j = int(self.indices[i]) if self.indices is not None else i
        iq_i = self.iq[j]
        if self._cache is not None and self._cache[i] is not None:
            tfi = self._cache[i]
        else:
            tfi = iq_to_tfi(iq_i, self.n_fft, self.hop, self.log, self.norm)
            if self._cache is not None:
                self._cache[i] = tfi
        return (
            torch.from_numpy(np.ascontiguousarray(iq_i)),
            torch.from_numpy(tfi).unsqueeze(0),            # (1,40,50)
            int(self.labels[j]),
            float(self.snrs[j]),
        )


def _load_csv_index(csv_path: str, max_samples=None):
    """Optional CSV index with columns: path, label, snr (path -> .npy IQ file)."""
    import pandas as pd
    df = pd.read_csv(csv_path)
    if max_samples is not None:
        df = df.head(max_samples)
    iq = np.stack([np.load(p) for p in df["path"]])
    return iq, df["label"].to_numpy(), df["snr"].to_numpy()


def load_dataset(cfg, split_indices=None, max_samples=None, cache_tfi=False,
                 arrays=None):
    """Build train/val/test ModulationDataset objects from CFG.

    `arrays` may be supplied as (iq, labels, snrs) to reuse already-loaded
    data; this avoids loading a 21 GB RadioML array twice in one process.

    Returns:
        (datasets_tuple, iq_all, labels_all, snrs_all)
        where datasets_tuple = (train_ds, val_ds, test_ds); each element can be
        None when split_indices is None.
    """
    from config import DATASET_SPECS

    if arrays is not None:
        iq, labels, snrs = arrays
    elif cfg.dataset == "synthetic":
        iq, labels, snrs = make_synthetic(
            num_classes=cfg.num_classes,
            snr_levels=cfg.snr_levels or DATASET_SPECS["synthetic"]["snr_levels"],
            iq_length=cfg.iq_length,
            per_class_snr=64,
        )
    elif cfg.csv_path:
        iq, labels, snrs = _load_csv_index(cfg.csv_path, max_samples)
    else:
        if not cfg.data_path:
            raise ValueError("cfg.data_path or cfg.csv_path must be set for HDF5 datasets")
        iq, labels, snrs = _read_hdf5(cfg.data_path, max_samples)

    if arrays is None:
        iq, labels, snrs = _normalize_arrays(iq, labels, snrs, cfg.num_classes)

    if split_indices is not None:
        tr, va, te = split_indices
        datasets = tuple(
            ModulationDataset(iq, labels, snrs, cfg.stft_n, cfg.stft_hop,
                              cfg.tfi_log, cfg.tfi_norm, cache_tfi, idx)
            for idx in (tr, va, te)
        )
    else:
        datasets = (None, None, None)
    return datasets, iq, labels, snrs


def make_synthetic(num_classes=6, snr_levels=None, iq_length=1024, per_class_snr=64,
                   seed=0):
    """Generate small M-PSK/QAM-like frames with AWGN for smoke testing."""
    if snr_levels is None:
        snr_levels = list(range(-20, 31, 5))
    rng = np.random.default_rng(seed)
    iqs, labs, snrs_out = [], [], []
    for c in range(num_classes):
        order = 2 ** (c + 1)                      # 2,4,8,16,32,64 ...
        const = _constellation(order)
        for snr in snr_levels:
            for _ in range(per_class_snr):
                sym = const[rng.integers(0, order, size=iq_length)]
                sig = sym / max(np.sqrt(np.mean(np.abs(sym) ** 2)), 1e-8)
                noise_var = np.mean(np.abs(sig) ** 2) / (10 ** (snr / 10.0))
                noise = np.sqrt(noise_var / 2) * (
                    rng.standard_normal(iq_length) + 1j * rng.standard_normal(iq_length))
                x = sig + noise
                iqs.append(np.stack([x.real, x.imag], axis=0).astype(np.float32))
                labs.append(c)
                snrs_out.append(snr)
    return np.stack(iqs), np.asarray(labs, dtype=np.int64), np.asarray(snrs_out, dtype=np.float32)


def _constellation(order: int) -> np.ndarray:
    """Square QAM if order is a perfect square, else PSK."""
    m = int(round(np.sqrt(order)))
    if m * m == order:
        real = np.arange(m) - (m - 1) / 2
        grid = np.array([r + 1j * q for r in real for q in real])
        return grid / np.sqrt(np.mean(np.abs(grid) ** 2))
    ang = 2 * np.pi * np.arange(order) / order
    return np.exp(1j * ang).astype(np.complex64)
