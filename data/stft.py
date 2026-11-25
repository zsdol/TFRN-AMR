"""STFT-based time-frequency image (TFI) generation.

Paper Section 3 / Eq. (2):
    X(m,k) = sum_{n=0}^{N-1} x[n + m L] w[n] exp(-j 2 pi k n / N)
with N = 40 (Hamming window / FFT size), L = 20 (hop), L0 = 1024.
For L0=1024 this yields M = floor((1024-40)/20)+1 = 50 frames and an
N x M = 40 x 50 magnitude spectrogram used as the TF-branch input.
"""
from __future__ import annotations

import numpy as np


def iq_to_tfi(iq: np.ndarray, n_fft: int = 40, hop: int = 20,
              log: bool = False, norm: str = "max") -> np.ndarray:
    """Convert one IQ frame (2, L0) to a magnitude spectrogram (n_fft, M).

    Args:
        iq:  float array of shape (2, L0), row 0 = I, row 1 = Q.
        n_fft: window/FFT length N.
        hop: hop size L.
        log: apply log1p to the magnitude (optional, off in the paper).
        norm: 'max' (per-image divide by max), 'zscore', or 'none'.
    Returns:
        float32 array of shape (n_fft, M).
    """
    x = np.asarray(iq[0], dtype=np.complex64) + 1j * np.asarray(iq[1], dtype=np.complex64)
    length = x.shape[0]
    if length < n_fft:
        raise ValueError(f"IQ length {length} < STFT window {n_fft}")

    n_frames = 1 + (length - n_fft) // hop
    window = np.hamming(n_fft).astype(np.float32)

    # stride trick: frame m starts at m*hop
    idx = np.arange(n_fft)[None, :] + hop * np.arange(n_frames)[:, None]
    segments = x[idx] * window[None, :]
    spec = np.fft.fft(segments, n=n_fft, axis=1)     # (M, n_fft)
    tfi = np.abs(spec).T.astype(np.float32)           # (n_fft, M) = (40, 50)

    if log:
        tfi = np.log1p(tfi)
    if norm == "max":
        tfi = tfi / (np.max(tfi) + 1e-8)
    elif norm == "zscore":
        tfi = (tfi - tfi.mean()) / (tfi.std() + 1e-8)
    elif norm not in ("none", None):
        raise ValueError(f"Unknown TFI normalization: {norm}")
    return tfi
