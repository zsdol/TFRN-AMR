"""Central configuration for TFRN experiments.

Values marked ASSUMPTION are not fixed by the paper (the LaTeX source leaves them
as \\todo{} placeholders); they are reasonable defaults and can be overridden on
the command line in train.py / evaluate.py.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Optional


@dataclass
class CFG:
    # ------------------------- data -------------------------------------
    dataset: str = "radioml2018"          # radioml2018 | 5gradio | synthetic
    data_path: str = ""                   # HDF5 file for radioml2018 / 5gradio
    csv_path: str = ""                    # optional CSV index instead of HDF5
    num_classes: int = 24                 # 24 for RadioML2018.01a, 6 for 5GRadio
    snr_levels: list = field(default_factory=list)  # filled from data if empty
    iq_length: int = 1024                 # L0
    # STFT (Section 3): N=40, L=20, Hamming window, magnitude spectrogram
    stft_n: int = 40
    stft_hop: int = 20
    tfi_log: bool = False                 # ASSUMPTION: False = raw |X(m,k)|
    tfi_norm: str = "max"                 # max | none | zscore ; ASSUMPTION
    split_seed: int = 42                  # shared stratified 6:2:2 split
    split_ratio: tuple = (0.6, 0.2, 0.2)

    # ------------------------- IQ branch --------------------------------
    d_model: int = 128                    # d = 128
    num_mscr: int = 6                     # six MSCR blocks (ablation: 2/4/6/8)
    mscr_no_multiscale: bool = False      # ablation: single conv of equal depth

    # ------------------------- TF branch --------------------------------
    tf_heads: int = 8                     # h = 8 (ablation: 4/8/16)
    tf_ff: int = 256                      # d_ff = 256
    tf_layers: int = 2                    # two Transformer layers
    use_transformer: bool = True          # ablation: False = No Trans (GAP)
    dropout: float = 0.1                  # ASSUMPTION

    # ------------------------- fusion -----------------------------------
    fusion: str = "cct"                   # cct | cst | cad
    fusion_heads: int = 8                 # h_f = 8
    mode: str = "both"                    # both | iq | tf (single-domain ablations)

    # ------------------------- training ---------------------------------
    epochs: int = 200
    batch_size: int = 128                 # ASSUMPTION (paper: \todo{value})
    lr: float = 1e-4
    weight_decay: float = 1e-4            # ASSUMPTION (AdamW)
    scheduler_factor: float = 0.1
    scheduler_patience: int = 10          # ReduceLROnPlateau patience = 10
    early_stop_patience: int = 20         # ASSUMPTION (paper: \todo{value})
    grad_clip: float = 5.0                # ASSUMPTION
    seeds: list = field(default_factory=lambda: [0, 1, 2])
    num_workers: int = 4
    device: str = "auto"
    amp: bool = False                     # mixed precision, ASSUMPTION

    # ------------------------- output -----------------------------------
    out_dir: str = "./runs"
    exp_name: str = "tfrn"
    save_ckpt: bool = True
    log_every: int = 50

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "CFG":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


DATASET_SPECS = {
    "radioml2018": dict(num_classes=24, snr_levels=list(range(-20, 31, 2))),   # 26 levels
    "5gradio": dict(num_classes=6, snr_levels=list(range(-20, 31, 5))),         # 11 levels
    "synthetic": dict(num_classes=6, snr_levels=list(range(-20, 31, 5))),
}

# DeepSig RadioML2018.01a modulation ordering (GOLD_XYZ_OSC.0001_16.label / README)
RADIOML2018_CLASSES = [
    "OOK", "4ASK", "8ASK", "BPSK", "QPSK", "8PSK", "16PSK", "32PSK",
    "16APSK", "32APSK", "64APSK", "128APSK", "16QAM", "32QAM", "64QAM",
    "128QAM", "256QAM", "AM-SSB-WC", "AM-SSB-SC", "AM-DSB-WC", "AM-DSB-SC",
    "FM", "GMSK", "OQPSK",
]

FIVEGRADIO_CLASSES = ["BPSK", "pi/2-BPSK", "QPSK", "16QAM", "64QAM", "256QAM"]
