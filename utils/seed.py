"""Reproducibility helpers: seeds affect initialization and data order only;
the class/SNR-stratified split is fixed and shared by all models/seeds.
"""
from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass
    os.environ["PYTHONHASHSEED"] = str(seed)
