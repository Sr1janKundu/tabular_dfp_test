"""
dfp_pipeline/utils/seed.py
---------------------------
set_seed() — global reproducibility helper.

Sets seeds for Python, NumPy, and PyTorch (CPU + CUDA).
Also sets torch.backends.cudnn.deterministic = True so conv operations
produce identical results across runs (at a small performance cost).
"""

import random
import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark     = False
