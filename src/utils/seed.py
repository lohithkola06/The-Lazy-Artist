# src/utils/seed.py
from __future__ import annotations

import os
import random
from dataclasses import dataclass

import numpy as np

try:
    import torch
except Exception:  # pragma: no cover
    torch = None


@dataclass(frozen=True)
class SeedConfig:
    seed: int = 42
    deterministic: bool = True  # ensures reproducibility (slight performance cost, acceptable for this project)


def seed_everything(cfg: SeedConfig | int) -> int:
    """
    Seed Python, NumPy, and PyTorch (if available).
    Returns the seed used.
    """
    seed = cfg.seed if isinstance(cfg, SeedConfig) else int(cfg)

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    if torch is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        # Optional determinism (reproducible but may be slower)
        deterministic = cfg.deterministic if isinstance(cfg, SeedConfig) else False
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        else:
            torch.backends.cudnn.benchmark = True

    return seed
