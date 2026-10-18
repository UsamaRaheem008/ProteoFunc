"""
Seed Management
===============
Ensures deterministic and reproducible training across runs.
"""

import os
import random
from typing import Optional

import numpy as np
import torch


def set_seed(seed: int = 42, deterministic: bool = True) -> None:
    """
    Set random seeds for full reproducibility.

    Args:
        seed: Integer seed value.
        deterministic: If True, enforce deterministic CUDA operations (may reduce speed).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # PyTorch >= 1.8
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        try:
            torch.use_deterministic_algorithms(True)
        except AttributeError:
            pass  # Older PyTorch versions


def get_device(preference: str = "auto") -> torch.device:
    """
    Resolve compute device.

    Args:
        preference: One of "auto", "cuda", "cpu".

    Returns:
        torch.device instance.
    """
    if preference == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(preference)
