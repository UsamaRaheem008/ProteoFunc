"""
Helper Utilities
================
Shared helper functions used across the project.
"""

import json
import os
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml
import numpy as np
import torch


def load_config(config_path: str = "configs/config.yaml") -> Dict[str, Any]:
    """
    Load YAML configuration file.

    Args:
        config_path: Path to the YAML config file.

    Returns:
        Dictionary of configuration parameters.
    """
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config


def save_json(data: Any, path: str) -> None:
    """Save data as JSON."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)


def load_json(path: str) -> Any:
    """Load data from JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_pickle(obj: Any, path: str) -> None:
    """Save object as pickle."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_pickle(path: str) -> Any:
    """Load object from pickle."""
    with open(path, "rb") as f:
        return pickle.load(f)


def ensure_dir(path: str) -> Path:
    """Create directory if it does not exist and return Path object."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def count_parameters(model: torch.nn.Module, trainable_only: bool = True) -> int:
    """
    Count model parameters.

    Args:
        model: PyTorch model.
        trainable_only: If True, count only trainable parameters.

    Returns:
        Number of parameters.
    """
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def format_number(n: int) -> str:
    """Format large numbers with K/M/B suffixes."""
    if n >= 1_000_000_000:
        return f"{n / 1_000_000_000:.2f}B"
    if n >= 1_000_000:
        return f"{n / 1_000_000:.2f}M"
    if n >= 1_000:
        return f"{n / 1_000:.2f}K"
    return str(n)


def compute_class_weights(
    label_matrix: np.ndarray, cap: float = 10.0
) -> torch.Tensor:
    """
    Compute positive class weights for imbalanced multi-label classification.

    Args:
        label_matrix: Binary label matrix of shape (N, C).
        cap: Maximum weight cap to prevent extreme values.

    Returns:
        Tensor of shape (C,) with positive class weights.
    """
    pos_counts = label_matrix.sum(axis=0).astype(float)
    neg_counts = label_matrix.shape[0] - pos_counts
    # Avoid division by zero
    pos_counts = np.clip(pos_counts, 1, None)
    weights = neg_counts / pos_counts
    weights = np.clip(weights, 1.0, cap)
    return torch.tensor(weights, dtype=torch.float32)
