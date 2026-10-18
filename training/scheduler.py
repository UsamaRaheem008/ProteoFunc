"""
Learning Rate Scheduler Module
================================
Configurable LR schedulers with warmup support.
"""

import math
from typing import Optional

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    ReduceLROnPlateau,
    StepLR,
    _LRScheduler,
)


class WarmupCosineScheduler(_LRScheduler):
    """
    Cosine annealing LR scheduler with linear warmup.

    During warmup:  lr = base_lr * (step / warmup_steps)
    After warmup:   lr follows cosine decay to min_lr
    """

    def __init__(
        self,
        optimizer: Optimizer,
        warmup_epochs: int = 5,
        total_epochs: int = 100,
        min_lr: float = 1e-6,
        last_epoch: int = -1,
    ):
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            # Linear warmup
            factor = (self.last_epoch + 1) / max(1, self.warmup_epochs)
            return [base_lr * factor for base_lr in self.base_lrs]
        else:
            # Cosine decay
            progress = (self.last_epoch - self.warmup_epochs) / max(
                1, self.total_epochs - self.warmup_epochs
            )
            cosine_factor = 0.5 * (1 + math.cos(math.pi * progress))
            return [
                self.min_lr + (base_lr - self.min_lr) * cosine_factor
                for base_lr in self.base_lrs
            ]


def get_scheduler(
    name: str,
    optimizer: Optimizer,
    total_epochs: int = 100,
    warmup_epochs: int = 5,
    min_lr: float = 1e-6,
    patience: int = 10,
    factor: float = 0.5,
    step_size: int = 30,
) -> dict:
    """
    Factory function to create LR scheduler.

    Args:
        name: Scheduler name ("cosine", "plateau", "step").
        optimizer: PyTorch optimizer.
        total_epochs: Total training epochs.
        warmup_epochs: Number of warmup epochs (cosine only).
        min_lr: Minimum learning rate.
        patience: Patience for plateau scheduler.
        factor: LR reduction factor for plateau/step.
        step_size: Step size for step scheduler.

    Returns:
        Dictionary with "scheduler" and "monitor" (metric name for plateau).
    """
    if name == "cosine":
        scheduler = WarmupCosineScheduler(
            optimizer,
            warmup_epochs=warmup_epochs,
            total_epochs=total_epochs,
            min_lr=min_lr,
        )
        return {"scheduler": scheduler, "monitor": None}

    elif name == "plateau":
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=factor,
            patience=patience,
            min_lr=min_lr,
            verbose=True,
        )
        return {"scheduler": scheduler, "monitor": "val_fmax"}

    elif name == "step":
        scheduler = StepLR(optimizer, step_size=step_size, gamma=factor)
        return {"scheduler": scheduler, "monitor": None}

    else:
        raise ValueError(f"Unknown scheduler: {name}")
