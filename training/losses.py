"""
Loss Functions Module
======================
Implements BCE, Weighted BCE, and Focal Loss for multi-label classification.
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class WeightedBCELoss(nn.Module):
    """
    Binary Cross-Entropy loss with per-class positive weights
    to handle class imbalance in multi-label prediction.
    """

    def __init__(self, pos_weight: Optional[torch.Tensor] = None):
        """
        Args:
            pos_weight: Tensor of shape (num_classes,) with positive class weights.
        """
        super().__init__()
        self.register_buffer("pos_weight", pos_weight)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute weighted BCE loss.

        Args:
            logits: Raw model output (B, C)  -  before sigmoid.
            targets: Binary label matrix (B, C).

        Returns:
            Scalar loss value.
        """
        return F.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=self.pos_weight
        )


class FocalLoss(nn.Module):
    """
    Focal Loss for multi-label classification.

    Addresses class imbalance by down-weighting easy examples and
    focusing training on hard negatives.

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Reference:
        Lin et al., "Focal Loss for Dense Object Detection", ICCV 2017
    """

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: float = 0.25,
        pos_weight: Optional[torch.Tensor] = None,
        reduction: str = "mean",
    ):
        """
        Args:
            gamma: Focusing parameter (higher = more focus on hard examples).
            alpha: Balancing factor for positive class.
            pos_weight: Optional per-class positive weights.
            reduction: Loss reduction method ("mean", "sum", "none").
        """
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self.register_buffer("pos_weight", pos_weight)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute focal loss.

        Args:
            logits: Raw logits (B, C).
            targets: Binary targets (B, C).

        Returns:
            Scalar loss value.
        """
        probs = torch.sigmoid(logits)
        # Clamp for numerical stability
        probs = torch.clamp(probs, min=1e-7, max=1 - 1e-7)

        # Focal weight
        p_t = targets * probs + (1 - targets) * (1 - probs)
        focal_weight = (1 - p_t) ** self.gamma

        # Alpha weighting
        alpha_t = targets * self.alpha + (1 - targets) * (1 - self.alpha)

        # BCE component
        bce = -targets * torch.log(probs) - (1 - targets) * torch.log(1 - probs)

        # Apply focal weight and alpha
        loss = alpha_t * focal_weight * bce

        # Apply per-class pos_weight if provided
        if self.pos_weight is not None:
            weight = targets * self.pos_weight.unsqueeze(0) + (1 - targets)
            loss = loss * weight

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


class AsymmetricLoss(nn.Module):
    """
    Asymmetric Loss for multi-label classification.

    Applies different focusing parameters for positive and negative samples.

    Reference:
        Ridnik et al., "Asymmetric Loss For Multi-Label Classification", ICCV 2021
    """

    def __init__(
        self,
        gamma_neg: float = 4.0,
        gamma_pos: float = 1.0,
        clip: float = 0.05,
        reduction: str = "mean",
    ):
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        probs = torch.clamp(probs, min=1e-7, max=1 - 1e-7)

        # Asymmetric clipping for negatives
        probs_neg = (probs - self.clip).clamp(min=1e-7)

        # Loss for positives and negatives
        loss_pos = targets * torch.log(probs) * ((1 - probs) ** self.gamma_pos)
        loss_neg = (1 - targets) * torch.log(1 - probs_neg) * (probs_neg ** self.gamma_neg)

        loss = -(loss_pos + loss_neg)

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


def get_loss_function(
    name: str,
    pos_weight: Optional[torch.Tensor] = None,
    focal_gamma: float = 2.0,
    focal_alpha: float = 0.25,
    **kwargs,
) -> nn.Module:
    """
    Factory function to get loss function by name.

    Args:
        name: Loss function name ("bce", "weighted_bce", "focal", "asymmetric").
        pos_weight: Per-class positive weights.
        focal_gamma: Focal loss gamma parameter.
        focal_alpha: Focal loss alpha parameter.
        **kwargs: Additional parameters (gamma_neg, gamma_pos, clip for ASL).

    Returns:
        Loss function module.
    """
    if name == "bce":
        return nn.BCEWithLogitsLoss()
    elif name == "weighted_bce":
        return WeightedBCELoss(pos_weight=pos_weight)
    elif name == "focal":
        return FocalLoss(gamma=focal_gamma, alpha=focal_alpha, pos_weight=pos_weight)
    elif name == "asymmetric":
        return AsymmetricLoss(
            gamma_neg=kwargs.get("gamma_neg", 4.0),
            gamma_pos=kwargs.get("gamma_pos", 1.0),
            clip=kwargs.get("clip", 0.05),
        )
    else:
        raise ValueError(f"Unknown loss function: {name}")

