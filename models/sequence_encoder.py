"""
Sequence Encoder Module
========================
Wraps pre-computed ESM-2 embeddings with optional projection layers.
For inference, includes on-the-fly ESM-2 embedding extraction.
"""

import os
from typing import Optional, Tuple

import torch
import torch.nn as nn
import numpy as np


class SequenceEncoder(nn.Module):
    """
    Sequence encoder that processes pre-computed ESM-2 embeddings.

    During training, uses cached embeddings for efficiency.
    During inference, can extract embeddings on-the-fly.

    Architecture:
        ESM-2 embedding -> LayerNorm -> Linear projection -> GELU -> Dropout -> Output
    """

    def __init__(
        self,
        input_dim: int = 1280,
        output_dim: int = 512,
        dropout: float = 0.1,
        use_projection: bool = True,
    ):
        """
        Args:
            input_dim: Dimension of ESM-2 embeddings (1280 for esm2_t33_650M).
            output_dim: Output projection dimension.
            dropout: Dropout rate.
            use_projection: Whether to apply a linear projection.
        """
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim if use_projection else input_dim
        self.use_projection = use_projection

        if use_projection:
            self.projection = nn.Sequential(
                nn.LayerNorm(input_dim),
                nn.Linear(input_dim, output_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            )
        else:
            self.projection = nn.Identity()

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through sequence encoder.

        Args:
            embeddings: Pre-computed ESM-2 embeddings (batch_size, input_dim).

        Returns:
            Encoded sequence representations (batch_size, output_dim).
        """
        return self.projection(embeddings)

    def get_output_dim(self) -> int:
        """Return the output dimension of this encoder."""
        return self.output_dim
