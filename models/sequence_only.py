"""
Sequence-Only Baseline Model
==============================
ESM-2 embeddings + MLP classifier (no network context).
"""

from typing import Dict, Optional

import torch
import torch.nn as nn

from models.sequence_encoder import SequenceEncoder
from models.classifier import ClassificationHead


class SequenceOnlyModel(nn.Module):
    """
    Baseline model using only protein sequence information.

    Pipeline: ESM-2 embedding -> Sequence Encoder -> Classification Head
    """

    def __init__(
        self,
        embedding_dim: int = 1280,
        encoder_dim: int = 512,
        num_classes: int = 100,
        hidden_dims: list = None,
        dropout: float = 0.4,
        activation: str = "gelu",
        norm: str = "layernorm",
        residual: bool = True,
    ):
        """
        Args:
            embedding_dim: ESM-2 embedding dimension.
            encoder_dim: Sequence encoder output dimension.
            num_classes: Number of GO terms.
            hidden_dims: MLP hidden dimensions.
            dropout: Dropout rate.
            activation: Activation function.
            norm: Normalization type.
        """
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [512, 256]

        self.sequence_encoder = SequenceEncoder(
            input_dim=embedding_dim,
            output_dim=encoder_dim,
            dropout=dropout * 0.5,
        )

        self.classifier = ClassificationHead(
            input_dim=encoder_dim,
            num_classes=num_classes,
            hidden_dims=hidden_dims,
            dropout=dropout,
            activation=activation,
            norm=norm,
            residual=residual,
        )

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            batch: Dictionary with "embedding" key containing ESM-2 embeddings.

        Returns:
            Dictionary with "logits" and "seq_embedding" keys.
        """
        seq_emb = self.sequence_encoder(batch["embedding"])
        logits = self.classifier(seq_emb)

        return {
            "logits": logits,
            "probabilities": torch.sigmoid(logits),
            "seq_embedding": seq_emb,
        }
