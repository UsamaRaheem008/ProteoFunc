"""
Classification Head Module
============================
Multi-layer perceptron for multi-label GO term prediction.
"""

from typing import List, Optional

import torch
import torch.nn as nn


class ClassificationHead(nn.Module):
    """
    Multi-label classification head with configurable MLP architecture.

    Architecture:
        Input -> [Linear -> Norm -> Activation -> Dropout] x N -> Sigmoid
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_dims: Optional[List[int]] = None,
        dropout: float = 0.4,
        activation: str = "gelu",
        norm: str = "layernorm",
        residual: bool = True,
    ):
        """
        Args:
            input_dim: Input feature dimension.
            num_classes: Number of GO term classes.
            hidden_dims: List of hidden layer dimensions.
            dropout: Dropout rate.
            activation: Activation function ("gelu" or "relu").
            norm: Normalization type ("layernorm" or "batchnorm").
        """
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [512, 256]

        # Activation function
        act_fn = nn.GELU() if activation == "gelu" else nn.ReLU()

        self.residual = residual
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        self.acts = nn.ModuleList()
        self.residual_projs = nn.ModuleList()

        prev_dim = input_dim

        for h_dim in hidden_dims:
            self.layers.append(nn.Linear(prev_dim, h_dim))

            if norm == "layernorm":
                self.norms.append(nn.LayerNorm(h_dim))
            elif norm == "batchnorm":
                self.norms.append(nn.BatchNorm1d(h_dim))
            else:
                self.norms.append(nn.Identity())

            self.acts.append(act_fn)
            self.dropouts.append(nn.Dropout(dropout))
            
            if self.residual and prev_dim != h_dim:
                self.residual_projs.append(nn.Linear(prev_dim, h_dim))
            else:
                self.residual_projs.append(None)
                
            prev_dim = h_dim

        # Final classification layer (no activation  -  sigmoid applied in loss or inference)
        self.final_layer = nn.Linear(prev_dim, num_classes)
        self.num_classes = num_classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input features (batch_size, input_dim).

        Returns:
            Raw logits (batch_size, num_classes). Apply sigmoid for probabilities.
        """
        h = x
        for i in range(len(self.layers)):
            h_prev = h
            h = self.layers[i](h)
            h = self.norms[i](h)
            
            if self.residual:
                if self.residual_projs[i] is not None:
                    h = h + self.residual_projs[i](h_prev)
                else:
                    h = h + h_prev
                    
            h = self.acts[i](h)
            h = self.dropouts[i](h)

        return self.final_layer(h)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Return sigmoid probabilities."""
        return torch.sigmoid(self.forward(x))
