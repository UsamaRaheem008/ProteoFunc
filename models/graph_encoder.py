"""
Graph Encoder Module
=====================
GraphSAGE-based encoder for protein-protein interaction networks.

Uses PyTorch Geometric's SAGEConv layers with batch normalization,
dropout, and residual connections.
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch_geometric.nn import SAGEConv, BatchNorm
    from torch_geometric.data import Data
    HAS_PYG = True
except ImportError:
    HAS_PYG = False


class GraphSAGEEncoder(nn.Module):
    """
    GraphSAGE encoder for protein-protein interaction networks.

    Architecture:
        Input -> [SAGEConv -> BatchNorm -> ReLU -> Dropout] x num_layers -> Output

    Supports residual connections when input and output dimensions match.
    """

    def __init__(
        self,
        input_dim: int = 1280,
        hidden_dim: int = 256,
        output_dim: int = 256,
        num_layers: int = 2,
        aggregation: str = "mean",
        dropout: float = 0.3,
        batch_norm: bool = True,
        residual: bool = True,
    ):
        """
        Args:
            input_dim: Input feature dimension (ESM-2 embedding dim).
            hidden_dim: Hidden layer dimension.
            output_dim: Output embedding dimension.
            num_layers: Number of GraphSAGE layers.
            aggregation: Aggregation method ("mean", "max", "lstm").
            dropout: Dropout rate.
            batch_norm: Whether to use batch normalization.
            residual: Whether to use residual connections.
        """
        super().__init__()

        if not HAS_PYG:
            raise ImportError(
                "PyTorch Geometric is required for GraphSAGEEncoder. "
                "Install with: pip install torch-geometric"
            )

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.residual = residual

        # Input projection to match hidden dim (for residual connections)
        self.input_proj = nn.Linear(input_dim, hidden_dim) if input_dim != hidden_dim else nn.Identity()

        # GraphSAGE convolution layers
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        for i in range(num_layers):
            in_dim = hidden_dim  # After input projection
            out_dim = output_dim if i == num_layers - 1 else hidden_dim

            self.convs.append(SAGEConv(in_dim, out_dim, aggr=aggregation))

            if batch_norm:
                self.norms.append(BatchNorm(out_dim))
            else:
                self.norms.append(nn.Identity())

        # Residual projection for dimension mismatch
        if residual and hidden_dim != output_dim:
            self.residual_proj = nn.Linear(hidden_dim, output_dim)
        else:
            self.residual_proj = None

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass through GraphSAGE layers.

        Args:
            x: Node feature matrix (num_nodes, input_dim).
            edge_index: Edge index tensor (2, num_edges).
            edge_weight: Optional edge weights (num_edges,).

        Returns:
            Node embeddings (num_nodes, output_dim).
        """
        # Project input
        h = self.input_proj(x)

        for i, (conv, norm) in enumerate(zip(self.convs, self.norms)):
            h_prev = h

            # GraphSAGE convolution
            h = conv(h, edge_index)

            # Batch normalization
            h = norm(h)

            # Activation (not on last layer)
            if i < self.num_layers - 1:
                h = F.relu(h)
                h = F.dropout(h, p=self.dropout, training=self.training)

                # Residual connection (within hidden layers)
                if self.residual and h.shape == h_prev.shape:
                    h = h + h_prev
            else:
                # Last layer with residual
                if self.residual:
                    if self.residual_proj is not None:
                        h = h + self.residual_proj(h_prev)
                    elif h.shape == h_prev.shape:
                        h = h + h_prev

                h = F.relu(h)
                h = F.dropout(h, p=self.dropout, training=self.training)

        return h

    def get_output_dim(self) -> int:
        """Return the output dimension of this encoder."""
        return self.output_dim


class GraphSAGEEncoderFallback(nn.Module):
    """
    Fallback graph encoder when PyTorch Geometric is not available.
    Uses simple message passing with linear layers.
    """

    def __init__(
        self,
        input_dim: int = 1280,
        hidden_dim: int = 256,
        output_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.3,
        **kwargs,
    ):
        super().__init__()
        self.output_dim = output_dim

        layers = []
        dims = [input_dim] + [hidden_dim] * (num_layers - 1) + [output_dim]

        for i in range(num_layers):
            layers.extend([
                nn.Linear(dims[i], dims[i + 1]),
                nn.LayerNorm(dims[i + 1]),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])

        self.encoder = nn.Sequential(*layers)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor = None,
        edge_weight: torch.Tensor = None,
    ) -> torch.Tensor:
        """Simple MLP fallback (ignores graph structure)."""
        return self.encoder(x)

    def get_output_dim(self) -> int:
        return self.output_dim


class GATEncoder(nn.Module):
    """
    Graph Attention Network (GAT) encoder for protein-protein interaction networks.
    
    Architecture:
        Input -> [GATv2Conv -> BatchNorm -> ReLU -> Dropout] x num_layers -> Output
    """
    def __init__(
        self,
        input_dim: int = 1280,
        hidden_dim: int = 256,
        output_dim: int = 256,
        num_layers: int = 2,
        heads: int = 4,
        dropout: float = 0.3,
        batch_norm: bool = True,
        residual: bool = True,
        aggregation: str = "mean", # ignored, for compat
    ):
        super().__init__()
        
        if not HAS_PYG:
            raise ImportError("PyTorch Geometric is required for GATEncoder.")
            
        from torch_geometric.nn import GATv2Conv
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.residual = residual
        
        self.input_proj = nn.Linear(input_dim, hidden_dim) if input_dim != hidden_dim else nn.Identity()
        
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        
        for i in range(num_layers):
            in_dim = hidden_dim
            out_dim = output_dim if i == num_layers - 1 else hidden_dim
            concat = False if i == num_layers - 1 else True
            out_channels = out_dim // heads if concat else out_dim
            
            self.convs.append(GATv2Conv(in_dim, out_channels, heads=heads, concat=concat, dropout=dropout))
            
            if batch_norm:
                self.norms.append(BatchNorm(out_dim))
            else:
                self.norms.append(nn.Identity())
                
        if residual and hidden_dim != output_dim:
            self.residual_proj = nn.Linear(hidden_dim, output_dim)
        else:
            self.residual_proj = None

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        h = self.input_proj(x)
        
        for i, (conv, norm) in enumerate(zip(self.convs, self.norms)):
            h_prev = h
            # GATv2Conv doesn't explicitly use edge_weight, but it learns its own attention weights.
            h = conv(h, edge_index)
            h = norm(h)
            
            if i < self.num_layers - 1:
                h = F.relu(h)
                h = F.dropout(h, p=self.dropout, training=self.training)
                if self.residual and h.shape == h_prev.shape:
                    h = h + h_prev
            else:
                if self.residual:
                    if self.residual_proj is not None:
                        h = h + self.residual_proj(h_prev)
                    elif h.shape == h_prev.shape:
                        h = h + h_prev
                h = F.relu(h)
                h = F.dropout(h, p=self.dropout, training=self.training)
                
        return h

    def get_output_dim(self) -> int:
        return self.output_dim


def get_graph_encoder(
    type: str,
    input_dim: int,
    hidden_dim: int,
    output_dim: int,
    num_layers: int,
    dropout: float = 0.3,
    aggregation: str = "mean",
    heads: int = 4,
    **kwargs
) -> nn.Module:
    if type.lower() == "gat":
        return GATEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            num_layers=num_layers,
            heads=heads,
            dropout=dropout,
        )
    elif type.lower() == "sage":
        return GraphSAGEEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            num_layers=num_layers,
            aggregation=aggregation,
            dropout=dropout,
        )
    else:
        raise ValueError(f"Unknown graph encoder type: {type}")
