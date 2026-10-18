"""
Network-Only Baseline Model
=============================
GraphSAGE encoder + MLP classifier (no sequence information).
"""

from typing import Dict, Optional

import torch
import torch.nn as nn

from models.graph_encoder import get_graph_encoder
from models.classifier import ClassificationHead


class NetworkOnlyModel(nn.Module):
    """
    Baseline model using only PPI network context.

    Pipeline: Node features -> GraphSAGE -> Classification Head
    """

    def __init__(
        self,
        input_dim: int = 1280,
        hidden_dim: int = 256,
        output_dim: int = 256,
        num_layers: int = 2,
        num_classes: int = 100,
        aggregation: str = "mean",
        graph_type: str = "sage",
        heads: int = 4,
        hidden_dims: list = None,
        dropout: float = 0.4,
        activation: str = "gelu",
        norm: str = "layernorm",
        residual: bool = True,
    ):
        """
        Args:
            input_dim: Node feature dimension.
            hidden_dim: GraphSAGE hidden dimension.
            output_dim: GraphSAGE output dimension.
            num_layers: Number of GraphSAGE layers.
            num_classes: Number of GO terms.
            aggregation: GraphSAGE aggregation method.
            graph_type: Type of graph encoder.
            heads: Number of heads for GAT.
            graph_type: Type of graph encoder.
            heads: Number of heads for GAT.
            hidden_dims: MLP hidden dimensions.
            dropout: Dropout rate.
            activation: Activation function.
            norm: Normalization type.
        """
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [256, 128]

        self.graph_encoder = get_graph_encoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            type=graph_type,
            heads=heads,
            num_layers=num_layers,
            aggregation=aggregation,
            dropout=dropout * 0.75,
        )

        self.classifier = ClassificationHead(
            input_dim=output_dim,
            num_classes=num_classes,
            hidden_dims=hidden_dims,
            dropout=dropout,
            activation=activation,
            norm=norm,
            residual=residual,
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: Optional[torch.Tensor] = None,
        node_indices: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            x: Node feature matrix (num_nodes, input_dim).
            edge_index: Edge index tensor (2, num_edges).
            edge_weight: Edge weights (num_edges,).
            node_indices: Indices of nodes to classify.

        Returns:
            Dictionary with logits and network embeddings.
        """
        # Encode graph
        node_emb = self.graph_encoder(x, edge_index, edge_weight)

        # Cast to match input dtype (fixes AMP float16/float32 mismatch)
        node_emb = node_emb.to(dtype=x.dtype)

        # Select specific nodes if indices provided
        if node_indices is not None:
            # Filter out invalid indices (-1 for proteins not in graph)
            valid_mask = (node_indices >= 0) & (node_indices < node_emb.size(0))
            
            batch_node_emb = torch.zeros(
                node_indices.size(0), node_emb.size(1),
                device=node_emb.device, dtype=node_emb.dtype
            )
            
            if valid_mask.any():
                valid_indices = node_indices[valid_mask]
                batch_node_emb[valid_mask] = node_emb[valid_indices]
                
            node_emb = batch_node_emb

        logits = self.classifier(node_emb)

        return {
            "logits": logits,
            "probabilities": torch.sigmoid(logits),
            "net_embedding": node_emb,
        }
