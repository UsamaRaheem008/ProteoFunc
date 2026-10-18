"""
Multimodal Model (SPFP)
========================
Full multimodal model combining ESM-2 sequence embeddings with
GraphSAGE PPI network context via configurable fusion mechanisms.

SPFP = Sequence-PPI Fusion Predictor
"""

from typing import Dict, Optional

import torch
import torch.nn as nn

from models.sequence_encoder import SequenceEncoder
from models.graph_encoder import get_graph_encoder
from models.fusion import get_fusion_module
from models.classifier import ClassificationHead


class MultimodalModel(nn.Module):
    """
    Multimodal protein function prediction model.

    Architecture:
        Sequence -> SequenceEncoder ──┐
                                     ├── Fusion -> ClassificationHead -> GO predictions
        PPI Graph -> GraphSAGEEncoder ┘

    The fusion mechanism is configurable (concat, weighted_sum, attention,
    gated, cross_attention) and provides interpretable modality weights.
    """

    def __init__(
        self,
        # Sequence encoder
        seq_input_dim: int = 1280,
        seq_output_dim: int = 512,
        # Graph encoder
        graph_input_dim: int = 1280,
        graph_hidden_dim: int = 256,
        graph_output_dim: int = 256,
        graph_num_layers: int = 2,
        graph_aggregation: str = "mean",
        graph_type: str = "sage",
        heads: int = 4,
        # Fusion
        fusion_method: str = "attention",
        fusion_dim: int = 512,
        fusion_num_heads: int = 4,
        # Classifier
        num_classes: int = 100,
        classifier_hidden_dims: list = None,
        # General
        dropout: float = 0.4,
        activation: str = "gelu",
        norm: str = "layernorm",
        residual: bool = True,
    ):
        """
        Args:
            seq_input_dim: ESM-2 embedding dimension.
            seq_output_dim: Sequence encoder output dimension.
            graph_input_dim: Graph node feature dimension.
            graph_hidden_dim: GraphSAGE hidden dimension.
            graph_output_dim: GraphSAGE output dimension.
            graph_num_layers: Number of GraphSAGE layers.
            graph_aggregation: GraphSAGE aggregation method.
            fusion_method: Fusion mechanism name.
            fusion_dim: Fusion output dimension.
            fusion_num_heads: Number of attention heads for fusion.
            num_classes: Number of GO terms.
            classifier_hidden_dims: MLP hidden layer dimensions.
            dropout: Dropout rate.
            activation: Activation function.
            norm: Normalization type.
        """
        super().__init__()

        if classifier_hidden_dims is None:
            classifier_hidden_dims = [512, 256]

        # Sequence encoder
        self.sequence_encoder = SequenceEncoder(
            input_dim=seq_input_dim,
            output_dim=seq_output_dim,
            dropout=dropout * 0.5,
        )

        # Graph encoder
        self.graph_encoder = get_graph_encoder(
            input_dim=graph_input_dim,
            hidden_dim=graph_hidden_dim,
            output_dim=graph_output_dim,
            type=graph_type,
            heads=heads,
            num_layers=graph_num_layers,
            aggregation=graph_aggregation,
            dropout=dropout * 0.75,
        )

        # Fusion module
        self.fusion = get_fusion_module(
            method=fusion_method,
            seq_dim=seq_output_dim,
            net_dim=graph_output_dim,
            output_dim=fusion_dim,
            num_heads=fusion_num_heads,
            dropout=dropout * 0.5,
        )

        # Classification head
        self.classifier = ClassificationHead(
            input_dim=fusion_dim,
            num_classes=num_classes,
            hidden_dims=classifier_hidden_dims,
            dropout=dropout,
            activation=activation,
            norm=norm,
            residual=residual,
        )

        self.fusion_method = fusion_method

        # Learnable embedding for proteins not present in the PPI graph.
        # Using seq_emb.clone() as fallback makes fusion combine two identical
        # vectors, which dilutes the sequence signal for non-graph proteins.
        self.missing_graph_emb = nn.Parameter(torch.zeros(1, graph_output_dim))
        nn.init.normal_(self.missing_graph_emb, mean=0, std=0.02)

    def forward(
        self,
        seq_embeddings: torch.Tensor,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: Optional[torch.Tensor] = None,
        node_indices: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the full multimodal pipeline.

        Args:
            seq_embeddings: Pre-computed ESM-2 embeddings (B, seq_input_dim).
            node_features: All graph node features (num_nodes, graph_input_dim).
            edge_index: Graph edge indices (2, num_edges).
            edge_weight: Edge weights (num_edges,).
            node_indices: Indices into graph for the current batch proteins.

        Returns:
            Dictionary with logits, probabilities, embeddings, and fusion info.
        """
        # 1. Encode sequences
        seq_emb = self.sequence_encoder(seq_embeddings)

        # 2. Encode graph (full graph forward pass)
        all_node_emb = self.graph_encoder(node_features, edge_index, edge_weight)

        # Cast graph embeddings to match sequence dtype (fixes AMP float16/float32 mismatch)
        all_node_emb = all_node_emb.to(dtype=seq_emb.dtype)

        # 3. Select batch nodes from graph
        if node_indices is not None:
            # Filter out invalid indices (proteins not in graph)
            valid_mask = (node_indices >= 0) & (node_indices < all_node_emb.size(0))
            
            # Use a learnable "missing graph" embedding for proteins not in
            # the PPI graph. This avoids feeding a duplicate of seq_emb which
            # would make fusion combine two identical vectors, diluting signal.
            batch_node_emb = self.missing_graph_emb.expand(
                seq_emb.size(0), -1
            ).clone()
                
            if valid_mask.any():
                valid_indices = node_indices[valid_mask]
                batch_node_emb[valid_mask] = all_node_emb[valid_indices]
        else:
            batch_node_emb = all_node_emb

        # 4. Fuse modalities
        fused_emb, fusion_info = self.fusion(seq_emb, batch_node_emb)

        # 5. Classify
        logits = self.classifier(fused_emb)

        return {
            "logits": logits,
            "probabilities": torch.sigmoid(logits),
            "seq_embedding": seq_emb,
            "net_embedding": batch_node_emb,
            "fused_embedding": fused_emb,
            "fusion_info": fusion_info,
        }

    def get_fusion_weights(
        self,
        seq_embeddings: torch.Tensor,
        node_features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: Optional[torch.Tensor] = None,
        node_indices: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Extract fusion attention weights for interpretability.

        Returns:
            Dictionary with per-protein modality contribution scores.
        """
        output = self.forward(seq_embeddings, node_features, edge_index, edge_weight, node_indices)
        return output["fusion_info"]
