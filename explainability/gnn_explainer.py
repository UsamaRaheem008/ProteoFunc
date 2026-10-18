"""
GNNExplainer Module
====================
Explains graph neural network predictions by identifying important
subgraph structures and node/edge contributions.
"""

import os
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

try:
    from torch_geometric.explain import Explainer, GNNExplainer as PYGExplainer
    HAS_PYG_EXPLAIN = True
except ImportError:
    HAS_PYG_EXPLAIN = False

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.logger import get_logger

logger = get_logger(__name__)


class GNNExplainerWrapper:
    """
    Wraps GNNExplainer for protein function prediction interpretability.

    Identifies the most important PPI neighbors and edges for a given
    protein's function prediction.
    """

    def __init__(
        self,
        model: nn.Module,
        epochs: int = 200,
        lr: float = 0.01,
        top_k_edges: int = 10,
        top_k_nodes: int = 5,
    ):
        """
        Args:
            model: Trained graph model (network_only or multimodal's graph_encoder).
            epochs: Training epochs for GNNExplainer.
            lr: Learning rate for explanation optimization.
            top_k_edges: Number of top edges to return.
            top_k_nodes: Number of top nodes to return.
        """
        self.model = model
        self.epochs = epochs
        self.lr = lr
        self.top_k_edges = top_k_edges
        self.top_k_nodes = top_k_nodes

    def explain_node(
        self,
        node_idx: int,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: Optional[torch.Tensor] = None,
        idx_to_protein: Optional[Dict[int, str]] = None,
    ) -> Dict[str, Any]:
        """
        Explain prediction for a single protein node.

        Uses learned edge/node masks to identify the important subgraph.

        Args:
            node_idx: Index of the target node.
            x: Node feature matrix.
            edge_index: Edge index tensor.
            edge_weight: Edge weights.
            idx_to_protein: Mapping from node index to protein accession.

        Returns:
            Dictionary with explanation results.
        """
        logger.info(f"Explaining node {node_idx}...")

        if HAS_PYG_EXPLAIN:
            return self._explain_with_pyg(node_idx, x, edge_index, edge_weight, idx_to_protein)
        else:
            return self._explain_manual(node_idx, x, edge_index, edge_weight, idx_to_protein)

    def _explain_manual(
        self,
        node_idx: int,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: Optional[torch.Tensor],
        idx_to_protein: Optional[Dict[int, str]],
    ) -> Dict[str, Any]:
        """
        Manual GNNExplainer implementation for environments without PyG explain module.

        Learns soft edge masks that maximize the model's prediction for the target node.
        """
        device = x.device

        # Find k-hop subgraph around target node
        neighbors = set()
        current_layer = {node_idx}

        for _ in range(2):  # 2-hop neighborhood
            next_layer = set()
            for n in current_layer:
                # Find edges connected to node n
                mask_src = edge_index[0] == n
                mask_dst = edge_index[1] == n
                connected = edge_index[1, mask_src].tolist() + edge_index[0, mask_dst].tolist()
                next_layer.update(connected)
            neighbors.update(current_layer)
            current_layer = next_layer - neighbors

        neighbors.update(current_layer)
        subgraph_nodes = sorted(neighbors)

        # Find edges within subgraph
        node_set = set(subgraph_nodes)
        edge_mask_sub = torch.tensor([
            (edge_index[0, i].item() in node_set and edge_index[1, i].item() in node_set)
            for i in range(edge_index.shape[1])
        ], device=device)

        subgraph_edge_idx = torch.where(edge_mask_sub)[0]

        # Initialize learnable edge mask
        edge_mask = torch.nn.Parameter(
            torch.randn(len(subgraph_edge_idx), device=device) * 0.1
        )
        optimizer = torch.optim.Adam([edge_mask], lr=self.lr)

        # Get original prediction
        self.model.eval()
        with torch.no_grad():
            if hasattr(self.model, 'graph_encoder'):
                original_emb = self.model.graph_encoder(x, edge_index, edge_weight)
            else:
                original_emb = self.model(x, edge_index, edge_weight)
            original_pred = original_emb[node_idx]

        # Optimize edge mask
        for epoch in range(self.epochs):
            optimizer.zero_grad()

            # Apply sigmoid mask to edge weights
            mask_sigmoid = torch.sigmoid(edge_mask)

            # Create masked edge weights
            if edge_weight is not None:
                masked_weights = edge_weight.clone()
            else:
                masked_weights = torch.ones(edge_index.shape[1], device=device)

            masked_weights[subgraph_edge_idx] *= mask_sigmoid

            # Forward pass with masked graph
            if hasattr(self.model, 'graph_encoder'):
                masked_emb = self.model.graph_encoder(x, edge_index, masked_weights)
            else:
                masked_emb = self.model(x, edge_index, masked_weights)

            # Loss: maximize similarity to original prediction + sparsity
            pred = masked_emb[node_idx]
            sim_loss = -torch.nn.functional.cosine_similarity(
                pred.unsqueeze(0), original_pred.unsqueeze(0)
            )
            sparsity_loss = 0.01 * mask_sigmoid.sum()
            entropy_loss = -0.1 * (mask_sigmoid * torch.log(mask_sigmoid + 1e-10) +
                                    (1 - mask_sigmoid) * torch.log(1 - mask_sigmoid + 1e-10)).mean()

            loss = sim_loss + sparsity_loss + entropy_loss
            loss.backward()
            optimizer.step()

        # Extract important edges and nodes
        final_mask = torch.sigmoid(edge_mask).detach().cpu().numpy()
        top_edge_indices = np.argsort(final_mask)[-self.top_k_edges:]

        important_edges = []
        important_nodes = set()
        for idx in top_edge_indices:
            edge_idx_global = subgraph_edge_idx[idx].item()
            src = edge_index[0, edge_idx_global].item()
            dst = edge_index[1, edge_idx_global].item()
            importance = float(final_mask[idx])

            src_name = idx_to_protein.get(src, str(src)) if idx_to_protein else str(src)
            dst_name = idx_to_protein.get(dst, str(dst)) if idx_to_protein else str(dst)

            important_edges.append({
                "source": src_name,
                "target": dst_name,
                "importance": importance,
            })
            important_nodes.add(src)
            important_nodes.add(dst)

        # Node importance: sum of adjacent edge masks
        node_importance = {}
        for idx, mask_val in enumerate(final_mask):
            edge_idx_global = subgraph_edge_idx[idx].item()
            src = edge_index[0, edge_idx_global].item()
            dst = edge_index[1, edge_idx_global].item()
            node_importance[src] = node_importance.get(src, 0) + mask_val
            node_importance[dst] = node_importance.get(dst, 0) + mask_val

        top_nodes = sorted(node_importance.items(), key=lambda x: x[1], reverse=True)[:self.top_k_nodes]
        important_node_list = []
        for node_id, importance in top_nodes:
            name = idx_to_protein.get(node_id, str(node_id)) if idx_to_protein else str(node_id)
            important_node_list.append({"node": name, "importance": float(importance)})

        target_name = idx_to_protein.get(node_idx, str(node_idx)) if idx_to_protein else str(node_idx)

        return {
            "target_protein": target_name,
            "important_edges": important_edges,
            "important_nodes": important_node_list,
            "subgraph_size": len(subgraph_nodes),
            "edge_mask": final_mask,
        }

    def _explain_with_pyg(
        self,
        node_idx: int,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: Optional[torch.Tensor],
        idx_to_protein: Optional[Dict[int, str]],
    ) -> Dict[str, Any]:
        """Use PyG's built-in GNNExplainer."""
        explainer = Explainer(
            model=self.model,
            algorithm=PYGExplainer(epochs=self.epochs, lr=self.lr),
            explanation_type="model",
            node_mask_type="attributes",
            edge_mask_type="object",
        )

        explanation = explainer(x, edge_index, index=node_idx)

        edge_mask = explanation.edge_mask.cpu().numpy()
        top_edges = np.argsort(edge_mask)[-self.top_k_edges:]

        important_edges = []
        for idx in top_edges:
            src = edge_index[0, idx].item()
            dst = edge_index[1, idx].item()
            src_name = idx_to_protein.get(src, str(src)) if idx_to_protein else str(src)
            dst_name = idx_to_protein.get(dst, str(dst)) if idx_to_protein else str(dst)
            important_edges.append({
                "source": src_name,
                "target": dst_name,
                "importance": float(edge_mask[idx]),
            })

        target_name = idx_to_protein.get(node_idx, str(node_idx)) if idx_to_protein else str(node_idx)

        return {
            "target_protein": target_name,
            "important_edges": important_edges,
            "edge_mask": edge_mask,
        }


def visualize_explanation(
    explanation: Dict[str, Any],
    output_path: str = "outputs/figures/gnn_explanation.png",
) -> None:
    """
    Visualize GNNExplainer results as a subgraph diagram.

    Args:
        explanation: GNNExplainer output dictionary.
        output_path: Output file path.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    try:
        import networkx as nx
    except ImportError:
        logger.warning("networkx required for visualization")
        return

    G = nx.Graph()
    target = explanation["target_protein"]
    G.add_node(target)

    for edge in explanation.get("important_edges", []):
        G.add_edge(edge["source"], edge["target"], weight=edge["importance"])

    fig, ax = plt.subplots(figsize=(12, 10))

    pos = nx.spring_layout(G, seed=42, k=2)
    edges = G.edges(data=True)
    weights = [d.get("weight", 0.5) * 3 for (u, v, d) in edges]
    edge_colors = [d.get("weight", 0.5) for (u, v, d) in edges]

    # Draw nodes
    node_colors = ["#FF5722" if n == target else "#2196F3" for n in G.nodes()]
    node_sizes = [800 if n == target else 400 for n in G.nodes()]

    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes, ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=8, ax=ax)

    if edges:
        nx.draw_networkx_edges(
            G, pos, width=weights,
            edge_color=edge_colors, edge_cmap=plt.cm.Reds,
            alpha=0.7, ax=ax,
        )

    ax.set_title(f"GNNExplainer: Important PPI Context for {target}", fontsize=14)
    ax.axis("off")

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    logger.info(f"Saved GNN explanation plot to {output_path}")
