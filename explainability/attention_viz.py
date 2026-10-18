"""
Attention Visualization Module
================================
Visualizes fusion attention weights and modality contributions.
"""

import os
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch


def extract_fusion_attention(
    model: torch.nn.Module,
    seq_embeddings: torch.Tensor,
    node_features: torch.Tensor,
    edge_index: torch.Tensor,
    edge_weight: Optional[torch.Tensor] = None,
    node_indices: Optional[torch.Tensor] = None,
    device: torch.device = torch.device("cpu"),
) -> Dict[str, np.ndarray]:
    """
    Extract fusion attention weights from the multimodal model.

    Args:
        model: Trained multimodal model.
        seq_embeddings: Sequence embeddings (B, D).
        node_features: Graph node features.
        edge_index: Edge index.
        edge_weight: Edge weights.
        node_indices: Node indices for batch.
        device: Compute device.

    Returns:
        Dictionary with attention weights and modality contributions.
    """
    model.eval()
    with torch.no_grad():
        output = model(
            seq_embeddings.to(device),
            node_features.to(device),
            edge_index.to(device),
            edge_weight.to(device) if edge_weight is not None else None,
            node_indices.to(device) if node_indices is not None else None,
        )

    fusion_info = output.get("fusion_info", {})

    result = {}
    for key, value in fusion_info.items():
        if isinstance(value, torch.Tensor):
            result[key] = value.cpu().numpy()
        else:
            result[key] = value

    return result


def plot_modality_contribution_distribution(
    seq_weights: np.ndarray,
    net_weights: np.ndarray,
    output_path: str = "outputs/figures/modality_distribution.png",
) -> None:
    """
    Plot distribution of modality contributions across proteins.

    Args:
        seq_weights: Sequence attention weights (N,).
        net_weights: Network attention weights (N,).
        output_path: Output path.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Distribution plot
    ax = axes[0]
    ax.hist(seq_weights.flatten(), bins=50, alpha=0.7, color="#2196F3", label="Sequence", density=True)
    ax.hist(net_weights.flatten(), bins=50, alpha=0.7, color="#FF5722", label="Network", density=True)
    ax.set_xlabel("Attention Weight")
    ax.set_ylabel("Density")
    ax.set_title("Modality Weight Distribution")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Scatter of seq vs net weights
    ax = axes[1]
    ax.scatter(seq_weights.flatten(), net_weights.flatten(), s=5, alpha=0.3, c="#4CAF50")
    ax.plot([0, 1], [1, 0], "k--", lw=1, alpha=0.5)
    ax.set_xlabel("Sequence Weight")
    ax.set_ylabel("Network Weight")
    ax.set_title("Sequence vs Network Contribution")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_protein_attention_analysis(
    attention_data: Dict[str, np.ndarray],
    protein_names: List[str],
    output_path: str = "outputs/figures/protein_attention.png",
    top_k: int = 20,
) -> None:
    """
    Analyze and visualize attention patterns for specific proteins.

    Shows which proteins rely more on sequence vs network context.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    seq_w = attention_data["seq_weight"].flatten()
    net_w = attention_data["net_weight"].flatten()

    # Find most sequence-dependent proteins
    seq_dominant = np.argsort(seq_w)[-top_k:]
    # Find most network-dependent proteins
    net_dominant = np.argsort(net_w)[-top_k:]

    fig, axes = plt.subplots(1, 2, figsize=(16, max(8, top_k * 0.35)))

    # Sequence-dominant
    ax = axes[0]
    names = [protein_names[i][:20] if i < len(protein_names) else f"P{i}" for i in seq_dominant]
    weights = np.stack([seq_w[seq_dominant], net_w[seq_dominant]], axis=1)
    y_pos = np.arange(len(names))
    ax.barh(y_pos, weights[:, 0], color="#2196F3", alpha=0.8, label="Sequence")
    ax.barh(y_pos, weights[:, 1], left=weights[:, 0], color="#FF5722", alpha=0.8, label="Network")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names)
    ax.set_xlabel("Attention Weight")
    ax.set_title("Most Sequence-Dependent Proteins")
    ax.legend()

    # Network-dominant
    ax = axes[1]
    names = [protein_names[i][:20] if i < len(protein_names) else f"P{i}" for i in net_dominant]
    weights = np.stack([seq_w[net_dominant], net_w[net_dominant]], axis=1)
    y_pos = np.arange(len(names))
    ax.barh(y_pos, weights[:, 0], color="#2196F3", alpha=0.8, label="Sequence")
    ax.barh(y_pos, weights[:, 1], left=weights[:, 0], color="#FF5722", alpha=0.8, label="Network")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names)
    ax.set_xlabel("Attention Weight")
    ax.set_title("Most Network-Dependent Proteins")
    ax.legend()

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
