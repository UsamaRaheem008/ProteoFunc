"""
Integrated Gradients Module
=============================
Computes feature attributions using Integrated Gradients.
"""

import os
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.logger import get_logger

logger = get_logger(__name__)


class IntegratedGradientsExplainer:
    """
    Compute feature-level attributions using Integrated Gradients.

    For sequence embeddings, this identifies which embedding dimensions
    are most important for function prediction.
    """

    def __init__(self, model: nn.Module, n_steps: int = 50):
        """
        Args:
            model: Trained model (sequence_only or multimodal).
            n_steps: Number of interpolation steps.
        """
        self.model = model
        self.n_steps = n_steps

    def explain(
        self,
        embedding: torch.Tensor,
        target_class: int,
        baseline: Optional[torch.Tensor] = None,
    ) -> np.ndarray:
        """
        Compute integrated gradients for a single input.

        Args:
            embedding: Input embedding (1, D).
            target_class: Target GO term index.
            baseline: Baseline input (default: zeros).

        Returns:
            Attribution scores (D,).
        """
        self.model.eval()

        if baseline is None:
            baseline = torch.zeros_like(embedding)

        # Interpolation
        alphas = torch.linspace(0, 1, self.n_steps + 1, device=embedding.device)
        gradients = []

        for alpha in alphas:
            interpolated = baseline + alpha * (embedding - baseline)
            interpolated.requires_grad_(True)

            # Forward pass
            batch = {"embedding": interpolated}
            output = self.model(batch)
            logits = output["logits"]

            # Backward for target class
            self.model.zero_grad()
            logits[0, target_class].backward(retain_graph=True)

            gradients.append(interpolated.grad.detach().clone())
            interpolated.requires_grad_(False)

        # Integrate
        gradients = torch.stack(gradients, dim=0)
        avg_gradients = gradients.mean(dim=0)
        attributions = (embedding - baseline) * avg_gradients

        return attributions.detach().cpu().numpy().flatten()


def plot_feature_attributions(
    attributions: np.ndarray,
    top_k: int = 30,
    protein_name: str = "",
    go_term: str = "",
    output_path: str = "outputs/figures/ig_attributions.png",
) -> None:
    """
    Plot top feature attributions from Integrated Gradients.

    Args:
        attributions: Attribution scores (D,).
        top_k: Number of top features to plot.
        protein_name: Protein identifier.
        go_term: GO term being explained.
        output_path: Output path.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    top_indices = np.argsort(np.abs(attributions))[-top_k:]
    top_values = attributions[top_indices]

    colors = ["#FF5722" if v > 0 else "#2196F3" for v in top_values]

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(range(top_k), top_values, color=colors, alpha=0.8)
    ax.set_yticks(range(top_k))
    ax.set_yticklabels([f"Dim {i}" for i in top_indices])
    ax.set_xlabel("Attribution Score")
    ax.set_title(f"Integrated Gradients: {protein_name} -> {go_term}")
    ax.axvline(x=0, color="black", linestyle="-", linewidth=0.5)
    ax.grid(True, axis="x", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def compute_integrated_gradients(
    model: nn.Module,
    embeddings: torch.Tensor,
    target_class: int = 0,
    n_steps: int = 50,
    batch_size: int = 10,
) -> np.ndarray:
    """
    Compute integrated gradients for a batch of embeddings.

    Wrapper function called by run_pipeline.py.

    Args:
        model: Trained sequence-only model.
        embeddings: Input embeddings (N, D).
        target_class: Target GO term index.
        n_steps: Number of interpolation steps.
        batch_size: Number of proteins to explain.

    Returns:
        Attribution matrix (batch_size, D).
    """
    logger.info(f"Computing integrated gradients for {min(batch_size, len(embeddings))} proteins, "
                f"target_class={target_class}, n_steps={n_steps}")

    explainer = IntegratedGradientsExplainer(model, n_steps=n_steps)
    attributions_list = []

    n_proteins = min(batch_size, len(embeddings))
    for i in range(n_proteins):
        try:
            emb = embeddings[i:i+1]  # (1, D)
            attr = explainer.explain(emb, target_class)
            attributions_list.append(attr)
        except Exception as e:
            logger.warning(f"IG failed for protein {i}: {e}")
            attributions_list.append(np.zeros(embeddings.shape[1]))

    attributions = np.stack(attributions_list, axis=0)
    logger.info(f"Computed attributions shape: {attributions.shape}")
    return attributions


def plot_feature_importance(
    attributions: np.ndarray,
    top_k: int = 30,
    output_path: str = "outputs/explainability/integrated_gradients.png",
) -> None:
    """
    Plot aggregated feature importance from integrated gradients.

    Wrapper function called by run_pipeline.py.

    Args:
        attributions: Attribution matrix (N, D)  -  averaged across proteins.
        top_k: Number of top features to display.
        output_path: Output file path.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Average attributions across all proteins
    mean_attr = np.mean(np.abs(attributions), axis=0)

    top_indices = np.argsort(mean_attr)[-top_k:]
    top_values = mean_attr[top_indices]

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    # Plot 1: Top-k important dimensions
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, top_k))
    axes[0].barh(range(top_k), top_values, color=colors, alpha=0.85)
    axes[0].set_yticks(range(top_k))
    axes[0].set_yticklabels([f"Dim {i}" for i in top_indices], fontsize=8)
    axes[0].set_xlabel("Mean |Attribution|")
    axes[0].set_title(f"Top-{top_k} Important Embedding Dimensions")
    axes[0].grid(True, axis="x", alpha=0.3)

    # Plot 2: Full attribution distribution
    axes[1].hist(mean_attr, bins=50, color="#2196F3", alpha=0.7, edgecolor="white")
    axes[1].set_xlabel("Mean |Attribution|")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Attribution Score Distribution")
    axes[1].axvline(x=np.percentile(mean_attr, 95), color="red",
                    linestyle="--", label="95th percentile")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.suptitle("Integrated Gradients  -  Feature Importance Analysis", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Feature importance plot saved to {output_path}")

