"""
Visualization Module
=====================
Generates publication-quality plots for evaluation results.
"""

import os
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import precision_recall_curve, roc_curve, auc
from sklearn.manifold import TSNE

# Set publication style
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 12,
    "axes.labelsize": 14,
    "axes.titlesize": 16,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})


def plot_roc_curves(
    model_results: Dict[str, Tuple[np.ndarray, np.ndarray]],
    output_path: str = "outputs/figures/roc_curves.png",
) -> None:
    """
    Plot ROC curves for multiple models.

    Args:
        model_results: Dict mapping model name -> (y_true, y_score).
        output_path: Output file path.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 8))
    colors = ["#2196F3", "#FF5722", "#4CAF50", "#9C27B0"]

    for i, (name, (y_true, y_score)) in enumerate(model_results.items()):
        # Flatten for micro-average
        fpr, tpr, _ = roc_curve(y_true.ravel(), y_score.ravel())
        roc_auc = auc(fpr, tpr)

        ax.plot(
            fpr, tpr, color=colors[i % len(colors)],
            lw=2, label=f"{name} (AUC = {roc_auc:.4f})"
        )

    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Receiver Operating Characteristic (ROC) Curves")
    ax.legend(loc="lower right")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_pr_curves(
    model_results: Dict[str, Tuple[np.ndarray, np.ndarray]],
    output_path: str = "outputs/figures/pr_curves.png",
) -> None:
    """Plot Precision-Recall curves for multiple models."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 8))
    colors = ["#2196F3", "#FF5722", "#4CAF50", "#9C27B0"]

    for i, (name, (y_true, y_score)) in enumerate(model_results.items()):
        precision, recall, _ = precision_recall_curve(y_true.ravel(), y_score.ravel())
        pr_auc = auc(recall, precision)

        ax.plot(
            recall, precision, color=colors[i % len(colors)],
            lw=2, label=f"{name} (AUPR = {pr_auc:.4f})"
        )

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curves")
    ax.legend(loc="upper right")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_performance_comparison(
    comparison_df: pd.DataFrame,
    output_path: str = "outputs/figures/model_comparison.png",
) -> None:
    """
    Plot bar chart comparing model performance across metrics.

    Args:
        comparison_df: DataFrame with models as rows and metrics as columns.
        output_path: Output path.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    metrics_to_plot = ["fmax", "auroc", "aupr", "micro_f1", "macro_f1"]
    available = [m for m in metrics_to_plot if m in comparison_df.columns]

    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(available))
    width = 0.25
    colors = ["#2196F3", "#FF5722", "#4CAF50"]

    for i, model in enumerate(comparison_df.index):
        values = [comparison_df.loc[model, m] for m in available]
        ax.bar(x + i * width, values, width, label=model, color=colors[i % len(colors)], alpha=0.85)

    ax.set_xlabel("Metric")
    ax.set_ylabel("Score")
    ax.set_title("Model Performance Comparison")
    ax.set_xticks(x + width)
    ax.set_xticklabels([m.upper().replace("_", " ") for m in available], rotation=45)
    ax.legend()
    ax.set_ylim(0, 1.05)
    ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_attention_heatmap(
    attention_weights: np.ndarray,
    protein_names: List[str],
    output_path: str = "outputs/figures/attention_heatmap.png",
    top_k: int = 30,
) -> None:
    """
    Plot attention weight heatmap showing modality contributions per protein.

    Args:
        attention_weights: (N, 2) array of [seq_weight, net_weight].
        protein_names: Protein identifiers.
        output_path: Output path.
        top_k: Number of proteins to display.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Select top-k proteins with most divergent attention
    attention_diff = np.abs(attention_weights[:, 0] - attention_weights[:, 1])
    top_indices = np.argsort(attention_diff)[-top_k:]

    data = attention_weights[top_indices]
    names = [protein_names[i] for i in top_indices]

    fig, ax = plt.subplots(figsize=(8, max(10, top_k * 0.3)))
    sns.heatmap(
        data, annot=True, fmt=".3f", cmap="RdYlBu_r",
        xticklabels=["Sequence", "Network"],
        yticklabels=names,
        ax=ax, vmin=0, vmax=1,
    )
    ax.set_title("Fusion Attention Weights per Protein")
    ax.set_ylabel("Protein")

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_embedding_projection(
    embeddings: np.ndarray,
    labels: np.ndarray,
    method: str = "tsne",
    output_path: str = "outputs/figures/embedding_projection.png",
    max_points: int = 3000,
) -> None:
    """
    Plot 2D projection of protein embeddings colored by function.

    Args:
        embeddings: Embedding matrix (N, D).
        labels: Label matrix (N, C)  -  uses most frequent label for coloring.
        method: Projection method ("tsne" or "umap").
        output_path: Output path.
        max_points: Maximum points to plot.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Subsample if too many points
    n = min(len(embeddings), max_points)
    indices = np.random.choice(len(embeddings), n, replace=False)
    emb_sub = embeddings[indices]
    lab_sub = labels[indices]

    # Get dominant label for coloring
    dominant_label = lab_sub.argmax(axis=1)

    # Project
    if method == "tsne":
        projector = TSNE(n_components=2, random_state=42, perplexity=30)
        coords = projector.fit_transform(emb_sub)
    elif method == "umap":
        try:
            import umap
            projector = umap.UMAP(n_components=2, random_state=42)
            coords = projector.fit_transform(emb_sub)
        except ImportError:
            projector = TSNE(n_components=2, random_state=42, perplexity=30)
            coords = projector.fit_transform(emb_sub)
    else:
        raise ValueError(f"Unknown method: {method}")

    fig, ax = plt.subplots(figsize=(10, 10))
    scatter = ax.scatter(
        coords[:, 0], coords[:, 1],
        c=dominant_label, cmap="tab20", s=5, alpha=0.6,
    )
    ax.set_title(f"Protein Embedding Projection ({method.upper()})")
    ax.set_xlabel("Dimension 1")
    ax.set_ylabel("Dimension 2")
    ax.grid(True, alpha=0.2)

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_degree_vs_accuracy(
    degrees: np.ndarray,
    per_protein_f1: np.ndarray,
    output_path: str = "outputs/figures/degree_vs_accuracy.png",
) -> None:
    """
    Plot protein degree vs prediction accuracy.

    Args:
        degrees: Per-protein node degrees.
        per_protein_f1: Per-protein F1 scores.
        output_path: Output path.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.scatter(degrees, per_protein_f1, s=10, alpha=0.3, c="#2196F3")

    # Binned average
    bin_edges = np.percentile(degrees[degrees > 0], np.arange(0, 101, 10))
    bin_edges = np.unique(bin_edges)

    if len(bin_edges) > 1:
        bin_means_x = []
        bin_means_y = []
        for i in range(len(bin_edges) - 1):
            mask = (degrees >= bin_edges[i]) & (degrees < bin_edges[i + 1])
            if mask.sum() > 5:
                bin_means_x.append((bin_edges[i] + bin_edges[i + 1]) / 2)
                bin_means_y.append(per_protein_f1[mask].mean())

        ax.plot(bin_means_x, bin_means_y, "r-o", lw=2, markersize=6, label="Binned mean")

    ax.set_xlabel("Node Degree (PPI Connections)")
    ax.set_ylabel("Protein-centric F1 Score")
    ax.set_title("Network Connectivity vs Prediction Accuracy")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_training_history(
    history: Dict[str, List[float]],
    output_path: str = "outputs/figures/training_history.png",
) -> None:
    """
    Plot training and validation metrics over epochs.

    Args:
        history: Training history dictionary.
        output_path: Output path.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Loss
    ax = axes[0, 0]
    ax.plot(history.get("train_loss", []), label="Train", color="#2196F3")
    ax.plot(history.get("val_loss", []), label="Validation", color="#FF5722")
    ax.set_title("Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # F-max
    ax = axes[0, 1]
    ax.plot(history.get("val_fmax", []), label="Val F-max", color="#4CAF50")
    ax.set_title("Validation F-max")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("F-max")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # AUROC
    ax = axes[1, 0]
    ax.plot(history.get("val_auroc", []), label="Val AUROC", color="#9C27B0")
    ax.set_title("Validation AUROC")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("AUROC")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Learning rate
    ax = axes[1, 1]
    ax.plot(history.get("lr", []), label="Learning Rate", color="#FF9800")
    ax.set_title("Learning Rate Schedule")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("LR")
    ax.set_yscale("log")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.suptitle("Training History", fontsize=18, y=1.02)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_confusion_heatmap(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    term_names: List[str],
    output_path: str = "outputs/figures/confusion_heatmap.png",
    top_k: int = 20,
) -> None:
    """
    Plot confusion-style heatmap for top-k GO terms.

    Args:
        y_true: True labels (N, C).
        y_pred: Predicted labels (N, C).
        term_names: GO term names.
        output_path: Output path.
        top_k: Number of terms to display.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Select top-k terms by support
    support = y_true.sum(axis=0)
    top_idx = np.argsort(support)[-top_k:]

    # Build confusion-like matrix: TP, FP, FN, TN rates
    tp = (y_true[:, top_idx] * y_pred[:, top_idx]).sum(axis=0)
    fp = ((1 - y_true[:, top_idx]) * y_pred[:, top_idx]).sum(axis=0)
    fn = (y_true[:, top_idx] * (1 - y_pred[:, top_idx])).sum(axis=0)

    precision_vals = tp / np.clip(tp + fp, 1, None)
    recall_vals = tp / np.clip(tp + fn, 1, None)
    f1_vals = 2 * precision_vals * recall_vals / np.clip(precision_vals + recall_vals, 1e-10, None)

    top_names = [term_names[i][:40] if i < len(term_names) else f"Term_{i}" for i in top_idx]

    data = np.stack([precision_vals, recall_vals, f1_vals], axis=1)
    df = pd.DataFrame(data, index=top_names, columns=["Precision", "Recall", "F1"])

    fig, ax = plt.subplots(figsize=(8, max(8, top_k * 0.35)))
    sns.heatmap(df, annot=True, fmt=".3f", cmap="YlGnBu", ax=ax, vmin=0, vmax=1)
    ax.set_title("Per-Term Performance (Top-k by Support)")

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_training_history(
    history: Dict[str, List[float]],
    output_path: str = "outputs/figures/training_history.png",
    model_name: str = "Model",
) -> None:
    """
    Plot 4-panel training history (Loss, F-max, AUROC, Learning Rate).

    Args:
        history: Training history dictionary.
        output_path: Output file path.
        model_name: Name of the model for title.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle(f"Training History  -  {model_name}", fontsize=16, fontweight="bold", y=1.01)

    epochs = range(1, len(history.get("train_loss", [])) + 1)

    # Loss
    ax = axes[0, 0]
    ax.plot(epochs, history.get("train_loss", []), color="#2196F3", lw=2, label="Train Loss")
    ax.plot(epochs, history.get("val_loss", []), color="#FF5722", lw=2, label="Val Loss", linestyle="--")
    ax.set_title("Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Val F-max
    ax = axes[0, 1]
    ax.plot(epochs, history.get("val_fmax", []), color="#4CAF50", lw=2, label="Val F-max")
    ax.set_title("Validation F-max")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("F-max")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Val AUROC
    ax = axes[1, 0]
    ax.plot(epochs, history.get("val_auroc", []), color="#9C27B0", lw=2, label="Val AUROC")
    ax.set_title("Validation AUROC")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("AUROC")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Learning Rate
    ax = axes[1, 1]
    ax.plot(epochs, history.get("lr", []), color="#FF9800", lw=2, label="Learning Rate")
    ax.set_title("Learning Rate Schedule")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("LR")
    ax.set_yscale("log")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def generate_all_plots(
    output_dir: str = "ProteoFunc_outputs/figures",
) -> None:
    """
    Generate all evaluation plots from saved results.

    Args:
        output_dir: Directory for output figures.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Check for saved results in ProteoFunc_outputs and fallback outputs
    base_dirs = ["ProteoFunc_outputs", "outputs"]

    model_results = {}
    for model_type in ["sequence_only", "network_only", "multimodal"]:
        for base in base_dirs:
            pred_path = f"{base}/evaluation/{model_type}/predictions.npy"
            label_path = f"{base}/evaluation/{model_type}/true_labels.npy"
            if os.path.exists(pred_path) and os.path.exists(label_path):
                y_score = np.load(pred_path)
                y_true = np.load(label_path)
                model_results[model_type] = (y_true, y_score)
                break

    if model_results:
        plot_roc_curves(model_results, os.path.join(output_dir, "roc_curves.png"))
        plot_pr_curves(model_results, os.path.join(output_dir, "pr_curves.png"))

    # Training histories
    from utils.helpers import load_json
    for model_type in ["sequence_only", "network_only", "multimodal"]:
        for base in base_dirs:
            hist_path = f"{base}/checkpoints/{model_type}_history.json"
            if os.path.exists(hist_path):
                history = load_json(hist_path)
                plot_training_history(
                    history,
                    os.path.join(output_dir, f"{model_type}_training_history.png"),
                    model_name=model_type.replace("_", " ").title(),
                )
                break

    # Model comparison
    for base in base_dirs:
        comp_path = f"{base}/evaluation/model_comparison.csv"
        if os.path.exists(comp_path):
            comp_df = pd.read_csv(comp_path, index_col=0)
            plot_performance_comparison(comp_df, os.path.join(output_dir, "model_comparison.png"))
            break
