"""
ProteoFunc Finalization Script
================================
Generates ALL missing outputs:
  1. Training history plots (loss, fmax, auroc, lr) for all 3 models
  2. ROC curves comparison
  3. Precision-Recall curves comparison
  4. Model performance comparison bar chart
  5. Integrated Gradients feature importance (sequence model)
  6. Fusion attention weights & modality distribution (multimodal model)

Run from the ProteoFunc/ directory:
    python finalize.py
"""

import os
import sys
import json
import numpy as np

# Fix Windows OpenMP conflict
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

OUTPUTS_DIR = os.path.join(ROOT, "ProteoFunc_outputs")
CKPT_DIR    = os.path.join(OUTPUTS_DIR, "checkpoints")
EVAL_DIR    = os.path.join(OUTPUTS_DIR, "evaluation")
FIG_DIR     = os.path.join(OUTPUTS_DIR, "figures")
EXPL_DIR    = os.path.join(OUTPUTS_DIR, "explainability")

os.makedirs(FIG_DIR,  exist_ok=True)
os.makedirs(EXPL_DIR, exist_ok=True)

MODEL_NAMES = ["sequence_only", "network_only", "multimodal"]
COLORS      = {"sequence_only": "#2196F3", "network_only": "#FF5722", "multimodal": "#4CAF50"}
LABELS      = {"sequence_only": "Sequence Only", "network_only": "Network Only", "multimodal": "Multimodal"}

# ── Matplotlib setup ───────────────────────────────────────────────────────────
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

plt.rcParams.update({
    "font.family":      "sans-serif",
    "font.size":        12,
    "axes.labelsize":   13,
    "axes.titlesize":   14,
    "xtick.labelsize":  11,
    "ytick.labelsize":  11,
    "legend.fontsize":  11,
    "figure.dpi":       120,
    "savefig.dpi":      200,
    "savefig.bbox":     "tight",
})

# ══════════════════════════════════════════════════════════════════════════════
#  STEP 1  -  Training History Plots
# ══════════════════════════════════════════════════════════════════════════════
print("\n[1/5] Generating training history plots...")

def plot_training_history(history, model_name, output_path):
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle(f"Training History  -  {LABELS[model_name]}", fontsize=16, fontweight="bold", y=1.01)

    epochs = range(1, len(history.get("train_loss", [])) + 1)

    # Loss
    ax = axes[0, 0]
    ax.plot(epochs, history.get("train_loss", []), color="#2196F3", lw=2, label="Train Loss")
    ax.plot(epochs, history.get("val_loss",   []), color="#FF5722", lw=2, label="Val Loss", linestyle="--")
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
    print(f"   ✓ Saved: {os.path.basename(output_path)}")

for model_name in MODEL_NAMES:
    hist_path = os.path.join(CKPT_DIR, f"{model_name}_history.json")
    if os.path.exists(hist_path):
        with open(hist_path) as f:
            history = json.load(f)
        plot_training_history(
            history,
            model_name,
            os.path.join(FIG_DIR, f"{model_name}_training_history.png"),
        )
    else:
        print(f"   ⚠ No history found for {model_name}")

# ══════════════════════════════════════════════════════════════════════════════
#  STEP 2  -  ROC + PR Curves Comparison
# ══════════════════════════════════════════════════════════════════════════════
print("\n[2/5] Generating ROC and Precision-Recall curves...")

from sklearn.metrics import roc_curve, precision_recall_curve, auc

model_data = {}
for model_name in MODEL_NAMES:
    pred_path  = os.path.join(EVAL_DIR, model_name, "predictions.npy")
    label_path = os.path.join(EVAL_DIR, model_name, "true_labels.npy")
    if os.path.exists(pred_path) and os.path.exists(label_path):
        model_data[model_name] = {
            "y_score": np.load(pred_path),
            "y_true":  np.load(label_path),
        }

if model_data:
    # --- ROC Curves ---
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5, label="Random")
    for model_name, data in model_data.items():
        fpr, tpr, _ = roc_curve(data["y_true"].ravel(), data["y_score"].ravel())
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=COLORS[model_name], lw=2.5,
                label=f"{LABELS[model_name]} (AUC={roc_auc:.4f})")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves  -  All Models")
    ax.legend(loc="lower right")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "roc_curves.png"))
    plt.close()
    print("   ✓ Saved: roc_curves.png")

    # --- PR Curves ---
    fig, ax = plt.subplots(figsize=(8, 7))
    for model_name, data in model_data.items():
        precision, recall, _ = precision_recall_curve(data["y_true"].ravel(), data["y_score"].ravel())
        pr_auc = auc(recall, precision)
        ax.plot(recall, precision, color=COLORS[model_name], lw=2.5,
                label=f"{LABELS[model_name]} (AUPR={pr_auc:.4f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curves  -  All Models")
    ax.legend(loc="upper right")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "pr_curves.png"))
    plt.close()
    print("   ✓ Saved: pr_curves.png")
else:
    print("   ⚠ No prediction files found  -  skipping curves")

# ══════════════════════════════════════════════════════════════════════════════
#  STEP 3  -  Model Comparison Bar Chart
# ══════════════════════════════════════════════════════════════════════════════
print("\n[3/5] Generating model comparison bar chart...")

metrics_store = {}
for model_name in MODEL_NAMES:
    metrics_path = os.path.join(EVAL_DIR, model_name, "overall_metrics.json")
    if os.path.exists(metrics_path):
        with open(metrics_path) as f:
            metrics_store[model_name] = json.load(f)

if metrics_store:
    plot_metrics = ["fmax", "auroc", "aupr", "micro_f1", "macro_f1"]
    metric_labels = ["Fmax", "AUROC", "AUPR", "Micro F1", "Macro F1"]

    x = np.arange(len(plot_metrics))
    width = 0.25

    fig, ax = plt.subplots(figsize=(13, 6))
    for i, (model_name, metrics) in enumerate(metrics_store.items()):
        values = [metrics.get(m, 0) for m in plot_metrics]
        bars = ax.bar(x + i * width, values, width,
                      label=LABELS[model_name],
                      color=COLORS[model_name], alpha=0.87, edgecolor="white")
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=8.5)

    ax.set_xlabel("Metric")
    ax.set_ylabel("Score")
    ax.set_title("Model Performance Comparison  -  All Metrics", fontweight="bold")
    ax.set_xticks(x + width)
    ax.set_xticklabels(metric_labels)
    ax.legend()
    ax.set_ylim(0, 1.05)
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "model_comparison.png"))
    plt.close()
    print("   ✓ Saved: model_comparison.png")

    # --- Namespace Stratified Chart ---
    stratified_metrics_store = {}
    for model_name in MODEL_NAMES:
        strat_path = os.path.join(EVAL_DIR, model_name, "stratified_metrics.json")
        if os.path.exists(strat_path):
            with open(strat_path) as f:
                stratified_metrics_store[model_name] = json.load(f)

    if stratified_metrics_store:
        namespaces   = ["namespace_MF", "namespace_BP", "namespace_CC"]
        ns_labels    = ["Molecular\nFunction", "Biological\nProcess", "Cellular\nComponent"]
        strat_metric = "fmax"

        fig, ax = plt.subplots(figsize=(10, 6))
        x = np.arange(len(namespaces))
        for i, (model_name, strat) in enumerate(stratified_metrics_store.items()):
            values = [strat.get(ns, {}).get(strat_metric, 0) for ns in namespaces]
            bars = ax.bar(x + i * width, values, width,
                          label=LABELS[model_name],
                          color=COLORS[model_name], alpha=0.87, edgecolor="white")
            for bar, val in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.003,
                        f"{val:.3f}", ha="center", va="bottom", fontsize=9)

        ax.set_xlabel("GO Namespace")
        ax.set_ylabel("F-max")
        ax.set_title("F-max by GO Namespace  -  All Models", fontweight="bold")
        ax.set_xticks(x + width)
        ax.set_xticklabels(ns_labels)
        ax.legend()
        ax.set_ylim(0, 0.65)
        ax.grid(True, axis="y", alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(FIG_DIR, "namespace_comparison.png"))
        plt.close()
        print("   ✓ Saved: namespace_comparison.png")

        # --- Degree Stratified Chart ---
        degree_groups  = ["low_degree", "medium_degree", "high_degree"]
        degree_labels  = ["Low\nDegree", "Medium\nDegree", "High\nDegree"]

        fig, ax = plt.subplots(figsize=(10, 6))
        x = np.arange(len(degree_groups))
        for i, (model_name, strat) in enumerate(stratified_metrics_store.items()):
            values = [strat.get(dg, {}).get(strat_metric, 0) for dg in degree_groups]
            bars = ax.bar(x + i * width, values, width,
                          label=LABELS[model_name],
                          color=COLORS[model_name], alpha=0.87, edgecolor="white")
            for bar, val in zip(bars, values):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.003,
                        f"{val:.3f}", ha="center", va="bottom", fontsize=9)

        ax.set_xlabel("Protein Network Degree")
        ax.set_ylabel("F-max")
        ax.set_title("F-max by Network Connectivity  -  All Models", fontweight="bold")
        ax.set_xticks(x + width)
        ax.set_xticklabels(degree_labels)
        ax.legend()
        ax.set_ylim(0, 0.65)
        ax.grid(True, axis="y", alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(FIG_DIR, "degree_stratified.png"))
        plt.close()
        print("   ✓ Saved: degree_stratified.png")

# ══════════════════════════════════════════════════════════════════════════════
#  STEP 4  -  Integrated Gradients (Sequence Model)
# ══════════════════════════════════════════════════════════════════════════════
print("\n[4/5] Running Integrated Gradients on sequence model...")

try:
    import torch
    from utils.helpers import load_config, load_pickle
    from utils.seed import get_device
    from training.train import build_model

    config     = load_config(os.path.join(ROOT, "configs/config.yaml"))
    # Override checkpoint/data dirs to point to ProteoFunc_outputs
    config["training"]["checkpoint"]["save_dir"] = CKPT_DIR
    device     = get_device(config["project"]["device"])
    proc_dir   = config["data"]["processed_dir"]
    emb_dir    = config["data"]["embeddings_dir"]

    split_data  = load_pickle(os.path.join(proc_dir, "split_data.pkl"))
    num_classes = split_data["num_classes"]

    seq_ckpt_path = os.path.join(CKPT_DIR, "sequence_only_best.pt")
    if os.path.exists(seq_ckpt_path):
        seq_model = build_model("sequence_only", config, num_classes)
        ckpt = torch.load(seq_ckpt_path, map_location=device, weights_only=False)
        seq_model.load_state_dict(ckpt["model_state_dict"])
        seq_model = seq_model.to(device).eval()

        embeddings = np.load(os.path.join(emb_dir, "esm2_embeddings.npy"))
        test_idx   = split_data["test_idx"][:50]
        sample_emb = torch.tensor(embeddings[test_idx], dtype=torch.float32).to(device)

        from explainability.integrated_gradients import (
            compute_integrated_gradients,
            plot_feature_importance,
        )

        ig_results = compute_integrated_gradients(
            model=seq_model,
            embeddings=sample_emb,
            target_class=0,
            n_steps=config["explainability"]["integrated_gradients"]["n_steps"],
            batch_size=50,
        )
        plot_feature_importance(
            ig_results,
            output_path=os.path.join(EXPL_DIR, "integrated_gradients.png"),
        )
        print("   ✓ Saved: integrated_gradients.png")
    else:
        print(f"   ⚠ sequence_only_best.pt not found at {seq_ckpt_path}")

except Exception as e:
    print(f"   ⚠ Integrated Gradients failed: {e}")

# ══════════════════════════════════════════════════════════════════════════════
#  STEP 5  -  Fusion Attention Weights (Multimodal Model)
# ══════════════════════════════════════════════════════════════════════════════
print("\n[5/5] Running fusion attention analysis on multimodal model...")

try:
    import torch
    from utils.helpers import load_config, load_pickle
    from utils.seed import get_device
    from training.train import build_model
    from explainability.attention_viz import (
        extract_fusion_attention,
        plot_modality_contribution_distribution,
        plot_protein_attention_analysis,
    )

    config = load_config(os.path.join(ROOT, "configs/config.yaml"))
    config["training"]["checkpoint"]["save_dir"] = CKPT_DIR
    device   = get_device(config["project"]["device"])
    proc_dir = config["data"]["processed_dir"]
    emb_dir  = config["data"]["embeddings_dir"]

    split_data  = load_pickle(os.path.join(proc_dir, "split_data.pkl"))
    num_classes = split_data["num_classes"]

    mm_ckpt_path = os.path.join(CKPT_DIR, "multimodal_best.pt")
    pyg_path     = os.path.join(proc_dir, "pyg_graph_data.pt")

    if os.path.exists(mm_ckpt_path) and os.path.exists(pyg_path):
        mm_model = build_model("multimodal", config, num_classes)
        ckpt = torch.load(mm_ckpt_path, map_location=device, weights_only=False)
        mm_model.load_state_dict(ckpt["model_state_dict"])
        mm_model = mm_model.to(device).eval()

        graph_data = torch.load(pyg_path, map_location=device, weights_only=False)
        for key in ["edge_index", "edge_weight", "x"]:
            if key in graph_data and graph_data[key] is not None:
                graph_data[key] = graph_data[key].to(device)

        embeddings = np.load(os.path.join(emb_dir, "esm2_embeddings.npy"))
        test_idx   = split_data["test_idx"][:200]
        sample_emb = torch.tensor(embeddings[test_idx], dtype=torch.float32).to(device)
        sample_acc = split_data["test_accessions"][:200]

        p2g          = graph_data.get("protein_to_idx", {})
        node_indices = torch.tensor(
            [p2g.get(acc, 0) for acc in sample_acc], dtype=torch.long
        ).to(device)

        attention_data = extract_fusion_attention(
            mm_model,
            sample_emb,
            graph_data["x"],
            graph_data["edge_index"],
            graph_data.get("edge_weight"),
            node_indices,
            device,
        )

        if "seq_weight" in attention_data and "net_weight" in attention_data:
            plot_modality_contribution_distribution(
                attention_data["seq_weight"],
                attention_data["net_weight"],
                output_path=os.path.join(EXPL_DIR, "modality_distribution.png"),
            )
            print("   ✓ Saved: modality_distribution.png")

            plot_protein_attention_analysis(
                attention_data,
                sample_acc,
                output_path=os.path.join(EXPL_DIR, "protein_attention.png"),
            )
            print("   ✓ Saved: protein_attention.png")
        else:
            # Fusion method may be 'concat' or 'weighted_sum'  -  no per-protein weights.
            # Save a summary chart with average modality scores if available.
            print("   ℹ No per-protein attention weights (fusion method may not be 'attention').")
            print("     Saving fusion info summary instead...")
            keys_found = list(attention_data.keys())
            print(f"     Available fusion keys: {keys_found}")

            # Fallback: write a simple text summary
            with open(os.path.join(EXPL_DIR, "fusion_info.txt"), "w") as f:
                f.write("Fusion attention keys returned by model:\n")
                for k, v in attention_data.items():
                    if isinstance(v, np.ndarray):
                        f.write(f"  {k}: shape={v.shape}, mean={v.mean():.4f}\n")
                    else:
                        f.write(f"  {k}: {v}\n")
            print("   ✓ Saved: fusion_info.txt")
    else:
        print(f"   ⚠ Missing checkpoint or graph data  -  skipping attention analysis")

except Exception as e:
    print(f"   ⚠ Attention analysis failed: {e}")

# ══════════════════════════════════════════════════════════════════════════════
#  DONE
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "="*65)
print("  FINALIZATION COMPLETE")
print("="*65)
print(f"\n  Figures  -> {FIG_DIR}")
print(f"  Explainability -> {EXPL_DIR}")
print("\n  Files generated:")
for fname in sorted(os.listdir(FIG_DIR)):
    print(f"    figures/{fname}")
for fname in sorted(os.listdir(EXPL_DIR)):
    print(f"    explainability/{fname}")
print()
