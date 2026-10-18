"""
Evaluation Script
==================
Evaluates trained models on test set with comprehensive metrics and stratified analysis.

Usage:
    python evaluation/evaluate.py --config configs/config.yaml --model sequence_only
    python evaluation/evaluate.py --config configs/config.yaml --model all
"""

import argparse
import gc
import os
import sys
from typing import Any, Dict, List, Optional

# Prevent PyTorch CUDA memory fragmentation
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.logger import get_logger
from utils.helpers import load_config, load_pickle, save_json, ensure_dir
from utils.seed import set_seed, get_device
from evaluation.metrics import compute_all_metrics, compute_per_class_metrics, apply_true_path_rule
from preprocessing.build_dataset import build_dataset
from training.train import build_model

logger = get_logger(__name__)


@torch.no_grad()
def predict(
    model: torch.nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    model_type: str,
    graph_data: Optional[Dict] = None,
) -> tuple:
    """
    Generate predictions for a dataset.

    Args:
        model: Trained model.
        data_loader: Data loader.
        device: Compute device.
        model_type: Model variant.
        graph_data: Graph data (for network/multimodal).

    Returns:
        Tuple of (all_probs, all_labels) as numpy arrays.
    """
    model.eval()
    all_probs = []
    all_labels = []

    for batch in tqdm(data_loader, desc="Predicting", leave=False):
        labels = batch["label"]

        if model_type == "sequence_only":
            batch_gpu = {k: v.to(device) for k, v in batch.items()}
            output = model(batch_gpu)
        elif model_type == "network_only":
            node_indices = batch["graph_idx"].to(device)
            output = model(
                x=graph_data["x"],
                edge_index=graph_data["edge_index"],
                edge_weight=graph_data.get("edge_weight"),
                node_indices=node_indices,
            )
        elif model_type == "multimodal":
            seq_emb = batch["embedding"].to(device)
            node_indices = batch["graph_idx"].to(device)
            output = model(
                seq_embeddings=seq_emb,
                node_features=graph_data["x"],
                edge_index=graph_data["edge_index"],
                edge_weight=graph_data.get("edge_weight"),
                node_indices=node_indices,
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        probs = output["probabilities"].cpu().numpy()
        all_probs.append(probs)
        all_labels.append(labels.numpy())

    return np.vstack(all_probs), np.vstack(all_labels)


def stratified_evaluation(
    y_true: np.ndarray,
    y_score: np.ndarray,
    proteins_df: pd.DataFrame,
    test_accessions: List[str],
    go_metadata: pd.DataFrame,
    hub_status: Optional[Dict] = None,
    centrality: Optional[Dict] = None,
) -> Dict[str, Any]:
    """
    Evaluate model performance stratified by various criteria.

    Args:
        y_true: True labels.
        y_score: Predicted probabilities.
        proteins_df: Protein metadata.
        test_accessions: Test set accession IDs.
        go_metadata: GO term metadata with namespace/category.
        hub_status: Dictionary of protein -> is_hub.
        centrality: Centrality metrics dictionary.

    Returns:
        Stratified evaluation results.
    """
    results = {}

    # 1. By GO namespace (MF, BP, CC)
    logger.info("Stratified evaluation by GO namespace...")
    for ns in ["MF", "BP", "CC"]:
        ns_mask = go_metadata["namespace"] == ns
        if ns_mask.sum() == 0:
            continue
        ns_metrics = compute_all_metrics(y_true[:, ns_mask], y_score[:, ns_mask])
        results[f"namespace_{ns}"] = ns_metrics
        logger.info(f"  {ns}: F-max={ns_metrics['fmax']:.4f}, AUROC={ns_metrics['auroc']:.4f}")

    # 2. By GO category (Signaling vs Metabolic)
    logger.info("Stratified evaluation by GO category...")
    for cat in go_metadata["category"].unique():
        cat_mask = go_metadata["category"] == cat
        if cat_mask.sum() == 0:
            continue
        cat_metrics = compute_all_metrics(y_true[:, cat_mask], y_score[:, cat_mask])
        results[f"category_{cat}"] = cat_metrics
        logger.info(f"  {cat}: F-max={cat_metrics['fmax']:.4f}")

    # 3. By hub vs non-hub proteins
    if hub_status:
        logger.info("Stratified evaluation by hub status...")
        test_hub = [hub_status.get(acc, False) for acc in test_accessions]
        hub_mask = np.array(test_hub)

        if hub_mask.sum() > 0:
            hub_metrics = compute_all_metrics(y_true[hub_mask], y_score[hub_mask])
            results["hub_proteins"] = hub_metrics
            logger.info(f"  Hub proteins ({hub_mask.sum()}): F-max={hub_metrics['fmax']:.4f}")

        non_hub_mask = ~hub_mask
        if non_hub_mask.sum() > 0:
            non_hub_metrics = compute_all_metrics(y_true[non_hub_mask], y_score[non_hub_mask])
            results["non_hub_proteins"] = non_hub_metrics
            logger.info(f"  Non-hub proteins ({non_hub_mask.sum()}): F-max={non_hub_metrics['fmax']:.4f}")

    # 4. By network connectivity (degree quartiles)
    if centrality and "degree" in centrality:
        logger.info("Stratified evaluation by network connectivity...")
        degrees = [centrality["degree"].get(acc, 0) for acc in test_accessions]
        degree_arr = np.array(degrees)

        if degree_arr.max() > 0:
            quartiles = np.percentile(degree_arr[degree_arr > 0], [25, 50, 75])
            for q_name, (low, high) in zip(
                ["low_degree", "medium_degree", "high_degree"],
                [(0, quartiles[0]), (quartiles[0], quartiles[1]), (quartiles[1], np.inf)],
            ):
                q_mask = (degree_arr >= low) & (degree_arr < high)
                if q_mask.sum() > 5:
                    q_metrics = compute_all_metrics(y_true[q_mask], y_score[q_mask])
                    results[q_name] = q_metrics
                    logger.info(f"  {q_name} ({q_mask.sum()} proteins): F-max={q_metrics['fmax']:.4f}")

    return results


def evaluate_model(model_type: str, config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Full evaluation pipeline for a single model.

    Args:
        model_type: Model variant name.
        config: Project configuration.

    Returns:
        Complete evaluation results.
    """
    logger.info("=" * 70)
    logger.info(f"EVALUATING: {model_type.upper()}")
    logger.info("=" * 70)

    device = get_device(config["project"]["device"])
    set_seed(config["project"]["seed"])
    data_cfg = config["data"]
    processed_dir = data_cfg["processed_dir"]

    # Build dataset
    dataset_result = build_dataset(config)
    test_loader = dataset_result["test_loader"]
    num_classes = dataset_result["split_data"]["num_classes"]

    # Build and load model
    model = build_model(model_type, config, num_classes)
    checkpoint_path = os.path.join(
        config["training"]["checkpoint"]["save_dir"], f"{model_type}_best.pt"
    )

    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"], strict=False)
        logger.info(f"Loaded checkpoint: {checkpoint_path}")
    else:
        logger.warning(f"No checkpoint found at {checkpoint_path}. Using random weights.")

    model = model.to(device)

    # Load graph data
    graph_data = None
    if model_type in ("network_only", "multimodal"):
        pyg_path = os.path.join(processed_dir, "pyg_graph_data.pt")
        if os.path.exists(pyg_path):
            graph_data = torch.load(pyg_path, map_location=device, weights_only=False)
            for key in ["edge_index", "edge_weight", "x", "y"]:
                if key in graph_data and graph_data[key] is not None:
                    graph_data[key] = graph_data[key].to(device)

    # Generate predictions
    y_score, y_true = predict(model, test_loader, device, model_type, graph_data)

    # Apply GO True Path Rule (Hierarchy Consistency Enforcement)
    go_terms_path = os.path.join(processed_dir, "go_terms.pkl")
    selected_path = os.path.join(processed_dir, "selected_go_terms.pkl")
    if os.path.exists(go_terms_path) and os.path.exists(selected_path):
        logger.info("Applying GO True Path Rule (Hierarchy Consistency)...")
        go_terms_dict = load_pickle(go_terms_path)
        selected_terms = load_pickle(selected_path)
        y_score = apply_true_path_rule(y_score, selected_terms, go_terms_dict)

    # Compute overall metrics
    eval_cfg = config["evaluation"]
    overall_metrics = compute_all_metrics(
        y_true, y_score, top_k_values=eval_cfg.get("top_k", [5, 10, 20])
    )

    logger.info("Overall Test Metrics:")
    for metric, value in sorted(overall_metrics.items()):
        logger.info(f"  {metric}: {value:.4f}")

    # Per-class metrics
    per_class = compute_per_class_metrics(y_true, y_score, overall_metrics.get("fmax_threshold", 0.5))

    # Stratified evaluation
    go_metadata = pd.read_csv(os.path.join(processed_dir, "go_term_metadata.tsv"), sep="\t")
    proteins_df = pd.read_csv(os.path.join(processed_dir, "proteins_annotated.tsv"), sep="\t")
    test_accessions = dataset_result["split_data"]["test_accessions"]

    hub_status = None
    hub_path = os.path.join(processed_dir, "hub_status.pkl")
    if os.path.exists(hub_path):
        hub_status = load_pickle(hub_path)

    centrality = None
    cent_path = os.path.join(processed_dir, "centrality_metrics.pkl")
    if os.path.exists(cent_path):
        centrality = load_pickle(cent_path)

    stratified = stratified_evaluation(
        y_true, y_score, proteins_df, test_accessions, go_metadata, hub_status, centrality
    )

    # Save results
    output_dir = ensure_dir(f"ProteoFunc_outputs/evaluation/{model_type}")
    save_json(overall_metrics, os.path.join(str(output_dir), "overall_metrics.json"))
    save_json(stratified, os.path.join(str(output_dir), "stratified_metrics.json"))
    np.save(os.path.join(str(output_dir), "predictions.npy"), y_score)
    np.save(os.path.join(str(output_dir), "true_labels.npy"), y_true)

    logger.info(f"Evaluation results saved to {output_dir}")

    # Explicit memory cleanup
    del model, test_loader
    if graph_data is not None:
        del graph_data
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

    return {
        "overall": overall_metrics,
        "stratified": stratified,
        "per_class": per_class,
        "predictions": y_score,
        "true_labels": y_true,
    }


def compare_models(config: Dict[str, Any]) -> pd.DataFrame:
    """
    Compare evaluation results across all model variants.
    Includes statistical significance tests (t-test, Wilcoxon, bootstrap CIs).

    Returns:
        DataFrame comparison table.
    """
    results = {}
    for model_type in ["sequence_only", "network_only", "multimodal"]:
        metrics_path = f"ProteoFunc_outputs/evaluation/{model_type}/overall_metrics.json"
        if os.path.exists(metrics_path):
            from utils.helpers import load_json
            results[model_type] = load_json(metrics_path)

    if not results:
        logger.warning("No evaluation results found.")
        return pd.DataFrame()

    comparison = pd.DataFrame(results).T
    comparison.index.name = "Model"

    logger.info("\n" + "=" * 70)
    logger.info("MODEL COMPARISON")
    logger.info("=" * 70)
    logger.info(f"\n{comparison.to_string()}")

    output_path = "ProteoFunc_outputs/evaluation/model_comparison.csv"
    ensure_dir(os.path.dirname(output_path))
    comparison.to_csv(output_path)
    logger.info(f"Comparison table saved to {output_path}")

    # --- Statistical significance tests ---
    try:
        from evaluation.statistical_tests import compare_models_statistical

        model_predictions = {}
        true_labels = None

        for model_type in ["sequence_only", "network_only", "multimodal"]:
            pred_path = f"ProteoFunc_outputs/evaluation/{model_type}/predictions.npy"
            label_path = f"ProteoFunc_outputs/evaluation/{model_type}/true_labels.npy"
            if os.path.exists(pred_path) and os.path.exists(label_path):
                model_predictions[model_type] = np.load(pred_path)
                if true_labels is None:
                    true_labels = np.load(label_path)

        if len(model_predictions) >= 2 and true_labels is not None:
            eval_cfg = config.get("evaluation", {})
            stat_cfg = eval_cfg.get("statistical", {})
            alpha = stat_cfg.get("alpha", 0.05)
            n_bootstrap = stat_cfg.get("num_bootstrap", 1000)

            logger.info("\nRunning statistical significance tests...")
            stat_results = compare_models_statistical(
                model_predictions, true_labels,
                alpha=alpha, n_bootstrap=n_bootstrap,
            )

            # Log key results
            for comp in stat_results.get("pairwise_comparisons", []):
                a, b = comp["model_a"], comp["model_b"]
                p_val = comp["ttest"]["p_value"]
                d = comp["ttest"]["cohens_d"]
                sig = "YES" if comp["ttest"]["significant"] else "no"
                logger.info(f"  {a} vs {b}: p={p_val:.4f}, Cohen's d={d:.4f}, significant={sig}")

            # Save statistical results
            stat_output = "ProteoFunc_outputs/evaluation/statistical_tests.json"
            save_json(stat_results, stat_output)
            logger.info(f"Statistical tests saved to {stat_output}")
        else:
            logger.warning("Not enough model predictions for statistical tests.")

    except Exception as e:
        logger.warning(f"Statistical tests failed: {e}")

    # Generate evaluation plots automatically
    try:
        from evaluation.visualizations import generate_all_plots
        generate_all_plots(output_dir="ProteoFunc_outputs/figures")
        logger.info("Evaluation plots generated in outputs/figures/")
    except Exception as e:
        logger.warning(f"Plot generation failed: {e}")

    return comparison


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained models")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument(
        "--model", type=str, default="all",
        choices=["sequence_only", "network_only", "multimodal", "all"],
    )
    parser.add_argument("--compare", action="store_true", help="Compare all models")
    args = parser.parse_args()

    config = load_config(args.config)

    if args.model == "all":
        for model_type in ["sequence_only", "network_only", "multimodal"]:
            evaluate_model(model_type, config)
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
        compare_models(config)
    elif args.compare:
        # If specific model or all evaluated, compare
        compare_models(config)
    else:
        evaluate_model(args.model, config)


if __name__ == "__main__":
    main()
