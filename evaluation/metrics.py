"""
Evaluation Metrics Module
==========================
Implements CAFA-standard and standard ML metrics for multi-label protein
function prediction evaluation.
"""

import numpy as np
from typing import Any, Dict, List, Optional, Tuple
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    hamming_loss,
    precision_recall_curve,
    roc_curve,
)


def compute_fmax(
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold_range: Tuple[float, float] = (0.01, 0.99),
    threshold_step: float = 0.01,
) -> Tuple[float, float]:
    """
    Compute CAFA F-max metric (protein-centric).

    F-max is the maximum F1 across all thresholds, computed per-protein
    and then averaged.

    Args:
        y_true: Binary label matrix (N, C).
        y_score: Prediction probabilities (N, C).
        threshold_range: Min and max threshold values.
        threshold_step: Step size for threshold sweep.

    Returns:
        Tuple of (best F-max score, optimal threshold).
    """
    thresholds = np.arange(threshold_range[0], threshold_range[1], threshold_step)
    best_fmax = 0.0
    best_threshold = 0.5

    for t in thresholds:
        y_pred = (y_score >= t).astype(float)

        # Protein-centric precision and recall
        tp = (y_pred * y_true).sum(axis=1)
        fp = (y_pred * (1 - y_true)).sum(axis=1)
        fn = ((1 - y_pred) * y_true).sum(axis=1)

        # Handle proteins with no predictions
        precision_per_protein = np.where(
            tp + fp > 0, tp / (tp + fp), 0.0
        )
        recall_per_protein = np.where(
            tp + fn > 0, tp / (tp + fn), 0.0
        )

        # Average across proteins that have at least one true label
        has_labels = y_true.sum(axis=1) > 0
        if has_labels.sum() == 0:
            continue

        avg_precision = precision_per_protein[has_labels].mean()
        avg_recall = recall_per_protein[has_labels].mean()

        if avg_precision + avg_recall > 0:
            fmax = 2 * avg_precision * avg_recall / (avg_precision + avg_recall)
        else:
            fmax = 0.0

        if fmax > best_fmax:
            best_fmax = fmax
            best_threshold = t

    return float(best_fmax), float(best_threshold)


def compute_smin(
    y_true: np.ndarray,
    y_score: np.ndarray,
    ic_weights: Optional[np.ndarray] = None,
    threshold_range: Tuple[float, float] = (0.01, 0.99),
    threshold_step: float = 0.01,
) -> Tuple[float, float]:
    """
    Compute CAFA S-min (semantic distance) metric.

    S-min measures the minimum remaining uncertainty, weighted by
    information content (IC) of GO terms.

    Args:
        y_true: Binary labels (N, C).
        y_score: Prediction probabilities (N, C).
        ic_weights: Information content weights per GO term (C,).
        threshold_range: Threshold sweep range.
        threshold_step: Threshold step size.

    Returns:
        Tuple of (S-min value, optimal threshold).
    """
    if ic_weights is None:
        # Use term frequency as proxy for IC
        term_freq = y_true.sum(axis=0) / max(y_true.shape[0], 1)
        ic_weights = -np.log2(np.clip(term_freq, 1e-10, 1.0))

    thresholds = np.arange(threshold_range[0], threshold_range[1], threshold_step)
    best_smin = float("inf")
    best_threshold = 0.5

    for t in thresholds:
        y_pred = (y_score >= t).astype(float)

        # Remaining uncertainty (weighted)
        # ru = sum of IC for false negatives (missed true terms)
        fn = y_true * (1 - y_pred)
        ru_per_protein = (fn * ic_weights).sum(axis=1)

        # Misinformation (weighted)
        # mi = sum of IC for false positives (wrong predictions)
        fp = (1 - y_true) * y_pred
        mi_per_protein = (fp * ic_weights).sum(axis=1)

        # S-min: sqrt(ru^2 + mi^2), averaged
        s_per_protein = np.sqrt(ru_per_protein**2 + mi_per_protein**2)
        s_mean = s_per_protein.mean()

        if s_mean < best_smin:
            best_smin = s_mean
            best_threshold = t

    return float(best_smin), float(best_threshold)


def compute_aupr(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """
    Compute macro-averaged Area Under Precision-Recall curve.

    Args:
        y_true: Binary labels (N, C).
        y_score: Prediction scores (N, C).

    Returns:
        Macro AUPR score.
    """
    valid_cols = y_true.sum(axis=0) > 0
    if valid_cols.sum() == 0:
        return 0.0

    try:
        return float(average_precision_score(
            y_true[:, valid_cols], y_score[:, valid_cols], average="macro"
        ))
    except Exception:
        return 0.0


def compute_auroc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """
    Compute macro-averaged Area Under ROC curve.

    Args:
        y_true: Binary labels (N, C).
        y_score: Prediction scores (N, C).

    Returns:
        Macro AUROC score.
    """
    valid_cols = y_true.sum(axis=0) > 0
    if valid_cols.sum() == 0:
        return 0.0

    try:
        return float(roc_auc_score(
            y_true[:, valid_cols], y_score[:, valid_cols], average="macro"
        ))
    except Exception:
        return 0.0


def compute_topk_accuracy(
    y_true: np.ndarray, y_score: np.ndarray, k: int = 5
) -> float:
    """
    Compute top-k prediction accuracy.

    For each protein, check if at least one of the top-k predicted terms
    is a true annotation.

    Args:
        y_true: Binary labels (N, C).
        y_score: Prediction scores (N, C).
        k: Number of top predictions.

    Returns:
        Top-k accuracy (fraction of proteins with at least one correct top-k prediction).
    """
    n_proteins = y_true.shape[0]
    correct = 0

    for i in range(n_proteins):
        top_k_indices = np.argsort(y_score[i])[-k:]
        if y_true[i, top_k_indices].any():
            correct += 1

    return correct / max(n_proteins, 1)


def compute_all_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: Optional[float] = None,
    top_k_values: List[int] = None,
) -> Dict[str, float]:
    """
    Compute all evaluation metrics.

    Args:
        y_true: Binary labels (N, C).
        y_score: Prediction probabilities (N, C).
        threshold: Decision threshold (if None, optimal is found via F-max).
        top_k_values: List of k values for top-k accuracy.

    Returns:
        Dictionary of all metric values.
    """
    if top_k_values is None:
        top_k_values = [5, 10, 20]

    metrics = {}

    # CAFA metrics
    fmax, fmax_t = compute_fmax(y_true, y_score)
    metrics["fmax"] = fmax
    metrics["fmax_threshold"] = fmax_t

    smin, smin_t = compute_smin(y_true, y_score)
    metrics["smin"] = smin
    metrics["smin_threshold"] = smin_t

    # AUC metrics
    metrics["aupr"] = compute_aupr(y_true, y_score)
    metrics["auroc"] = compute_auroc(y_true, y_score)

    # Threshold-based metrics
    t = threshold if threshold is not None else fmax_t
    y_pred = (y_score >= t).astype(float)

    # Remove columns with no true/predicted labels for safe metric computation
    valid = (y_true.sum(axis=0) > 0) | (y_pred.sum(axis=0) > 0)
    y_true_v = y_true[:, valid]
    y_pred_v = y_pred[:, valid]
    y_score_v = y_score[:, valid]

    try:
        metrics["micro_f1"] = float(f1_score(y_true_v, y_pred_v, average="micro", zero_division=0))
    except Exception:
        metrics["micro_f1"] = 0.0

    try:
        metrics["macro_f1"] = float(f1_score(y_true_v, y_pred_v, average="macro", zero_division=0))
    except Exception:
        metrics["macro_f1"] = 0.0

    try:
        metrics["micro_precision"] = float(precision_score(y_true_v, y_pred_v, average="micro", zero_division=0))
        metrics["micro_recall"] = float(recall_score(y_true_v, y_pred_v, average="micro", zero_division=0))
    except Exception:
        metrics["micro_precision"] = 0.0
        metrics["micro_recall"] = 0.0

    metrics["hamming_loss"] = float(hamming_loss(y_true, y_pred))

    # Top-k accuracy
    for k in top_k_values:
        metrics[f"top_{k}_accuracy"] = compute_topk_accuracy(y_true, y_score, k=k)

    return metrics


def compute_per_class_metrics(
    y_true: np.ndarray, y_score: np.ndarray, threshold: float = 0.5
) -> Dict[str, np.ndarray]:
    """
    Compute per-class (per-GO-term) metrics.

    Returns:
        Dictionary with arrays of per-class AUROC, AUPR, F1, precision, recall.
    """
    n_classes = y_true.shape[1]
    y_pred = (y_score >= threshold).astype(float)

    per_class = {
        "auroc": np.zeros(n_classes),
        "aupr": np.zeros(n_classes),
        "f1": np.zeros(n_classes),
        "precision": np.zeros(n_classes),
        "recall": np.zeros(n_classes),
        "support": y_true.sum(axis=0),
    }

    for i in range(n_classes):
        if y_true[:, i].sum() == 0:
            continue
        try:
            per_class["auroc"][i] = roc_auc_score(y_true[:, i], y_score[:, i])
        except Exception:
            pass
        try:
            per_class["aupr"][i] = average_precision_score(y_true[:, i], y_score[:, i])
        except Exception:
            pass
        try:
            per_class["f1"][i] = f1_score(y_true[:, i], y_pred[:, i], zero_division=0)
            per_class["precision"][i] = precision_score(y_true[:, i], y_pred[:, i], zero_division=0)
            per_class["recall"][i] = recall_score(y_true[:, i], y_pred[:, i], zero_division=0)
        except Exception:
            pass

    return per_class


def apply_true_path_rule(
    y_score: np.ndarray,
    selected_terms: List[str],
    go_terms_dict: Dict[str, Dict[str, Any]],
) -> np.ndarray:
    """
    Enforce GO True Path Rule (Hierarchy Consistency).
    
    If term C is a child of parent P, then P must have at least as high
    a prediction score as C: p(P) = max(p(P), max_{C in children(P)} p(C)).
    
    Args:
        y_score: Prediction probabilities (N, C).
        selected_terms: List of GO term IDs in column order (C,).
        go_terms_dict: Dictionary mapping GO ID -> {parents: [...]}.
        
    Returns:
        Hierarchy-consistent probabilities matrix (N, C).
    """
    term_to_idx = {term: i for i, term in enumerate(selected_terms)}
    y_score_tpr = y_score.copy()
    
    parent_child_pairs = []
    for c_idx, child_id in enumerate(selected_terms):
        if child_id in go_terms_dict:
            parents = go_terms_dict[child_id].get("parents", [])
            for p_id in parents:
                if p_id in term_to_idx:
                    p_idx = term_to_idx[p_id]
                    parent_child_pairs.append((p_idx, c_idx))

    if not parent_child_pairs:
        return y_score_tpr

    for _ in range(5):
        changed = False
        for p_idx, c_idx in parent_child_pairs:
            max_val = np.maximum(y_score_tpr[:, p_idx], y_score_tpr[:, c_idx])
            if not np.array_equal(y_score_tpr[:, p_idx], max_val):
                y_score_tpr[:, p_idx] = max_val
                changed = True
        if not changed:
            break
            
    return y_score_tpr
