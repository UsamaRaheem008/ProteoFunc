"""
Statistical Testing Module
============================
Implements paired statistical tests for model comparison.
"""

import numpy as np
from typing import Any, Dict, List, Optional, Tuple
from scipy import stats


def paired_ttest(
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    alpha: float = 0.05,
) -> Dict[str, float]:
    """
    Paired two-tailed t-test for comparing model performance.

    Args:
        scores_a: Per-sample metric for model A (N,).
        scores_b: Per-sample metric for model B (N,).
        alpha: Significance level.

    Returns:
        Dictionary with t-statistic, p-value, and significance.
    """
    t_stat, p_value = stats.ttest_rel(scores_a, scores_b)
    effect_size = cohens_d(scores_a, scores_b)

    return {
        "test": "paired_ttest",
        "t_statistic": float(t_stat),
        "p_value": float(p_value),
        "significant": bool(p_value < alpha),
        "alpha": alpha,
        "cohens_d": float(effect_size),
        "mean_diff": float(np.mean(scores_a - scores_b)),
        "std_diff": float(np.std(scores_a - scores_b)),
    }


def wilcoxon_test(
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    alpha: float = 0.05,
) -> Dict[str, float]:
    """
    Wilcoxon signed-rank test (non-parametric alternative to paired t-test).

    Args:
        scores_a: Per-sample metric for model A (N,).
        scores_b: Per-sample metric for model B (N,).
        alpha: Significance level.

    Returns:
        Dictionary with test results.
    """
    # Remove zero differences (tied samples)
    diff = scores_a - scores_b
    non_zero = diff != 0

    if non_zero.sum() < 10:
        return {
            "test": "wilcoxon",
            "statistic": float("nan"),
            "p_value": 1.0,
            "significant": False,
            "alpha": alpha,
            "note": "Too few non-tied samples",
        }

    stat, p_value = stats.wilcoxon(scores_a[non_zero], scores_b[non_zero])

    return {
        "test": "wilcoxon",
        "statistic": float(stat),
        "p_value": float(p_value),
        "significant": bool(p_value < alpha),
        "alpha": alpha,
        "n_samples": int(non_zero.sum()),
    }


def bonferroni_correction(
    p_values: List[float], alpha: float = 0.05
) -> Dict[str, any]:
    """
    Apply Bonferroni correction for multiple hypothesis testing.

    Args:
        p_values: List of p-values.
        alpha: Original significance level.

    Returns:
        Dictionary with corrected significance results.
    """
    n_tests = len(p_values)
    corrected_alpha = alpha / n_tests
    significant = [p < corrected_alpha for p in p_values]

    return {
        "original_alpha": alpha,
        "corrected_alpha": corrected_alpha,
        "n_tests": n_tests,
        "p_values": p_values,
        "significant": significant,
        "n_significant": sum(significant),
    }


def cohens_d(group_a: np.ndarray, group_b: np.ndarray) -> float:
    """
    Compute Cohen's d effect size for paired samples.

    Args:
        group_a: Scores for group A.
        group_b: Scores for group B.

    Returns:
        Cohen's d value.
    """
    diff = group_a - group_b
    return float(np.mean(diff) / max(np.std(diff, ddof=1), 1e-10))


def bootstrap_confidence_interval(
    scores: np.ndarray,
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """
    Compute bootstrap confidence interval for a metric.

    Args:
        scores: Per-sample metric values.
        n_bootstrap: Number of bootstrap iterations.
        confidence: Confidence level.
        seed: Random seed.

    Returns:
        Tuple of (mean, lower bound, upper bound).
    """
    rng = np.random.RandomState(seed)
    bootstrap_means = []

    for _ in range(n_bootstrap):
        sample = rng.choice(scores, size=len(scores), replace=True)
        bootstrap_means.append(np.mean(sample))

    bootstrap_means = np.array(bootstrap_means)
    lower_percentile = (1 - confidence) / 2 * 100
    upper_percentile = (1 + confidence) / 2 * 100

    return (
        float(np.mean(scores)),
        float(np.percentile(bootstrap_means, lower_percentile)),
        float(np.percentile(bootstrap_means, upper_percentile)),
    )


def compare_models_statistical(
    model_predictions: Dict[str, np.ndarray],
    true_labels: np.ndarray,
    alpha: float = 0.05,
    n_bootstrap: int = 1000,
) -> Dict[str, Any]:
    """
    Comprehensive statistical comparison between model variants.

    Args:
        model_predictions: Dict mapping model name -> prediction matrix.
        true_labels: True label matrix.
        alpha: Significance level.
        n_bootstrap: Number of bootstrap samples.

    Returns:
        Complete statistical comparison results.
    """
    from evaluation.metrics import compute_fmax

    results = {}
    model_names = list(model_predictions.keys())

    # Compute per-protein F1 at best threshold for each model
    per_protein_f1 = {}
    for name, preds in model_predictions.items():
        fmax, threshold = compute_fmax(true_labels, preds)
        y_pred = (preds >= threshold).astype(float)

        tp = (y_pred * true_labels).sum(axis=1)
        fp = (y_pred * (1 - true_labels)).sum(axis=1)
        fn = ((1 - y_pred) * true_labels).sum(axis=1)

        precision = np.where(tp + fp > 0, tp / (tp + fp), 0.0)
        recall = np.where(tp + fn > 0, tp / (tp + fn), 0.0)
        f1 = np.where(
            precision + recall > 0,
            2 * precision * recall / (precision + recall),
            0.0,
        )
        per_protein_f1[name] = f1

        # Bootstrap CI
        mean_val, ci_low, ci_high = bootstrap_confidence_interval(
            f1, n_bootstrap=n_bootstrap
        )
        results[f"{name}_f1_mean"] = mean_val
        results[f"{name}_f1_ci"] = (ci_low, ci_high)

    # Pairwise comparisons
    comparisons = []
    p_values = []

    for i in range(len(model_names)):
        for j in range(i + 1, len(model_names)):
            name_a, name_b = model_names[i], model_names[j]
            f1_a, f1_b = per_protein_f1[name_a], per_protein_f1[name_b]

            # Paired t-test
            ttest = paired_ttest(f1_a, f1_b, alpha=alpha)
            # Wilcoxon
            wilcox = wilcoxon_test(f1_a, f1_b, alpha=alpha)

            comparison = {
                "model_a": name_a,
                "model_b": name_b,
                "ttest": ttest,
                "wilcoxon": wilcox,
            }
            comparisons.append(comparison)
            p_values.append(ttest["p_value"])

    # Bonferroni correction
    if p_values:
        correction = bonferroni_correction(p_values, alpha=alpha)
        results["bonferroni"] = correction

    results["pairwise_comparisons"] = comparisons

    return results
