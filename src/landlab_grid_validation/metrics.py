"""Metrics for comparing mapped fields."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class BinaryCounts:
    """Confusion-matrix counts for a binary comparison."""

    true_positive: int
    false_positive: int
    true_negative: int
    false_negative: int

    def as_dict(self) -> dict[str, int]:
        return {
            "true_positive": self.true_positive,
            "false_positive": self.false_positive,
            "true_negative": self.true_negative,
            "false_negative": self.false_negative,
        }


def finite_pair_mask(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Mask cells where both arrays carry finite values."""
    return np.isfinite(a) & np.isfinite(b)


def continuous_metrics(predicted: np.ndarray, observed: np.ndarray) -> dict[str, float]:
    """Metrics for continuous predicted and observed values."""
    residual = predicted - observed
    metrics = {
        "n": int(residual.size),
        "bias": float(np.mean(residual)),
        "mae": float(np.mean(np.abs(residual))),
        "rmse": float(np.sqrt(np.mean(residual**2))),
        "observed_mean": float(np.mean(observed)),
        "predicted_mean": float(np.mean(predicted)),
    }

    if predicted.size > 1 and np.std(predicted) > 0 and np.std(observed) > 0:
        metrics["pearson_r"] = float(np.corrcoef(predicted, observed)[0, 1])
    else:
        metrics["pearson_r"] = np.nan
    return metrics


def binary_counts(predicted: np.ndarray, observed: np.ndarray) -> BinaryCounts:
    """Confusion-matrix counts for boolean or 0/1 arrays."""
    p = predicted.astype(bool)
    o = observed.astype(bool)
    return BinaryCounts(
        true_positive=int(np.sum(p & o)),
        false_positive=int(np.sum(p & ~o)),
        true_negative=int(np.sum(~p & ~o)),
        false_negative=int(np.sum(~p & o)),
    )


def binary_metrics(predicted: np.ndarray, observed: np.ndarray) -> dict[str, float]:
    """Metrics for binary predicted and observed values."""
    c = binary_counts(predicted, observed)
    tp, fp, tn, fn = (
        c.true_positive,
        c.false_positive,
        c.true_negative,
        c.false_negative,
    )

    def div(num: float, den: float) -> float:
        return float(num / den) if den else np.nan

    out: dict[str, float] = {
        **c.as_dict(),
        "n": int(tp + fp + tn + fn),
        "prevalence": div(tp + fn, tp + fp + tn + fn),
        "accuracy": div(tp + tn, tp + fp + tn + fn),
        "precision": div(tp, tp + fp),
        "recall": div(tp, tp + fn),
        "specificity": div(tn, tn + fp),
        "false_alarm_rate": div(fp, fp + tn),
        "iou_csi": div(tp, tp + fp + fn),
    }
    p = out["precision"]
    r = out["recall"]
    out["f1"] = float(2 * p * r / (p + r)) if np.isfinite(p + r) and (p + r) else np.nan
    return out


def probability_metrics(
    predicted_probability: np.ndarray,
    observed_binary: np.ndarray,
    threshold: float,
) -> dict[str, float]:
    """Metrics for continuous probability against binary observations."""
    prob = np.clip(predicted_probability.astype(float), 0.0, 1.0)
    obs = observed_binary.astype(bool)
    binary_pred = prob >= threshold
    out = binary_metrics(binary_pred, obs)
    out.update(
        {
            "threshold": float(threshold),
            "brier_score": float(np.mean((prob - obs.astype(float)) ** 2)),
            "roc_auc": roc_auc(prob, obs),
            "average_precision": average_precision(prob, obs),
        }
    )
    return out


def roc_auc(scores: np.ndarray, observed_binary: np.ndarray) -> float:
    """ROC AUC using the Mann-Whitney rank formulation with tie handling."""
    y = observed_binary.astype(bool)
    n_pos = int(np.sum(y))
    n_neg = int(np.sum(~y))
    if n_pos == 0 or n_neg == 0:
        return np.nan

    order = np.argsort(scores)
    sorted_scores = scores[order]
    ranks = np.empty(scores.size, dtype=float)

    i = 0
    while i < scores.size:
        j = i + 1
        while j < scores.size and sorted_scores[j] == sorted_scores[i]:
            j += 1
        # Ranks are 1-based; ties get their average rank.
        ranks[order[i:j]] = (i + 1 + j) / 2.0
        i = j

    rank_sum_pos = float(np.sum(ranks[y]))
    return float((rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def average_precision(scores: np.ndarray, observed_binary: np.ndarray) -> float:
    """Average precision for binary observations and continuous scores."""
    y = observed_binary.astype(bool)
    n_pos = int(np.sum(y))
    if n_pos == 0:
        return np.nan

    order = np.argsort(-scores)
    y_sorted = y[order]
    tp = np.cumsum(y_sorted)
    fp = np.cumsum(~y_sorted)
    precision = tp / (tp + fp)
    return float(np.sum(precision[y_sorted]) / n_pos)


def categorical_metrics(
    predicted: np.ndarray,
    observed: np.ndarray,
    labels: list[Any] | None = None,
) -> dict[str, Any]:
    """Confusion matrix and per-class scores for categorical fields."""
    if labels is None:
        labels = sorted(set(predicted.tolist()) | set(observed.tolist()))

    label_keys = [_label_key(label) for label in labels]
    matrix = {
        obs_key: {pred_key: 0 for pred_key in label_keys}
        for obs_key in label_keys
    }
    for pred, obs in zip(predicted, observed, strict=True):
        matrix[_label_key(obs)][_label_key(pred)] += 1

    per_class: dict[str, dict[str, float]] = {}
    correct = 0
    for label, key in zip(labels, label_keys, strict=True):
        tp = matrix[key][key]
        fp = sum(
            matrix[other_key][key]
            for other, other_key in zip(labels, label_keys, strict=True)
            if other != label
        )
        fn = sum(
            matrix[key][other_key]
            for other, other_key in zip(labels, label_keys, strict=True)
            if other != label
        )
        support = sum(matrix[key].values())
        correct += tp

        precision = tp / (tp + fp) if (tp + fp) else np.nan
        recall = tp / (tp + fn) if (tp + fn) else np.nan
        f1 = (
            2 * precision * recall / (precision + recall)
            if np.isfinite(precision + recall) and (precision + recall)
            else np.nan
        )
        per_class[key] = {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "support": int(support),
        }

    n = int(predicted.size)
    return {
        "n": n,
        "overall_accuracy": float(correct / n) if n else np.nan,
        "labels": label_keys,
        "confusion_matrix": matrix,
        "per_class": per_class,
    }


def continuous_by_category(
    continuous: np.ndarray,
    categories: np.ndarray,
) -> dict[str, Any]:
    """Summarize continuous values grouped by observed categories."""
    out: dict[str, Any] = {"n": int(continuous.size), "by_class": {}}
    for cls in sorted(set(categories.tolist())):
        vals = continuous[categories == cls]
        out["by_class"][str(cls)] = {
            "n": int(vals.size),
            "mean": float(np.mean(vals)),
            "median": float(np.median(vals)),
            "min": float(np.min(vals)),
            "max": float(np.max(vals)),
        }
    return out


def _label_key(value: Any) -> str:
    """Stable string key for numeric categorical labels.

    Landlab fields are commonly float arrays, so class code 1 may arrive as
    1.0. For confusion matrices, 1 and 1.0 should be the same class.
    """
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)
