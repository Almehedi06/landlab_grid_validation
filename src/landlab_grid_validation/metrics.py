"""Metrics for comparing mapped fields."""

from __future__ import annotations

from dataclasses import dataclass
from statistics import NormalDist
from typing import Any

import numpy as np

AUC_CONFIDENCE_LEVEL = 0.95


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


def score_vs_binary_metrics(
    scores: np.ndarray,
    observed_binary: np.ndarray,
    threshold: float,
    *,
    probability: bool,
) -> dict[str, float]:
    """Continuous scores against yes/no observations.

    Scores must increase towards "yes" (for erosion, pass depth or -dz). The
    confusion counts use ``scores >= threshold``; AUC and average precision use
    the ranking and need no threshold. The Brier score is added for probabilities.
    """
    obs = observed_binary.astype(bool)
    out = binary_metrics(scores >= threshold, obs)
    out["threshold"] = float(threshold)
    out["roc_auc"] = roc_auc(scores, obs)
    out["roc_auc_ci_low"], out["roc_auc_ci_high"] = roc_auc_ci(scores, obs)
    out["roc_auc_ci_level"] = AUC_CONFIDENCE_LEVEL
    out["average_precision"] = average_precision(scores, obs)
    if probability:
        out["brier_score"] = float(np.mean((scores - obs.astype(float)) ** 2))
    return out


def probability_metrics(
    predicted_probability: np.ndarray,
    observed_binary: np.ndarray,
    threshold: float,
) -> dict[str, float]:
    """Probability against yes/no observations (kept for backward compatibility)."""
    return score_vs_binary_metrics(
        predicted_probability, observed_binary, threshold, probability=True
    )


def midranks(values: np.ndarray) -> np.ndarray:
    """1-based ranks of ``values``; tied values share their mean rank."""
    _, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
    ends = np.cumsum(counts)
    return (ends - (counts - 1) / 2.0)[inverse.reshape(-1)]


def roc_auc(scores: np.ndarray, observed_binary: np.ndarray) -> float:
    """ROC AUC by the Mann-Whitney rank formula; tied scores share their mean rank."""
    y = observed_binary.astype(bool)
    n_pos = int(np.sum(y))
    n_neg = int(np.sum(~y))
    if n_pos == 0 or n_neg == 0:
        return np.nan
    ranks = midranks(scores)
    return float((np.sum(ranks[y]) - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def roc_auc_variance(scores: np.ndarray, observed_binary: np.ndarray) -> float:
    """Variance of the AUC by the DeLong method, with midranks for tied scores.

    DeLong et al. (1988), computed the fast way of Sun and Xu (2014): the AUC is
    the mean of per-observation placement values, so its variance follows from
    the spread of those values. Exact and deterministic, unlike a bootstrap, and
    it costs one sort rather than a thousand.
    """
    y = observed_binary.astype(bool)
    positives, negatives = scores[y], scores[~y]
    n_pos, n_neg = positives.size, negatives.size
    if n_pos < 2 or n_neg < 2:
        return np.nan  # a variance needs at least two of each

    combined = midranks(scores)
    # Placement of each observation among the other class, ties counted as a half.
    v_pos = (combined[y] - midranks(positives)) / n_neg
    v_neg = 1.0 - (combined[~y] - midranks(negatives)) / n_pos
    return float(np.var(v_pos, ddof=1) / n_pos + np.var(v_neg, ddof=1) / n_neg)


def roc_auc_ci(
    scores: np.ndarray,
    observed_binary: np.ndarray,
    level: float = AUC_CONFIDENCE_LEVEL,
) -> tuple[float, float]:
    """Confidence interval for the AUC, through the logit so it stays inside [0, 1].

    With few observed yes cells the AUC is uncertain, and an interval is the
    honest way to report it: 0.66 from 89 points may not differ from 0.57.
    Returns (nan, nan) when the variance is undefined, such as one class only.

    The interval assumes the compared cells are independent. Mapped points are
    usually clustered, and ``tolerance_cells`` grows each one into a block, so
    neighbouring cells repeat the same evidence. The interval is then narrower
    than the truth: treat it as a lower bound on the uncertainty.
    """
    auc = roc_auc(scores, observed_binary)
    variance = roc_auc_variance(scores, observed_binary)
    # A perfect separation, or scores that are all tied, gives DeLong a variance of
    # zero. Reporting a zero-width interval there would claim a certainty the data
    # do not support, so call it undefined instead.
    if not np.isfinite(auc) or not np.isfinite(variance) or variance <= 0:
        return (np.nan, np.nan)

    z = NormalDist().inv_cdf(0.5 + level / 2.0)
    standard_error = float(np.sqrt(variance))
    if 0.0 < auc < 1.0:
        centre = np.log(auc / (1.0 - auc))
        half_width = z * standard_error / (auc * (1.0 - auc))
        low, high = (1.0 / (1.0 + np.exp(-(centre + sign * half_width))) for sign in (-1, 1))
    else:  # an AUC of exactly 0 or 1 has no logit; fall back to a plain interval
        low, high = auc - z * standard_error, auc + z * standard_error
    return (float(np.clip(low, 0.0, 1.0)), float(np.clip(high, 0.0, 1.0)))


def average_precision(scores: np.ndarray, observed_binary: np.ndarray) -> float:
    """Step-wise average precision over distinct score thresholds.

    Tied scores form one step, so the result does not depend on how ties are
    ordered (the same definition as scikit-learn). This matters for sparse
    model fields, where most cells share the value 0.
    """
    y = observed_binary.astype(bool)
    n_pos = int(np.sum(y))
    if n_pos == 0:
        return np.nan
    order = np.argsort(-scores, kind="mergesort")
    s, ys = scores[order], y[order]
    last = np.r_[np.flatnonzero(np.diff(s)), s.size - 1]   # last index of each tie group
    tp = np.cumsum(ys)[last]
    precision = tp / (last + 1)
    recall = tp / n_pos
    return float(np.sum(np.diff(np.r_[0.0, recall]) * precision))


def categorical_metrics(
    predicted: np.ndarray,
    observed: np.ndarray,
    labels: list[Any] | None = None,
) -> dict[str, Any]:
    """Confusion matrix and per-class scores for categorical fields."""
    if labels is None:
        labels = sorted(set(np.unique(predicted).tolist()) | set(np.unique(observed).tolist()))

    codes = [float(label) for label in labels]
    keys = [_label_key(label) for label in labels]
    counts = np.array(
        [[int(np.sum((observed == co) & (predicted == cp))) for cp in codes] for co in codes]
    )
    matrix = {
        obs_key: {pred_key: int(counts[i, j]) for j, pred_key in enumerate(keys)}
        for i, obs_key in enumerate(keys)
    }

    per_class: dict[str, dict[str, float]] = {}
    for i, key in enumerate(keys):
        tp = counts[i, i]
        fp = counts[:, i].sum() - tp
        fn = counts[i, :].sum() - tp
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
            "support": int(counts[i, :].sum()),
        }

    n = int(predicted.size)
    return {
        "n": n,
        "overall_accuracy": float(np.trace(counts) / n) if n else np.nan,
        "labels": keys,
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
        out["by_class"][_label_key(cls)] = {
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
