# SPDX-License-Identifier: LGPL-3.0-or-later
"""Stable classification metrics aggregated by structures, not by systems."""

from typing import (
    Any,
)

import numpy as np


def classification_probabilities(logits: np.ndarray) -> np.ndarray:
    """Convert finite raw logits to probabilities without changing argmax."""
    logits = np.asarray(logits, dtype=np.float64)
    if logits.ndim != 2 or logits.shape[1] < 2 or not np.isfinite(logits).all():
        raise ValueError("Expected finite logits of shape (n_structures, n_classes>=2)")
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def classification_statistics(logits: np.ndarray, labels: np.ndarray) -> dict[str, Any]:
    """Collect additive counts and sums for exact dataset-wide metrics."""
    logits = np.asarray(logits, dtype=np.float64)
    probabilities = classification_probabilities(logits)
    n, num_classes = logits.shape
    labels = np.asarray(labels)
    if labels.ndim == 2 and labels.shape[1] == 1:
        labels = labels[:, 0]
    if labels.shape != (n,) or not n:
        raise ValueError(
            "Expected one class ID per structure and at least one structure"
        )
    if not np.isfinite(labels).all() or not np.equal(labels, np.floor(labels)).all():
        raise ValueError("Class labels must be finite integers")
    if (labels < 0).any() or (labels >= num_classes).any():
        raise ValueError(f"Class labels must be in [0, {num_classes})")
    labels = labels.astype(np.int64)
    predicted = logits.argmax(axis=1)
    confusion = np.bincount(
        labels * num_classes + predicted, minlength=num_classes**2
    ).reshape(num_classes, num_classes)
    shifted = logits - logits.max(axis=1, keepdims=True)
    log_probabilities = shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))
    error = probabilities - np.eye(num_classes)[labels]
    return {
        "n": n,
        "num_classes": num_classes,
        "confusion_matrix": confusion.tolist(),
        "ce_sum": float(-log_probabilities[np.arange(n), labels].sum()),
        "absolute_error_sum": float(np.abs(error).sum()),
        "squared_error_sum": float(np.square(error).sum()),
    }


def merge_classification_statistics(statistics: list[dict[str, Any]]) -> dict[str, Any]:
    """Combine systems/checkpoint chunks before computing nonlinear metrics."""
    if not statistics:
        raise ValueError("No classification statistics to merge")
    num_classes = statistics[0]["num_classes"]
    if any(stat["num_classes"] != num_classes for stat in statistics):
        raise ValueError("Cannot merge different numbers of classes")
    result = {"num_classes": num_classes}
    for key in ("n", "ce_sum", "absolute_error_sum", "squared_error_sum"):
        result[key] = sum(stat[key] for stat in statistics)
    result["confusion_matrix"] = np.sum(
        [stat["confusion_matrix"] for stat in statistics], axis=0
    ).tolist()
    return result


def classification_metrics(statistics: dict[str, Any]) -> dict[str, Any]:
    """Compute metrics from a global confusion matrix and probability sums.

    Confusion rows are true classes, columns are predictions. Macro-F1 includes
    all configured classes with zero division mapped to zero; balanced accuracy
    averages recall over classes present in the evaluated labels. In binary
    tasks class 0 is negative and class 1 is positive.
    """
    n, num_classes = statistics["n"], statistics["num_classes"]
    matrix = np.asarray(statistics["confusion_matrix"], dtype=np.int64)
    if n <= 0 or matrix.shape != (num_classes, num_classes) or matrix.sum() != n:
        raise ValueError("Inconsistent classification counts")
    support = matrix.sum(axis=1)
    predicted_count = matrix.sum(axis=0)
    true_positive = matrix.diagonal()
    precision = np.divide(
        true_positive,
        predicted_count,
        out=np.zeros(num_classes),
        where=predicted_count > 0,
    )
    recall = np.divide(
        true_positive, support, out=np.zeros(num_classes), where=support > 0
    )
    denominator = support + predicted_count
    f1 = np.divide(
        2 * true_positive, denominator, out=np.zeros(num_classes), where=denominator > 0
    )
    mse = statistics["squared_error_sum"] / (n * num_classes)
    result = {
        "n": int(n),
        "num_classes": int(num_classes),
        "ce": statistics["ce_sum"] / n,
        "accuracy": float(true_positive.sum() / n),
        "macro_f1": float(f1.mean()),
        "balanced_accuracy": float(recall[support > 0].mean()),
        "mae": statistics["absolute_error_sum"] / (n * num_classes),
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "confusion_matrix": matrix.tolist(),
        "precision_per_class": precision.tolist(),
        "recall_per_class": recall.tolist(),
        "f1_per_class": f1.tolist(),
        "support_per_class": support.tolist(),
    }
    if num_classes == 2:
        tn, fp, fn, tp = (int(value) for value in matrix.ravel())
        result.update(
            tn=tn,
            fp=fp,
            fn=fn,
            tp=tp,
            false_positive_rate=fp / (tn + fp) if tn + fp else None,
        )
    return result
