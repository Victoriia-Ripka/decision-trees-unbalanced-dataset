# Autorzy: Viktoriia Nowotka, Paweł Łasica
import numpy as np


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Compute binary classification metrics.

    All metrics are computed with respect to the minority (positive) class.
    Minority class is assumed to be the label with fewer samples in y_true.
    """
    classes, counts = np.unique(y_true, return_counts=True)
    pos_label = classes[np.argmin(counts)]

    y_true_bin = (y_true == pos_label).astype(int)
    y_pred_bin = (y_pred == pos_label).astype(int)

    tp = int(np.sum((y_true_bin == 1) & (y_pred_bin == 1)))
    tn = int(np.sum((y_true_bin == 0) & (y_pred_bin == 0)))
    fp = int(np.sum((y_true_bin == 0) & (y_pred_bin == 1)))
    fn = int(np.sum((y_true_bin == 1) & (y_pred_bin == 0)))

    total = tp + tn + fp + fn
    accuracy  = (tp + tn) / total if total > 0 else 0.0
    tpr       = tp / (tp + fn)  if (tp + fn) > 0 else 0.0   # sensitivity / recall
    fpr       = fp / (fp + tn)  if (fp + tn) > 0 else 0.0
    precision = tp / (tp + fp)  if (tp + fp) > 0 else 0.0
    f1        = (2 * precision * tpr) / (precision + tpr) if (precision + tpr) > 0 else 0.0

    return {
        "accuracy":  accuracy,
        "error":     1.0 - accuracy,
        "tpr":       tpr,
        "fpr":       fpr,
        "precision": precision,
        "f1":        f1,
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
    }


def prefix_metrics(metrics: dict, prefix: str) -> dict:
    return {f"{prefix}_{k}": v for k, v in metrics.items()}
