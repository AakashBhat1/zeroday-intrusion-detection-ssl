"""Alarm thresholds. A flow is flagged when its score is strictly above the threshold."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import precision_recall_curve


def percentile_threshold(scores: np.ndarray, target_fpr: float) -> float:
    """Threshold at the (1 - target_fpr) quantile of ``scores`` (validation flows only)."""
    scores = np.asarray(scores, dtype="float64")
    if scores.size == 0:
        raise ValueError("cannot calibrate a threshold on zero flows")
    if not 0 < target_fpr < 1:
        raise ValueError("target_fpr must be in (0, 1)")
    return float(np.quantile(scores, 1 - target_fpr))


def oracle_threshold(scores: np.ndarray, is_positive: np.ndarray) -> float:
    """Post-hoc threshold maximising F1 on the given labelled flows.

    DIAGNOSTIC ONLY: this reads test labels and must never be used as an operating point.
    """
    precision, recall, thresholds = precision_recall_curve(is_positive, scores)
    f1 = 2 * precision[:-1] * recall[:-1] / np.clip(precision[:-1] + recall[:-1], 1e-12, None)
    best = int(np.argmax(f1))
    # precision_recall_curve flags score >= t; step just below t to flag with a strict ">".
    return float(np.nextafter(thresholds[best], -np.inf))
