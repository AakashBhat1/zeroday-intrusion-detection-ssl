"""Ranking, operational and closed-world metrics."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
from sklearn.metrics import (accuracy_score, average_precision_score,
                             precision_recall_fscore_support, roc_auc_score)

from .calibrate import oracle_threshold, percentile_threshold


def ranking_metrics(scores: np.ndarray, is_zero_day: np.ndarray) -> dict:
    return {
        "auroc": float(roc_auc_score(is_zero_day, scores)),
        "auprc": float(average_precision_score(is_zero_day, scores)),
    }


def _ratio(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def operational_metrics(scores: np.ndarray, is_zero_day: np.ndarray, threshold: float,
                        realistic_prevalence: float | None = None) -> dict:
    """Confusion-matrix metrics for the rule ``score > threshold`` means zero-day."""
    positive = np.asarray(is_zero_day).astype(bool)
    flagged = np.asarray(scores) > threshold
    tp = int((flagged & positive).sum())
    fp = int((flagged & ~positive).sum())
    fn = int((~flagged & positive).sum())
    tn = int((~flagged & ~positive).sum())
    recall, fpr = _ratio(tp, tp + fn), _ratio(fp, fp + tn)
    precision = _ratio(tp, tp + fp)
    result = {
        "threshold": float(threshold), "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": precision, "recall": recall, "tnr": 1 - fpr, "fpr": fpr, "fnr": 1 - recall,
        "f1": _ratio(2 * precision * recall, precision + recall),
        "accuracy": _ratio(tp + tn, tp + fp + fn + tn),
    }
    if realistic_prevalence is not None:
        hits = recall * realistic_prevalence
        result["precision_at_realistic_prevalence"] = _ratio(
            hits, hits + fpr * (1 - realistic_prevalence)
        )
    return result


def closed_world_metrics(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> dict:
    """Accuracy and macro precision/recall/F1 over all catalogued classes."""
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=np.arange(n_classes), average="macro", zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(precision), "macro_recall": float(recall), "macro_f1": float(f1),
    }


def evaluate_zero_day(test_scores: np.ndarray, is_zero_day: np.ndarray, tune_scores: np.ndarray,
                      tune_is_benign: np.ndarray, eval_cfg: Mapping[str, Any]) -> dict:
    """All zero-day numbers for one scorer.

    ``benign`` is the primary regime (threshold from tuning benign flows only),
    ``mixture`` the ablation (threshold from the whole tuning set), and ``oracle``
    a post-hoc diagnostic that reads test labels.
    """
    target, prevalence = eval_cfg["target_fpr"], eval_cfg.get("realistic_prevalence")
    thresholds = {
        "benign": percentile_threshold(tune_scores[np.asarray(tune_is_benign, dtype=bool)], target),
        "mixture": percentile_threshold(tune_scores, target),
        "oracle": oracle_threshold(test_scores, is_zero_day),
    }
    return {
        **ranking_metrics(test_scores, is_zero_day),
        "n_calibration_benign": int(np.asarray(tune_is_benign, dtype=bool).sum()),
        "operating_points": {
            name: operational_metrics(test_scores, is_zero_day, tau, prevalence)
            for name, tau in thresholds.items()
        },
    }
