"""Paired statistics: exact Wilcoxon signed-rank, Benjamini-Hochberg, Cliff's delta."""

from __future__ import annotations

import numpy as np
from scipy.stats import rankdata

MAX_EXACT_N = 20


def wilcoxon_exact(x, y) -> dict:
    """Exact two-sided Wilcoxon signed-rank test for paired samples.

    Zero differences are dropped; ties get mid-ranks and the null distribution is
    enumerated over all sign assignments, so the p-value is exact even with ties.
    """
    diff = np.asarray(x, dtype="float64") - np.asarray(y, dtype="float64")
    if diff.ndim != 1 or len(diff) == 0:
        raise ValueError("wilcoxon_exact needs two equal-length 1-D samples")
    diff = diff[diff != 0]
    n = len(diff)
    if n == 0:
        return {"n": 0, "w_plus": 0.0, "w_minus": 0.0, "statistic": 0.0, "p_value": 1.0}
    if n > MAX_EXACT_N:
        raise ValueError(f"exact enumeration supports n <= {MAX_EXACT_N}, got {n}")
    ranks = rankdata(np.abs(diff))
    w_plus = float(ranks[diff > 0].sum())
    w_minus = float(ranks[diff < 0].sum())
    statistic = min(w_plus, w_minus)
    # Distribution of W+ under the null: every rank is positive with probability 1/2.
    signs = (np.arange(2 ** n)[:, None] >> np.arange(n)) & 1
    null_w = signs @ ranks
    total = ranks.sum()
    tol = 1e-9
    p_value = float(np.mean((null_w <= statistic + tol) | (null_w >= total - statistic - tol)))
    return {"n": n, "w_plus": w_plus, "w_minus": w_minus, "statistic": statistic,
            "p_value": min(1.0, p_value)}


def benjamini_hochberg(p_values) -> np.ndarray:
    """BH-adjusted p-values, in the input order."""
    p = np.asarray(p_values, dtype="float64")
    if p.ndim != 1 or len(p) == 0:
        raise ValueError("benjamini_hochberg needs a non-empty 1-D array")
    if ((p < 0) | (p > 1)).any():
        raise ValueError("p-values must lie in [0, 1]")
    m = len(p)
    order = np.argsort(p)
    scaled = p[order] * m / np.arange(1, m + 1)
    monotone = np.minimum.accumulate(scaled[::-1])[::-1]
    adjusted = np.empty(m)
    adjusted[order] = np.clip(monotone, 0, 1)
    return adjusted


def cliffs_delta(x, y) -> float:
    """P(x > y) - P(x < y) over all cross pairs."""
    x, y = np.asarray(x, dtype="float64"), np.asarray(y, dtype="float64")
    if len(x) == 0 or len(y) == 0:
        raise ValueError("cliffs_delta needs non-empty samples")
    greater = (x[:, None] > y[None, :]).sum()
    less = (x[:, None] < y[None, :]).sum()
    return float((greater - less) / (len(x) * len(y)))


def delta_magnitude(delta: float) -> str:
    """Romano et al. thresholds."""
    size = abs(delta)
    if size < 0.147:
        return "Negligible"
    if size < 0.33:
        return "Small"
    if size < 0.474:
        return "Medium"
    return "Large"
