"""Paper figures (print: colour-blind-safe hues plus a distinct marker and dash per series)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

from .metrics import operational_metrics

# Okabe-Ito hues in a fixed order; colour follows the series, never its rank.
SERIES_STYLE = {
    "S4_E06 (Decoupled)": ("#0072B2", "o", "-"),
    "S4_E06 (MSP Probe)": ("#56B4E9", "s", "--"),
    "S4_E05 (SSL Base)": ("#D55E00", "^", "-."),
    "Random Forest": ("#009E73", "D", ":"),
    "XGBoost": ("#E69F00", "v", "--"),
    "Deep SVDD": ("#000000", "X", "-"),
    "SCARF (SSL)": ("#CC79A7", "P", "-."),
}
REGIME_STYLE = {"benign": ("#0072B2", "o"), "mixture": ("#D55E00", "s"), "oracle": ("#000000", "^")}
REGIME_LABEL = {"benign": "Benign calibration", "mixture": "Mixture calibration",
                "oracle": "Oracle (post-hoc)"}


def _style_axes(ax: plt.Axes) -> None:
    ax.grid(True, color="#DDDDDD", linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def fig2_budget_trajectory(trajectory: pd.DataFrame, out_path: Path) -> Path:
    """Zero-day AUROC and AUPRC against the label budget, mean with std error bars."""
    budgets = sorted(trajectory["budget"].unique(), reverse=True)
    position = {b: i for i, b in enumerate(budgets)}
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.4), sharey=True)
    for ax, metric, title in zip(axes, ("auroc", "auprc"), ("Zero-day AUROC", "Zero-day AUPRC")):
        for label, (color, marker, dash) in SERIES_STYLE.items():
            series = trajectory.loc[trajectory["row"] == label].sort_values("budget", ascending=False)
            if not len(series):
                continue
            x = [position[b] for b in series["budget"]]
            ax.errorbar(x, series[f"{metric}_mean"], yerr=series[f"{metric}_std"].fillna(0),
                        color=color, marker=marker, linestyle=dash, linewidth=2, markersize=6,
                        capsize=2.5, elinewidth=1, label=label)
        ax.set_xticks(range(len(budgets)), [f"{100 * b:g}%" for b in budgets])
        ax.set_xlabel("Label budget")
        ax.set_title(title, fontsize=10)
        ax.set_ylim(0, 1.02)
        _style_axes(ax)
    axes[0].set_ylabel("Mean over seeds (bars: std)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False, fontsize=8)
    fig.tight_layout(rect=(0, 0.14, 1, 1))
    return _save(fig, out_path)


def fig3_operating(scores: np.ndarray, is_zero_day: np.ndarray, thresholds: dict[str, float],
                   out_path: Path, title_suffix: str = "") -> Path:
    """(a) ROC with the calibration regimes marked; (b) recall/precision/FPR against the threshold."""
    fpr, tpr, _ = roc_curve(is_zero_day, scores)
    auc = roc_auc_score(is_zero_day, scores)
    fig, (ax_roc, ax_sweep) = plt.subplots(1, 2, figsize=(9.0, 3.6))
    ax_roc.plot(fpr, tpr, color="#0072B2", linewidth=2, label=f"ROC (AUC = {auc:.4f})")
    ax_roc.plot([0, 1], [0, 1], color="#888888", linewidth=1, linestyle="--", label="Chance")
    for regime, tau in thresholds.items():
        point = operational_metrics(scores, is_zero_day, tau)
        color, marker = REGIME_STYLE[regime]
        ax_roc.scatter(point["fpr"], point["recall"], s=64, color=color, marker=marker, zorder=3,
                       edgecolor="white", linewidth=1.5, label=f"{REGIME_LABEL[regime]} ($\\tau$={tau:.4g})")
    ax_roc.set_xlabel("False positive rate")
    ax_roc.set_ylabel("True positive rate")
    ax_roc.set_title(f"(a) ROC and operating points{title_suffix}", fontsize=10)
    ax_roc.legend(frameon=False, fontsize=8, loc="lower right")
    _style_axes(ax_roc)

    low, high = np.quantile(scores, [0.005, 0.995])
    grid = np.linspace(low, high, 200)
    sweep = [operational_metrics(scores, is_zero_day, tau) for tau in grid]
    for name, color, dash in (("recall", "#0072B2", "-"), ("precision", "#009E73", "--"), ("fpr", "#D55E00", ":")):
        label = {"recall": "Recall (zero-day)", "precision": "Precision", "fpr": "FPR"}[name]
        ax_sweep.plot(grid, [p[name] for p in sweep], color=color, linestyle=dash, linewidth=2, label=label)
    for regime, tau in thresholds.items():
        if low <= tau <= high:
            ax_sweep.axvline(tau, color=REGIME_STYLE[regime][0], linewidth=1, alpha=0.7)
            ax_sweep.text(tau, 1.03, REGIME_LABEL[regime].split()[0], rotation=90, fontsize=7,
                          ha="center", va="bottom", color="#333333")
    ax_sweep.set_xlabel("Alarm threshold $\\tau$")
    ax_sweep.set_ylabel("Metric value")
    ax_sweep.set_ylim(0, 1.02)
    ax_sweep.set_title("(b) Threshold sweep", fontsize=10, pad=28)
    ax_sweep.legend(frameon=False, fontsize=8, loc="center right")
    _style_axes(ax_sweep)
    fig.tight_layout()
    return _save(fig, out_path)


def _save(fig: plt.Figure, out_path: Path) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    return out_path
