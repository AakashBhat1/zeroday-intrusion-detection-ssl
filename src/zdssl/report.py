"""Paper tables from ``runs.jsonl``. Every printed number comes from this module."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from .stats import benjamini_hochberg, cliffs_delta, delta_magnitude, wilcoxon_exact

# (row label, model, scorer) in the order the paper prints them.
ROWS = [
    ("S4_E06 (Decoupled)", "S4_E06", "decoupled"),
    ("S4_E06 (MSP Probe)", "S4_E06", "msp"),
    ("S4_E05 (SSL Base)", "S4_E05", "msp"),
    ("Random Forest", "RF", "msp"),
    ("XGBoost", "XGB", "msp"),
    ("Deep SVDD", "DeepSVDD", "svdd"),
    ("SCARF (SSL)", "SCARF", "msp"),
]
PROPOSED = "S4_E06 (Decoupled)"
MAIN_VARIANT = {"S4_E06": "F", "S4_E05": "C"}
OPERATING_COLUMNS = ["threshold", "precision", "recall", "tnr", "fpr", "fnr", "f1", "accuracy"]
CLOSED_WORLD_COLUMNS = ["accuracy", "macro_precision", "macro_recall", "macro_f1"]
ABLATION_LABELS = {"A": "A. Reconstruction only", "B": "B. Contrastive only",
                   "C": "C. Recon + Contrastive", "F": "F. Recon + Contrastive + Novelty"}


def load_runs(runs_path: Path, only_hash: str | None = None) -> list[dict]:
    """Finished runs (made with config ``only_hash`` if given); a later line replaces an earlier one."""
    if not Path(runs_path).is_file():
        raise FileNotFoundError(f"no results file at {runs_path}")
    by_key: dict[str, dict] = {}
    with Path(runs_path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue  # torn final line from an interrupted write
                if only_hash is None or record["config_hash"] == only_hash:
                    by_key[record["key"]] = record
    return list(by_key.values())


def flatten(runs: list[dict]) -> pd.DataFrame:
    """One row per (run, scorer) with ranking, operating-point and closed-world columns."""
    rows = []
    for run in runs:
        for scorer, zd in run["zero_day"].items():
            row = {k: run[k] for k in ("key", "model", "variant", "condition", "budget", "seed", "n_train")}
            row.update(scorer=scorer, auroc=zd["auroc"], auprc=zd["auprc"])
            for regime, point in zd["operating_points"].items():
                row.update({f"{regime}_{name}": value for name, value in point.items()})
            row.update({f"cw_{k}": v for k, v in (run["closed_world"] or {}).items()})
            row.update({f"time_{k}": v for k, v in run["timing"].items()})
            row.update({f"params_{k}": v for k, v in run["params"].items()})
            row["device"] = run["env"]["device"]
            rows.append(row)
    return pd.DataFrame(rows)


def select(flat: pd.DataFrame, model: str, scorer: str, budget: float, condition: str = "B",
           variant: str | None = None) -> pd.DataFrame:
    variant = variant or MAIN_VARIANT.get(model, "-")
    mask = ((flat["model"] == model) & (flat["scorer"] == scorer) & (flat["variant"] == variant)
            & (flat["condition"] == condition) & np.isclose(flat["budget"], budget))
    return flat.loc[mask].sort_values("seed")


def mean_std(values: pd.Series) -> dict:
    """Mean and sample standard deviation (ddof=1), plus the seed count behind them."""
    values = values.dropna()
    return {"mean": float(values.mean()) if len(values) else float("nan"),
            "std": float(values.std(ddof=1)) if len(values) > 1 else float("nan"),
            "n": int(len(values))}


def _summary(flat: pd.DataFrame, budget: float, columns: list[str]) -> pd.DataFrame:
    rows = []
    for label, model, scorer in ROWS:
        chosen = select(flat, model, scorer, budget)
        row: dict[str, Any] = {"row": label, "n_seeds": len(chosen)}
        for column in columns:
            stats = mean_std(chosen[column]) if column in chosen else {"mean": np.nan, "std": np.nan}
            row[f"{column}_mean"], row[f"{column}_std"] = stats["mean"], stats["std"]
        rows.append(row)
    return pd.DataFrame(rows)


def table3(flat: pd.DataFrame, budget: float) -> pd.DataFrame:
    """Ranking + closed-world, mean and std over seeds."""
    return _summary(flat, budget, ["auroc", "auprc", *[f"cw_{c}" for c in CLOSED_WORLD_COLUMNS]])


def table4_group_a(flat: pd.DataFrame, budget: float, regime: str = "benign") -> pd.DataFrame:
    """Operational metrics at the validation-calibrated threshold, mean and std over seeds."""
    return _summary(flat, budget, [f"{regime}_{c}" for c in OPERATING_COLUMNS])


def table4_group_b(flat: pd.DataFrame, budget: float, seed: int) -> pd.DataFrame:
    """Single-seed case study of the calibration regimes for the proposed model."""
    cases = [("Decoupled, benign calibration", "decoupled", "benign"),
             ("MSP, benign calibration", "msp", "benign"),
             ("Decoupled, mixture calibration (ablation)", "decoupled", "mixture"),
             ("MSP, mixture calibration (ablation)", "msp", "mixture"),
             ("Decoupled, oracle (post-hoc diagnostic)", "decoupled", "oracle")]
    rows = []
    for label, scorer, regime in cases:
        chosen = select(flat, "S4_E06", scorer, budget)
        chosen = chosen.loc[chosen["seed"] == seed]
        if len(chosen) == 1:
            run = chosen.iloc[0]
            rows.append({"row": label, **{c: run[f"{regime}_{c}"] for c in [*OPERATING_COLUMNS, "tp", "fp"]}})
    return pd.DataFrame(rows)


def table5a(flat: pd.DataFrame, budget: float, metric: str = "auroc") -> pd.DataFrame:
    """Proposed vs each other row: exact Wilcoxon on seed-paired values, BH over the family."""
    proposed = select(flat, "S4_E06", "decoupled", budget).set_index("seed")[metric]
    rows = []
    for label, model, scorer in ROWS:
        if label == PROPOSED:
            continue
        other = select(flat, model, scorer, budget).set_index("seed")[metric]
        seeds = proposed.index.intersection(other.index)
        if len(seeds) < 2:
            continue
        a, b = proposed.loc[seeds].to_numpy(), other.loc[seeds].to_numpy()
        test = wilcoxon_exact(a, b)
        delta = cliffs_delta(a, b)
        rows.append({"comparison": f"{PROPOSED} vs {label}", "n_pairs": len(seeds),
                     "mean_difference": float(a.mean() - b.mean()), "w_statistic": test["statistic"],
                     "p_raw": test["p_value"], "cliffs_delta": delta, "magnitude": delta_magnitude(delta)})
    table = pd.DataFrame(rows)
    if len(table):
        table["p_bh"] = benjamini_hochberg(table["p_raw"].to_numpy())
        table["significant_at_0.05"] = table["p_bh"] < 0.05
    return table


def table5b(flat: pd.DataFrame, budget: float) -> pd.DataFrame:
    """Objective ablation with a frozen probe: both scorers, mean and std over seeds."""
    rows = []
    for variant, label in ABLATION_LABELS.items():
        row: dict[str, Any] = {"row": label}
        for scorer in ("decoupled", "msp"):
            chosen = select(flat, "S4_E06", scorer, budget, variant=variant)
            row["n_seeds"] = len(chosen)
            for metric in ("auroc", "auprc"):
                stats = mean_std(chosen[metric])
                row[f"{scorer}_{metric}_mean"], row[f"{scorer}_{metric}_std"] = stats["mean"], stats["std"]
        rows.append(row)
    return pd.DataFrame(rows)


def budget_trajectory(flat: pd.DataFrame, metrics: tuple[str, ...] = ("auroc", "auprc", "cw_macro_f1")
                      ) -> pd.DataFrame:
    """Fig. 2 data: mean and std per row label and budget."""
    rows = []
    for label, model, scorer in ROWS:
        for budget in sorted(flat["budget"].unique(), reverse=True):
            chosen = select(flat, model, scorer, budget)
            if not len(chosen):
                continue
            row = {"row": label, "budget": budget, "n_seeds": len(chosen)}
            for metric in metrics:
                stats = mean_std(chosen[metric]) if metric in chosen else {"mean": np.nan, "std": np.nan}
                row[f"{metric}_mean"], row[f"{metric}_std"] = stats["mean"], stats["std"]
            rows.append(row)
    return pd.DataFrame(rows)


def scorer_table(flat: pd.DataFrame, budget: float) -> pd.DataFrame:
    """Every zero-day score recorded for the S4_E06 encoder (latent and raw-input references)."""
    rows = []
    mask = (flat["model"] == "S4_E06") & (flat["variant"] == "F") & (flat["condition"] == "B")         & np.isclose(flat["budget"], budget)
    for scorer, chosen in flat.loc[mask].groupby("scorer"):
        row: dict[str, Any] = {"row": scorer, "n_seeds": len(chosen)}
        for metric in ("auroc", "auprc", "benign_recall", "benign_fpr", "benign_f1"):
            stats = mean_std(chosen[metric])
            row[f"{metric}_mean"], row[f"{metric}_std"] = stats["mean"], stats["std"]
        rows.append(row)
    return pd.DataFrame(rows).sort_values("auroc_mean", ascending=False) if rows else pd.DataFrame()


def table2(flat: pd.DataFrame, budget: float) -> pd.DataFrame:
    """Cost table: measured medians over seeds on the device recorded in the runs."""
    rows = []
    for label, model, scorer in ROWS:
        if (model, scorer) == ("S4_E06", "msp"):
            continue  # same model as the decoupled row
        chosen = select(flat, model, scorer, budget)
        if not len(chosen):
            continue
        size_columns = [c for c in ("params_trained_total", "params_deployed", "params_nodes", "params_trees")
                        if c in chosen and chosen[c].notna().any()]
        row = {"row": label.replace(" (Decoupled)", ""), "device": chosen["device"].iloc[0], "n_seeds": len(chosen)}
        row.update({c: float(chosen[c].median()) for c in size_columns})
        for column in ("time_pretrain_seconds", "time_adapt_seconds", "time_infer_ms_per_flow",
                       "time_peak_memory_mb", "time_checkpoint_mb"):
            row[column] = float(chosen[column].median()) if column in chosen and chosen[column].notna().any() else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def condition_a(flat: pd.DataFrame, budget: float) -> pd.DataFrame:
    """Shortcut check: Random Forest with and without port/protocol."""
    rows = []
    for condition in ("A", "B"):
        chosen = select(flat, "RF", "msp", budget, condition=condition)
        row = {"row": f"Random Forest, Condition {condition}", "n_seeds": len(chosen)}
        for metric in ("auroc", "auprc", "benign_recall", "benign_fpr"):
            stats = mean_std(chosen[metric]) if metric in chosen else {"mean": np.nan, "std": np.nan}
            row[f"{metric}_mean"], row[f"{metric}_std"] = stats["mean"], stats["std"]
        rows.append(row)
    return pd.DataFrame(rows)


def profile_bot_flows(test_frame: pd.DataFrame, is_zero_day: np.ndarray, scores: np.ndarray,
                      threshold: float, column: str, cut: float) -> pd.DataFrame:
    """Split zero-day test flows at ``column > cut`` and report each group's size and recall."""
    if column not in test_frame.columns:
        raise KeyError(f"profiling column {column!r} is not in the test frame")
    positive = np.asarray(is_zero_day).astype(bool)
    values = test_frame[column].to_numpy()[positive]
    flagged = (np.asarray(scores) > threshold)[positive]
    rows = []
    for name, members in ((f"{column} > {cut:g}", values > cut), (f"{column} <= {cut:g}", values <= cut)):
        rows.append({"group": name, "flows": int(members.sum()), "detected": int(flagged[members].sum()),
                     "recall": float(flagged[members].mean()) if members.any() else float("nan")})
    return pd.DataFrame(rows)


def format_pm(mean: float, std: float, digits: int = 4) -> str:
    if pd.isna(mean):
        return "N/A"
    return f"{mean:.{digits}f}" if pd.isna(std) else f"{mean:.{digits}f} $\\pm$ {std:.{digits}f}"


def to_latex_pm(table: pd.DataFrame, digits: int = 4) -> str:
    """LaTeX tabular body where every ``*_mean``/``*_std`` pair becomes ``mean ± std``."""
    metrics = [c[:-5] for c in table.columns if c.endswith("_mean")]
    lines = [" & ".join(["Model", *[m.replace("_", r"\_") for m in metrics]]) + r" \\ \hline"]
    for _, row in table.iterrows():
        cells = [format_pm(row[f"{m}_mean"], row[f"{m}_std"], digits) for m in metrics]
        lines.append(" & ".join([str(row["row"]).replace("_", r"\_"), *cells]) + r" \\")
    return "\n".join(lines)


def write_report(flat: pd.DataFrame, cfg: Mapping[str, Any], report_dir: Path) -> dict[str, pd.DataFrame]:
    """Write every table as CSV (full precision) and LaTeX. Returns the tables."""
    budget, seed = cfg["matrix"]["primary_budget"], cfg["matrix"]["pilot_seed"]
    tables = {
        "table2_cost": table2(flat, budget),
        "table3_ranking_closed_world": table3(flat, budget),
        "table4a_operational_benign": table4_group_a(flat, budget, "benign"),
        "table4a_operational_mixture": table4_group_a(flat, budget, "mixture"),
        "table4b_seed_case_study": table4_group_b(flat, budget, seed),
        "table5a_statistics": table5a(flat, budget),
        "table5b_ablation": table5b(flat, budget),
        "s4_e06_all_scorers": scorer_table(flat, budget),
        "fig2_budget_trajectory": budget_trajectory(flat),
        "condition_a_shortcut": condition_a(flat, budget),
    }
    report_dir.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        table.to_csv(report_dir / f"{name}.csv", index=False)
        if any(c.endswith("_mean") for c in table.columns):
            (report_dir / f"{name}.tex").write_text(to_latex_pm(table), encoding="utf-8")
        else:
            (report_dir / f"{name}.tex").write_text(
                table.to_latex(index=False, float_format="%.4f"), encoding="utf-8")
    return tables
