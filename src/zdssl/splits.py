"""Build the historical sample, the train/validation splits and the locked test set.

Split names stored in the ``split`` column:
  train    80% of the historical sample; also the SSL pretraining pool (labels hidden)
  tune     half of validation; its benign flows calibrate the alarm threshold
  cw_test  other half of validation; closed-world 14-class test
  test     locked test day, benign + zero-day
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.model_selection import train_test_split

from .config import write_text_atomic
from .data import DAY_STRIDE, ROW_ID
from .features import SPLIT_COL, behavioral_columns, select_features

log = logging.getLogger(__name__)

TRAIN, TUNE, CW_TEST, TEST = "train", "tune", "cw_test", "test"
HISTORICAL = (TRAIN, TUNE, CW_TEST)


class LeakageError(RuntimeError):
    """A split invariant is violated."""


def collect_candidates(
    parquet_paths: Iterable[Path], keep_prob: Mapping[str, float], label_col: str,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Stream the parquet files and keep each row with its class's probability."""
    kept = []
    for path in parquet_paths:
        parquet = pq.ParquetFile(path)
        for group in range(parquet.num_row_groups):
            frame = parquet.read_row_group(group).to_pandas()
            prob = frame[label_col].map(keep_prob).fillna(0.0).to_numpy()
            mask = rng.random(len(frame)) < prob
            if mask.any():
                kept.append(frame.loc[mask])
    if not kept:
        raise LeakageError("no candidate rows were collected")
    return pd.concat(kept, ignore_index=True)


def dedup_exact(frame: pd.DataFrame, label_col: str) -> tuple[pd.DataFrame, dict[str, int]]:
    """Drop rows identical in every column except the row id. Returns removed counts per class."""
    subset = [c for c in frame.columns if c != ROW_ID]
    duplicated = frame.duplicated(subset=subset, keep="first")
    removed = frame.loc[duplicated, label_col].value_counts().to_dict()
    return frame.loc[~duplicated].reset_index(drop=True), {k: int(v) for k, v in removed.items()}


def cap_per_class(
    frame: pd.DataFrame, caps: Mapping[str, int], label_col: str, rng: np.random.Generator
) -> pd.DataFrame:
    """Random sample of at most ``caps[label]`` rows per class (classes absent from caps are dropped)."""
    parts = []
    for label, group in frame.groupby(label_col, sort=True):
        cap = caps.get(label, 0)
        if cap <= 0:
            continue
        if len(group) > cap:
            group = group.iloc[np.sort(rng.choice(len(group), size=cap, replace=False))]
        parts.append(group)
    return pd.concat(parts, ignore_index=True)


def feature_hash(frame: pd.DataFrame, columns: list[str]) -> np.ndarray:
    """One uint64 per row over the behavioral columns (NaN-stable)."""
    return pd.util.hash_pandas_object(frame[columns], index=False).to_numpy()


def budget_subset(labels: np.ndarray, fraction: float, seed: int, min_per_class: int = 1) -> np.ndarray:
    """Stratified subset of positions: round(fraction * n_c) per class, at least ``min_per_class``."""
    if not 0 < fraction <= 1:
        raise ValueError(f"fraction must be in (0, 1], got {fraction}")
    if fraction == 1:
        return np.arange(len(labels))
    rng = np.random.default_rng(seed)
    chosen = []
    for label in np.unique(labels):
        positions = np.flatnonzero(labels == label)
        take = min(len(positions), max(min_per_class, int(round(fraction * len(positions)))))
        chosen.append(rng.choice(positions, size=take, replace=False))
    return np.sort(np.concatenate(chosen))


def _sample_historical(day_stats: list[dict], cfg: Mapping[str, Any], rng: np.random.Generator):
    data_cfg, split_cfg = cfg["data"], cfg["splits"]
    label_col = data_cfg["label_column"]
    parquet_dir = Path(cfg["paths"]["parquet_dir"])
    historical = [s for s in day_stats if s["day"] != data_cfg["test_date"]]
    totals: dict[str, int] = {}
    for stats in historical:
        for label, count in stats["label_counts"].items():
            totals[label] = totals.get(label, 0) + count
    totals.pop(data_cfg["zero_day_label"], None)
    caps = {
        label: split_cfg["benign_cap"] if label == data_cfg["benign_label"] else split_cfg["attack_cap"]
        for label in totals
    }
    keep_prob = {
        label: min(1.0, split_cfg["candidate_factor"] * caps[label] / totals[label]) for label in totals
    }
    candidates = collect_candidates(
        (parquet_dir / s["parquet"] for s in historical), keep_prob, label_col, rng
    )
    unique, removed = dedup_exact(candidates, label_col)
    sample = cap_per_class(unique, caps, label_col, rng)
    info = {
        "historical_class_totals": dict(sorted(totals.items())),
        "candidates": int(len(candidates)),
        "exact_duplicates_removed": removed,
        "sample_class_counts": sample[label_col].value_counts().sort_index().to_dict(),
    }
    return sample, info


def _sample_test(
    day_stats: list[dict], cfg: Mapping[str, Any], historical_hashes: np.ndarray,
    hash_cols: list[str], rng: np.random.Generator,
):
    data_cfg, split_cfg = cfg["data"], cfg["splits"]
    label_col = data_cfg["label_column"]
    test_days = [s for s in day_stats if s["day"] == data_cfg["test_date"]]
    if len(test_days) != 1:
        raise LeakageError(f"expected exactly one test day, found {len(test_days)}")
    frame = pd.read_parquet(Path(cfg["paths"]["parquet_dir"]) / test_days[0]["parquet"])
    wanted = [data_cfg["benign_label"], data_cfg["zero_day_label"]]
    frame = frame.loc[frame[label_col].isin(wanted)].reset_index(drop=True)
    unique, removed = dedup_exact(frame, label_col)
    seen = np.isin(feature_hash(unique, hash_cols), historical_hashes)
    fresh = unique.loc[~seen].reset_index(drop=True)
    caps = {label: split_cfg["test_per_class"] for label in wanted}
    sample = cap_per_class(fresh, caps, label_col, rng)
    info = {
        "test_day_rows": int(len(frame)),
        "test_exact_duplicates_removed": removed,
        "test_rows_removed_seen_in_historical": int(seen.sum()),
        "test_class_counts": sample[label_col].value_counts().sort_index().to_dict(),
    }
    return sample, info


def _assign_historical_splits(sample: pd.DataFrame, cfg: Mapping[str, Any]) -> pd.Series:
    split_cfg, label_col = cfg["splits"], cfg["data"]["label_column"]
    positions = np.arange(len(sample))
    labels = sample[label_col].to_numpy()
    train_pos, val_pos = train_test_split(
        positions, test_size=split_cfg["val_fraction"], stratify=labels, random_state=split_cfg["seed"]
    )
    tune_pos, cw_pos = train_test_split(
        val_pos, train_size=split_cfg["tune_fraction"], stratify=labels[val_pos],
        random_state=split_cfg["seed"],
    )
    assignment = np.empty(len(sample), dtype=object)
    assignment[train_pos], assignment[tune_pos], assignment[cw_pos] = TRAIN, TUNE, CW_TEST
    return pd.Series(assignment, index=sample.index)


def capture_days(frame: pd.DataFrame, days: list[str]) -> pd.Series:
    """Capture day of each row, from the file it came from (encoded in the row id)."""
    return (frame[ROW_ID] // DAY_STRIDE).map(dict(enumerate(days)))


def _quarantine(frame: pd.DataFrame, ts_col: str, days: list[str]) -> dict:
    """Gap between the last historical flow and the first test flow, in seconds.

    Only timestamps that fall on the row's own capture day are trusted; the rest
    (unparseable or mis-dated in the source files) are counted and left out.
    """
    stamps = frame[ts_col]
    trusted = stamps.dt.strftime("%Y-%m-%d") == capture_days(frame, days)
    first_test = stamps[trusted & (frame[SPLIT_COL] == TEST)].min()
    gaps = {"timestamps_not_on_capture_day": int((~trusted).sum())}
    for name, members in (("train", [TRAIN]), ("validation", [TUNE, CW_TEST])):
        last = stamps[trusted & frame[SPLIT_COL].isin(members)].max()
        gaps[f"seconds_from_{name}"] = float((first_test - last).total_seconds())
        gaps[f"last_{name}_flow"] = str(last)
    gaps["first_test_flow"] = str(first_test)
    # If no hour exceeds 12 the source uses a 12-hour clock without AM/PM, and the gap is a lower bound.
    gaps["max_hour_of_day_seen"] = int(stamps[trusted].dt.hour.max())
    return gaps


def check_leakage(frame: pd.DataFrame, manifest: Mapping[str, Any], cfg: Mapping[str, Any]) -> dict:
    """Run every split invariant; raise ``LeakageError`` listing all failures."""
    data_cfg = cfg["data"]
    label_col, zero_day = data_cfg["label_column"], data_cfg["zero_day_label"]
    split = frame[SPLIT_COL]
    hashes = feature_hash(frame, manifest["hash_columns"])
    train_hashes = hashes[(split == TRAIN).to_numpy()]
    historical_hashes = hashes[split.isin(HISTORICAL).to_numpy()]
    test_hashes = hashes[(split == TEST).to_numpy()]
    ssl_ids = set(manifest["ssl_pool_row_ids"])
    val_ids = set(frame.loc[split.isin([TUNE, CW_TEST]), ROW_ID])
    quarantine_hours = manifest["quarantine"]["seconds_from_train"] / 3600
    checks = {
        "row_ids_unique_across_splits": bool(frame[ROW_ID].is_unique),
        "only_known_split_names": bool(split.isin([*HISTORICAL, TEST]).all()),
        "no_zero_day_outside_test": not bool(((frame[label_col] == zero_day) & (split != TEST)).any()),
        "test_has_only_benign_and_zero_day": bool(
            frame.loc[split == TEST, label_col].isin([data_cfg["benign_label"], zero_day]).all()
        ),
        "ssl_pool_equals_train": ssl_ids == set(frame.loc[split == TRAIN, ROW_ID]),
        "ssl_pool_disjoint_from_validation": not (ssl_ids & val_ids),
        "no_feature_duplicates_train_vs_test": not bool(np.isin(test_hashes, train_hashes).any()),
        "no_feature_duplicates_historical_vs_test": not bool(np.isin(test_hashes, historical_hashes).any()),
        "historical_rows_come_from_days_before_test_day": bool(
            (capture_days(frame, manifest["days"])[split != TEST] < data_cfg["test_date"]).all()
        ),
        "test_rows_come_from_test_day": bool(
            (capture_days(frame, manifest["days"])[split == TEST] == data_cfg["test_date"]).all()
        ),
        "quarantine_at_least_minimum": quarantine_hours >= cfg["splits"]["min_quarantine_hours"],
    }
    failed = [name for name, ok in checks.items() if not ok]
    if failed:
        raise LeakageError(f"leakage checks failed: {failed}")
    return checks


def build_splits(cfg: Mapping[str, Any], day_stats: list[dict]) -> tuple[pd.DataFrame, dict]:
    """Sample, split, select features and verify. Returns the split frame and its manifest."""
    data_cfg = cfg["data"]
    label_col, ts_col = data_cfg["label_column"], data_cfg["timestamp_column"]
    rng = np.random.default_rng(cfg["splits"]["seed"])

    historical, hist_info = _sample_historical(day_stats, cfg, rng)
    historical[SPLIT_COL] = _assign_historical_splits(historical, cfg)
    hash_cols = behavioral_columns(historical.columns, cfg)
    test, test_info = _sample_test(day_stats, cfg, feature_hash(historical, hash_cols), hash_cols, rng)
    test[SPLIT_COL] = TEST
    frame = pd.concat([historical, test], ignore_index=True)

    train = frame.loc[frame[SPLIT_COL] == TRAIN]
    features_b, dropped = select_features(train, hash_cols)
    hashes = feature_hash(frame, hash_cols)
    is_train = (frame[SPLIT_COL] == TRAIN).to_numpy()
    is_val = frame[SPLIT_COL].isin([TUNE, CW_TEST]).to_numpy()
    classes = sorted(set(historical[label_col]))
    manifest = {
        "classes": classes,
        "benign_label": data_cfg["benign_label"],
        "zero_day_label": data_cfg["zero_day_label"],
        "hash_columns": hash_cols,
        "features_B": features_b,
        "features_A": [*data_cfg["id_columns"], *features_b],
        "dropped_constant_columns": dropped,
        "n_features_B": len(features_b),
        "expected_n_features_B": cfg["features"]["expected_condition_b"],
        "split_sizes": frame[SPLIT_COL].value_counts().to_dict(),
        "split_class_counts": {
            name: group[label_col].value_counts().sort_index().to_dict()
            for name, group in frame.groupby(SPLIT_COL)
        },
        "ssl_pool_row_ids": [int(i) for i in train[ROW_ID]],
        "validation_rows_sharing_a_train_feature_profile": int(np.isin(hashes[is_val], hashes[is_train]).sum()),
        "days": [s["day"] for s in day_stats],
        "quarantine": _quarantine(frame, ts_col, [s["day"] for s in day_stats]),
        "splits_seed": cfg["splits"]["seed"],
        **hist_info,
        **test_info,
    }
    manifest["leakage_checks"] = check_leakage(frame, manifest, cfg)
    if manifest["n_features_B"] != manifest["expected_n_features_B"]:
        log.warning("Condition B has %d features, the paper says %d",
                    manifest["n_features_B"], manifest["expected_n_features_B"])
    return frame, manifest


def save_splits(frame: pd.DataFrame, manifest: Mapping[str, Any], splits_path: Path, manifest_path: Path) -> None:
    splits_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = splits_path.with_suffix(".tmp")
    frame.to_parquet(tmp, index=False, compression="zstd")
    tmp.replace(splits_path)
    write_text_atomic(manifest_path, json.dumps(manifest, indent=2, default=str))


def load_splits(splits_path: Path, manifest_path: Path) -> tuple[pd.DataFrame, dict]:
    if not splits_path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError(f"splits not built yet: {splits_path} / {manifest_path}")
    return pd.read_parquet(splits_path), json.loads(manifest_path.read_text(encoding="utf-8"))
