"""Load the saved splits and turn them into model-ready arrays."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import pandas as pd

from .config import resolve_paths
from .features import SPLIT_COL, fit_preprocessor
from .splits import CW_TEST, TEST, TRAIN, TUNE, load_splits

CONDITIONS = ("A", "B")


@dataclass(frozen=True)
class Dataset:
    condition: str
    feature_names: tuple[str, ...]
    classes: tuple[str, ...]      # the catalogued classes, index = class id
    benign_index: int
    x_train: np.ndarray           # 100% training split = SSL pretraining pool
    y_train: np.ndarray
    x_tune: np.ndarray
    y_tune: np.ndarray
    x_cw: np.ndarray
    y_cw: np.ndarray
    x_test: np.ndarray
    test_is_zero_day: np.ndarray  # 1 = zero-day flow, 0 = benign

    @property
    def n_classes(self) -> int:
        return len(self.classes)


def build_dataset(frame: pd.DataFrame, manifest: Mapping[str, Any], cfg: Mapping[str, Any],
                  condition: str = "B") -> Dataset:
    """Fit the preprocessor on the training split only and transform every split with it."""
    if condition not in CONDITIONS:
        raise ValueError(f"condition must be one of {CONDITIONS}, got {condition!r}")
    label_col = cfg["data"]["label_column"]
    columns = manifest[f"features_{condition}"]
    classes = tuple(manifest["classes"])
    class_id = {name: i for i, name in enumerate(classes)}
    parts = {name: frame.loc[frame[SPLIT_COL] == name] for name in (TRAIN, TUNE, CW_TEST, TEST)}
    pre = fit_preprocessor(parts[TRAIN], columns, cfg["features"]["clip_abs"],
                           cfg["features"]["unit_variance"])

    def labels(part: pd.DataFrame) -> np.ndarray:
        ids = part[label_col].map(class_id)
        if ids.isna().any():
            raise ValueError("a historical split contains a label outside the catalogued classes")
        return ids.to_numpy(dtype="int64")

    return Dataset(
        condition=condition,
        feature_names=tuple(columns),
        classes=classes,
        benign_index=class_id[manifest["benign_label"]],
        x_train=pre.transform(parts[TRAIN]), y_train=labels(parts[TRAIN]),
        x_tune=pre.transform(parts[TUNE]), y_tune=labels(parts[TUNE]),
        x_cw=pre.transform(parts[CW_TEST]), y_cw=labels(parts[CW_TEST]),
        x_test=pre.transform(parts[TEST]),
        test_is_zero_day=(parts[TEST][label_col] == manifest["zero_day_label"]).to_numpy(dtype="int64"),
    )


def load_dataset(cfg: Mapping[str, Any], condition: str = "B") -> Dataset:
    paths = resolve_paths(cfg)
    frame, manifest = load_splits(paths["splits"], paths["manifest"])
    return build_dataset(frame, manifest, cfg, condition)
