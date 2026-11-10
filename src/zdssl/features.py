"""Feature selection (Conditions A and B) and train-only preprocessing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler

from .data import ROW_ID

SPLIT_COL = "split"


def behavioral_columns(columns: Sequence[str], cfg: Mapping[str, Any]) -> list[str]:
    """Every flow statistic: all columns except identifiers, timestamp, label and bookkeeping."""
    data_cfg = cfg["data"]
    excluded = {
        ROW_ID, SPLIT_COL, data_cfg["label_column"], data_cfg["timestamp_column"],
        *data_cfg["id_columns"], *data_cfg["extra_columns"],
    }
    return [c for c in columns if c not in excluded]


def select_features(train: pd.DataFrame, candidates: Sequence[str]) -> tuple[list[str], list[str]]:
    """Split ``candidates`` into (kept, dropped) where dropped columns are constant on train."""
    values = train[list(candidates)].to_numpy(dtype="float64")
    all_nan = np.isnan(values).all(axis=0)
    spread = np.where(all_nan, 0.0, np.nanmax(np.where(np.isnan(values), -np.inf, values), axis=0)
                      - np.nanmin(np.where(np.isnan(values), np.inf, values), axis=0))
    kept = [c for c, s in zip(candidates, spread) if s > 0]
    dropped = [c for c, s in zip(candidates, spread) if not s > 0]
    return kept, dropped


@dataclass(frozen=True)
class Preprocessor:
    """Median imputation + RobustScaler + clipping, all fitted on the training split."""

    columns: tuple[str, ...]
    median: np.ndarray
    center: np.ndarray
    scale: np.ndarray
    clip_abs: float

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        values = frame[list(self.columns)].to_numpy(dtype="float64")
        values = np.where(np.isnan(values), self.median, values)
        scaled = (values - self.center) / self.scale
        return np.clip(scaled, -self.clip_abs, self.clip_abs).astype("float32")


def fit_preprocessor(
    train: pd.DataFrame, columns: Sequence[str], clip_abs: float, unit_variance: bool = True
) -> Preprocessor:
    """Fit on the 100% training split only; reused unchanged for every budget and split."""
    if clip_abs <= 0:
        raise ValueError("clip_abs must be positive")
    values = train[list(columns)].to_numpy(dtype="float64")
    median = np.nanmedian(values, axis=0)
    if np.isnan(median).any():
        bad = [c for c, m in zip(columns, median) if np.isnan(m)]
        raise ValueError(f"columns are entirely NaN on the training split: {bad}")
    filled = np.where(np.isnan(values), median, values)
    scaler = RobustScaler(unit_variance=unit_variance).fit(filled)
    return Preprocessor(
        columns=tuple(columns),
        median=median,
        center=scaler.center_.copy(),
        scale=scaler.scale_.copy(),
        clip_abs=float(clip_abs),
    )
