"""Download, clean and verify the CSE-CIC-IDS2018 daily CSVs.

Each daily CSV becomes one cleaned parquet file (written chunk by chunk so the
3.8 GB Feb 20 file never has to fit in memory) plus a small stats JSON. Both
are skipped on re-run, so a Colab disconnect only costs the file in progress.
"""

from __future__ import annotations

import json
import logging
import re
import subprocess
from datetime import date
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from .config import write_text_atomic

log = logging.getLogger(__name__)

ROW_ID = "row_id"
DAY_STRIDE = 100_000_000  # row_id = day_index * DAY_STRIDE + row number within the day
_DATE_RE = re.compile(r"(\d{2})-(\d{2})-(\d{4})")


class DataVerificationError(RuntimeError):
    """The downloaded data does not match the expected dataset."""


def download_raw(cfg: Mapping[str, Any]) -> None:
    """Sync the official public bucket prefix into ``paths.raw_dir`` (resumable)."""
    raw_dir = Path(cfg["paths"]["raw_dir"])
    raw_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "aws", "s3", "sync", "--no-sign-request", "--region", cfg["data"]["s3_region"],
        cfg["data"]["s3_uri"], str(raw_dir),
    ]
    log.info("running: %s", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise DataVerificationError(f"aws s3 sync failed: {result.stderr.strip()[-2000:]}")


def parse_day(filename: str) -> date:
    """Capture day from a name like ``Friday-02-03-2018_TrafficForML_CICFlowMeter.csv``."""
    match = _DATE_RE.search(filename)
    if not match:
        raise DataVerificationError(f"no dd-mm-yyyy date in file name: {filename}")
    day, month, year = (int(g) for g in match.groups())
    return date(year, month, day)


def clean_chunk(chunk: pd.DataFrame, cfg: Mapping[str, Any]) -> tuple[pd.DataFrame, int]:
    """Clean one CSV chunk. Returns the cleaned frame and the number of header rows removed.

    Repeated header rows are dropped, the Feb 20 extra columns are dropped, every
    feature becomes float64 with ``inf`` turned into NaN, timestamps are parsed.
    """
    data_cfg = cfg["data"]
    label_col, ts_col = data_cfg["label_column"], data_cfg["timestamp_column"]
    chunk = chunk.rename(columns=lambda c: str(c).strip())
    if label_col not in chunk.columns or ts_col not in chunk.columns:
        raise DataVerificationError(f"chunk lacks {label_col!r} or {ts_col!r} column")

    labels = chunk[label_col].astype(str).str.strip()
    is_header = labels == label_col
    kept = chunk.loc[~is_header]
    labels = labels.loc[~is_header]
    extra = [c for c in data_cfg["extra_columns"] if c in kept.columns]
    numeric_cols = [c for c in kept.columns if c not in (label_col, ts_col, *extra)]

    numeric = {}
    for col in numeric_cols:
        values = kept[col]
        if values.dtype == object:
            values = pd.to_numeric(values, errors="coerce")
        numeric[col] = values.astype("float64").replace([np.inf, -np.inf], np.nan)

    cleaned = pd.DataFrame(numeric, index=kept.index)
    cleaned[ts_col] = pd.to_datetime(
        kept[ts_col], format=data_cfg["timestamp_format"], errors="coerce"
    )
    cleaned[label_col] = labels
    return cleaned.reset_index(drop=True), int(is_header.sum())


def ingest_day(csv_path: Path, out_path: Path, day_index: int, cfg: Mapping[str, Any]) -> dict:
    """Clean one daily CSV into ``out_path``. Returns the day's statistics."""
    stats_path = out_path.with_suffix(".stats.json")
    if out_path.is_file() and stats_path.is_file():
        return json.loads(stats_path.read_text(encoding="utf-8"))

    label_col = cfg["data"]["label_column"]
    ts_col = cfg["data"]["timestamp_column"]
    tmp_path = out_path.with_suffix(".tmp")
    raw_rows = header_rows = clean_rows = bad_timestamps = 0
    label_counts: dict[str, int] = {}
    columns: list[str] | None = None
    writer: pq.ParquetWriter | None = None
    try:
        reader = pd.read_csv(csv_path, chunksize=cfg["data"]["chunk_rows"], low_memory=False)
        for chunk in reader:
            raw_rows += len(chunk)
            cleaned, n_header = clean_chunk(chunk, cfg)
            header_rows += n_header
            if columns is None:
                columns = list(cleaned.columns)
            elif list(cleaned.columns) != columns:
                raise DataVerificationError(f"column order changed inside {csv_path.name}")
            cleaned.insert(0, ROW_ID, day_index * DAY_STRIDE + clean_rows + np.arange(len(cleaned)))
            clean_rows += len(cleaned)
            bad_timestamps += int(cleaned[ts_col].isna().sum())
            for label, count in cleaned[label_col].value_counts().items():
                label_counts[label] = label_counts.get(label, 0) + int(count)
            table = pa.Table.from_pandas(cleaned, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(tmp_path, table.schema, compression="zstd")
            writer.write_table(table.cast(writer.schema))
    finally:
        if writer is not None:
            writer.close()
    if writer is None or columns is None:
        raise DataVerificationError(f"{csv_path.name} contains no rows")
    if clean_rows >= DAY_STRIDE:
        raise DataVerificationError(f"{csv_path.name} has more rows than the row-id stride")

    stats = {
        "file": csv_path.name,
        "day": parse_day(csv_path.name).isoformat(),
        "day_index": day_index,
        "parquet": out_path.name,
        "raw_rows": raw_rows,
        "header_rows": header_rows,
        "clean_rows": clean_rows,
        "bad_timestamps": bad_timestamps,
        "label_counts": label_counts,
        "columns": columns,
    }
    tmp_path.replace(out_path)
    write_text_atomic(stats_path, json.dumps(stats, indent=2))
    log.info("ingested %s: %d rows (%d header rows removed)", csv_path.name, clean_rows, header_rows)
    return stats


def ingest_all(cfg: Mapping[str, Any]) -> list[dict]:
    """Clean every daily CSV in ``paths.raw_dir``; returns per-day stats sorted by day."""
    raw_dir = Path(cfg["paths"]["raw_dir"])
    parquet_dir = Path(cfg["paths"]["parquet_dir"])
    parquet_dir.mkdir(parents=True, exist_ok=True)
    csv_paths = sorted(raw_dir.rglob("*.csv"), key=lambda p: parse_day(p.name))
    if not csv_paths:
        raise DataVerificationError(f"no CSV files under {raw_dir}")
    return [
        ingest_day(path, parquet_dir / f"{parse_day(path.name).isoformat()}.parquet", index, cfg)
        for index, path in enumerate(csv_paths)
    ]


def summarize(day_stats: list[dict]) -> dict:
    """Dataset-level totals from the per-day stats."""
    label_counts: dict[str, int] = {}
    for stats in day_stats:
        for label, count in stats["label_counts"].items():
            label_counts[label] = label_counts.get(label, 0) + count
    return {
        "days": [s["day"] for s in day_stats],
        "raw_rows": sum(s["raw_rows"] for s in day_stats),
        "header_rows": sum(s["header_rows"] for s in day_stats),
        "clean_rows": sum(s["clean_rows"] for s in day_stats),
        "bad_timestamps": sum(s["bad_timestamps"] for s in day_stats),
        "label_counts": dict(sorted(label_counts.items())),
    }


def verify_counts(day_stats: list[dict], cfg: Mapping[str, Any]) -> dict:
    """Stop unless the data is the official dataset. Returns the summary."""
    data_cfg = cfg["data"]
    summary = summarize(day_stats)
    zero_day = data_cfg["zero_day_label"]
    problems = []
    if summary["days"] != list(data_cfg["expected_days"]):
        problems.append(f"days {summary['days']} != expected {data_cfg['expected_days']}")
    if summary["raw_rows"] != data_cfg["expected_raw_rows"]:
        problems.append(f"raw rows {summary['raw_rows']} != {data_cfg['expected_raw_rows']}")
    n_zero_day = summary["label_counts"].get(zero_day, 0)
    if n_zero_day != data_cfg["expected_zero_day_rows"]:
        problems.append(f"{zero_day} rows {n_zero_day} != {data_cfg['expected_zero_day_rows']}")
    if len(summary["label_counts"]) != data_cfg["expected_n_labels"]:
        problems.append(f"labels found: {sorted(summary['label_counts'])}")
    for stats in day_stats:
        if stats["day"] != data_cfg["test_date"] and stats["label_counts"].get(zero_day, 0):
            problems.append(f"{zero_day} rows found on historical day {stats['day']}")
    if len({tuple(s["columns"]) for s in day_stats}) != 1:
        problems.append("daily files do not share one column set after cleaning")
    if problems:
        raise DataVerificationError("; ".join(problems))
    return summary
