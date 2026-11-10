"""Small synthetic stand-in for the CSE-CIC-IDS2018 CSVs (tests and local smoke runs only).

It reproduces the quirks the real files have: repeated header rows, ``Infinity``
values, extra columns in the Feb 20 file, exact duplicate rows, constant-zero
columns, and a zero-day class that exists only on the last day.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

DAYS = [
    ("Wednesday", "14-02-2018"), ("Thursday", "15-02-2018"), ("Friday", "16-02-2018"),
    ("Thuesday", "20-02-2018"), ("Wednesday", "21-02-2018"), ("Thursday", "22-02-2018"),
    ("Friday", "23-02-2018"), ("Wednesday", "28-02-2018"), ("Thursday", "01-03-2018"),
    ("Friday", "02-03-2018"),
]
ATTACKS = [
    "FTP-BruteForce", "SSH-Bruteforce", "DoS attacks-GoldenEye", "DoS attacks-Slowloris",
    "DoS attacks-SlowHTTPTest", "DoS attacks-Hulk", "DDoS attacks-LOIC-HTTP", "DDOS attack-HOIC",
    "DDOS attack-LOIC-UDP", "Brute Force -Web", "Brute Force -XSS", "SQL Injection", "Infilteration",
]
N_VARYING = 12
CONSTANT = ["Bwd PSH Flags", "Bwd URG Flags"]
VARYING = ["Flow Duration", "Flow Byts/s", "Flow Pkts/s"] + [f"Feat {i}" for i in range(N_VARYING - 3)]
COLUMNS = ["Dst Port", "Protocol", "Timestamp", *VARYING, *CONSTANT, "Label"]
EXTRA = ["Flow ID", "Src IP", "Src Port", "Dst IP"]


def _class_rows(label: str, n: int, class_seed: int, rng: np.random.Generator) -> pd.DataFrame:
    # Each class has its own mean pattern across features (benign, seed 0, sits at the origin).
    centre = np.random.default_rng(class_seed).normal(0.0, 2.0, N_VARYING) if class_seed else 0.0
    values = rng.normal(loc=centre, scale=1.0, size=(n, N_VARYING))
    values[:, 0] = np.abs(values[:, 0]) * 1e6  # duration-like heavy scale
    frame = pd.DataFrame(values, columns=VARYING)
    for col in CONSTANT:
        frame[col] = 0
    frame.insert(0, "Protocol", rng.choice([6, 17], size=n))
    frame.insert(0, "Dst Port", rng.integers(1, 1024, size=n))
    frame["Label"] = label
    return frame


def make_synthetic_csvs(out_dir: Path, rows_per_day: int = 1500, seed: int = 0) -> dict:
    """Write ten daily CSVs into ``out_dir``. Returns the expected raw-row and zero-day counts."""
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    raw_rows = zero_day_rows = 0
    for index, (weekday, stamp) in enumerate(DAYS):
        is_test_day = index == len(DAYS) - 1
        n_benign = rows_per_day // 2
        parts = [_class_rows("Benign", n_benign, 0, rng)]
        if is_test_day:
            parts.append(_class_rows("Bot", rows_per_day - n_benign, 99, rng))
        else:
            per_attack = (rows_per_day - n_benign) // len(ATTACKS)
            for k, attack in enumerate(ATTACKS):
                parts.append(_class_rows(attack, per_attack, k + 1, rng))
        frame = pd.concat(parts, ignore_index=True)
        frame = frame.iloc[rng.permutation(len(frame))].reset_index(drop=True)
        day, month, year = stamp.split("-")
        seconds = np.sort(rng.integers(8 * 3600, 11 * 3600, size=len(frame)))
        frame["Timestamp"] = [
            f"{day}/{month}/{year} {s // 3600:02d}:{s % 3600 // 60:02d}:{s % 60:02d}" for s in seconds
        ]
        frame = frame[COLUMNS].astype(object)
        frame.iloc[5, frame.columns.get_loc("Flow Byts/s")] = "Infinity"
        frame.iloc[7, frame.columns.get_loc("Flow Pkts/s")] = np.nan
        frame = pd.concat([frame, frame.iloc[[10, 11]]], ignore_index=True)  # exact duplicates
        zero_day_rows += int((frame["Label"] == "Bot").sum())
        header = pd.DataFrame([COLUMNS], columns=COLUMNS)
        frame = pd.concat([frame.iloc[:100], header, frame.iloc[100:]], ignore_index=True)
        if stamp == "20-02-2018":
            for col in EXTRA:
                frame.insert(0, col, col if col != "Flow ID" else "id")
            frame.loc[frame["Label"] == "Label", EXTRA] = EXTRA
        raw_rows += len(frame)
        frame.to_csv(out_dir / f"{weekday}-{stamp}_TrafficForML_CICFlowMeter.csv", index=False)
    return {"raw_rows": raw_rows, "zero_day_rows": zero_day_rows}


def synthetic_overrides(work_dir: Path, counts: dict) -> dict:
    """Config overrides that point the pipeline at a synthetic dataset with tiny settings."""
    return {
        "paths": {
            "raw_dir": str(work_dir / "raw"),
            "parquet_dir": str(work_dir / "parquet"),
            "drive_root": str(work_dir / "drive"),
        },
        "data": {
            "expected_raw_rows": counts["raw_rows"],
            "expected_zero_day_rows": counts["zero_day_rows"],
            "chunk_rows": 400,
        },
        "splits": {"benign_cap": 2000, "attack_cap": 300, "test_per_class": 400},
        "features": {"expected_condition_b": N_VARYING},
        "pretrain": {"epochs": 2, "warmup_epochs": 1, "batch_size": 128, "centroid_min_flows": 100},
        "probe": {"epochs": 2},
        "finetune": {"epochs": 2},
        "scarf": {"epochs": 2, "batch_size": 128},
        "svdd": {"epochs": 2},
        "trees": {"rf": {"n_estimators": 10, "max_depth": 5}, "xgb": {"n_estimators": 5}},
        "matrix": {"seeds": [42, 43], "budgets": [1.0, 0.1]},
    }
