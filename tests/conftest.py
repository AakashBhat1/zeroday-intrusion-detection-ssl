"""Shared fixtures: a synthetic dataset pushed through the real pipeline once per session."""

from __future__ import annotations

from pathlib import Path

import pytest

from zdssl.config import load_config, resolve_paths
from zdssl.data import ingest_all, verify_counts
from zdssl.splits import build_splits, save_splits
from zdssl.synth import make_synthetic_csvs, synthetic_overrides

CONFIG = Path(__file__).resolve().parents[1] / "configs" / "paper1.yaml"


@pytest.fixture(scope="session")
def pipeline(tmp_path_factory):
    work = tmp_path_factory.mktemp("synthetic")
    counts = make_synthetic_csvs(work / "raw")
    cfg = load_config(CONFIG, synthetic_overrides(work, counts))
    day_stats = ingest_all(cfg)
    summary = verify_counts(day_stats, cfg)
    frame, manifest = build_splits(cfg, day_stats)
    paths = resolve_paths(cfg)
    save_splits(frame, manifest, paths["splits"], paths["manifest"])
    return {"cfg": cfg, "day_stats": day_stats, "summary": summary, "frame": frame,
            "manifest": manifest, "paths": paths, "counts": counts}
