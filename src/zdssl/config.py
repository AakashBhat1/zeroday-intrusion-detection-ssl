"""Configuration loading and path resolution."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import yaml

REQUIRED_SECTIONS = (
    "paths", "data", "splits", "features", "model", "pretrain", "probe",
    "finetune", "scarf", "svdd", "trees", "eval", "matrix",
)


class ConfigError(ValueError):
    """Raised when the configuration is missing or inconsistent."""


def deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict:
    """Return a new dict with ``override`` merged into ``base`` (inputs untouched)."""
    merged = copy.deepcopy(dict(base))
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def validate_config(cfg: Mapping[str, Any]) -> None:
    missing = [s for s in REQUIRED_SECTIONS if s not in cfg]
    if missing:
        raise ConfigError(f"config is missing sections: {missing}")
    pre = cfg["pretrain"]
    if not 0 < pre["warmup_epochs"] < pre["epochs"]:
        raise ConfigError("pretrain.warmup_epochs must be in (0, pretrain.epochs)")
    for name, weights in pre["variants"].items():
        if len(weights) != 3 or sum(weights) <= 0:
            raise ConfigError(f"pretrain.variants.{name} needs three weights, not all zero")
    if not 0 < cfg["eval"]["target_fpr"] < 1:
        raise ConfigError("eval.target_fpr must be in (0, 1)")
    for budget in cfg["matrix"]["budgets"]:
        if not 0 < budget <= 1:
            raise ConfigError(f"budget {budget} must be in (0, 1]")


def load_config(path: str | Path, overrides: Mapping[str, Any] | None = None) -> dict:
    """Load the YAML config, apply ``overrides`` and validate."""
    path = Path(path)
    if not path.is_file():
        raise ConfigError(f"config file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    if not isinstance(cfg, dict):
        raise ConfigError(f"config file {path} did not parse to a mapping")
    if overrides:
        cfg = deep_merge(cfg, overrides)
    validate_config(cfg)
    return cfg


def config_hash(cfg: Mapping[str, Any], sections: tuple[str, ...] | None = None) -> str:
    """Hash of the settings that decide results (never paths or the run list)."""
    names = sections or tuple(k for k in cfg if k not in ("paths", "matrix"))
    payload = {k: cfg[k] for k in sorted(names)}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:12]


def write_text_atomic(path: Path, text: str) -> None:
    """Write through a temporary file so a disconnect never leaves a torn file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def resolve_paths(cfg: Mapping[str, Any]) -> dict[str, Path]:
    """All filesystem locations used by the pipeline, derived from ``cfg['paths']``."""
    root = Path(cfg["paths"]["drive_root"])
    return {
        "raw_dir": Path(cfg["paths"]["raw_dir"]),
        "parquet_dir": Path(cfg["paths"]["parquet_dir"]),
        "drive_root": root,
        "splits": root / "data" / "splits.parquet",
        "manifest": root / "data" / "manifest.json",
        "ingest_stats": root / "data" / "ingest_stats.json",
        "ckpt_dir": root / "ckpt",
        "results_dir": root / "results",
        "runs": root / "results" / "runs.jsonl",
        "errors": root / "results" / "errors.jsonl",
        "scores_dir": root / "scores",
        "report_dir": root / "report",
    }
