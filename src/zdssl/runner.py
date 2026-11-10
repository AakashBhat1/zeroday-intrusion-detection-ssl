"""Resumable run matrix. One JSON line per finished run in ``results/runs.jsonl``."""

from __future__ import annotations

import json
import logging
import os
import platform
import time
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from . import __version__
from .config import config_hash, resolve_paths
from .dataset import Dataset, load_dataset
from .experiments import Context, Outputs, run_model
from .metrics import closed_world_metrics, evaluate_zero_day

log = logging.getLogger(__name__)

STAGES = ("pilot", "p1", "p2", "ablation", "condA", "all")
DEFAULT_VARIANT = {"S4_E06": "F", "S4_E05": "C"}


@dataclass(frozen=True)
class RunSpec:
    model: str
    variant: str
    condition: str
    budget: float
    seed: int

    @property
    def key(self) -> str:
        return f"{self.model}|{self.variant}|{self.condition}|{self.budget:g}|{self.seed}"


def _spec(model: str, budget: float, seed: int, variant: str | None = None, condition: str = "B") -> RunSpec:
    return RunSpec(model, variant or DEFAULT_VARIANT.get(model, "-"), condition, float(budget), int(seed))


def build_matrix(cfg: Mapping[str, Any], stage: str) -> list[RunSpec]:
    """Run list for a stage, ordered seed-major so pretraining checkpoints are reused immediately."""
    if stage not in STAGES:
        raise ValueError(f"stage must be one of {STAGES}, got {stage!r}")
    m = cfg["matrix"]
    primary, models, seeds = m["primary_budget"], m["models"], m["seeds"]
    stages = {
        "pilot": [_spec(model, primary, m["pilot_seed"]) for model in models],
        "p1": [_spec(model, primary, seed) for seed in seeds for model in models],
        "p2": [_spec(model, budget, seed) for seed in seeds
               for budget in m["budgets"] if budget != primary for model in models],
        "ablation": [_spec("S4_E06", primary, seed, variant=v)
                     for seed in seeds for v in m["ablation_variants"]],
        "condA": [_spec("RF", primary, seed, condition="A") for seed in seeds],
    }
    if stage == "all":
        return [spec for name in ("p1", "p2", "ablation", "condA") for spec in stages[name]]
    return stages[stage]


def completed_keys(runs_path: Path, only_hash: str | None = None) -> set[str]:
    """Keys already in ``runs.jsonl`` (made with config ``only_hash`` if given). Torn lines are ignored."""
    if not runs_path.is_file():
        return set()
    keys = set()
    with runs_path.open("r", encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                if only_hash is None or record["config_hash"] == only_hash:
                    keys.add(record["key"])
            except (json.JSONDecodeError, KeyError):
                log.warning("ignoring unreadable line %d of %s", number, runs_path)
    return keys


def _append_jsonl(path: Path, record: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    needs_newline = False  # a crash can leave a torn last line; never glue a record onto it
    if path.is_file() and path.stat().st_size > 0:
        with path.open("rb") as tail:
            tail.seek(-1, os.SEEK_END)
            needs_newline = tail.read(1) != b"\n"
    with path.open("a", encoding="utf-8") as handle:
        handle.write(("\n" if needs_newline else "") + json.dumps(record) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _environment(device: torch.device) -> dict:
    return {
        "device": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
        "cpu_count": os.cpu_count(),
        "torch": torch.__version__,
        "python": platform.python_version(),
        "zdssl": __version__,
    }


def build_record(spec: RunSpec, out: Outputs, ds: Dataset, cfg: Mapping[str, Any],
                 device: torch.device, wall_seconds: float) -> dict:
    tune_is_benign = ds.y_tune == ds.benign_index
    return {
        "key": spec.key,
        **asdict(spec),
        "n_train": out.n_train,
        "n_features": len(ds.feature_names),
        "zero_day": {
            scorer: evaluate_zero_day(test, ds.test_is_zero_day, tune, tune_is_benign, cfg["eval"])
            for scorer, (tune, test) in out.scores.items()
        },
        "closed_world": (closed_world_metrics(ds.y_cw, out.cw_pred, ds.n_classes)
                         if out.cw_pred is not None else None),
        "params": out.params,
        "timing": {**out.timing, "wall_seconds": wall_seconds},
        "env": _environment(device),
        "config_hash": config_hash(cfg),
        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }


def scores_path(scores_dir: Path, key: str) -> Path:
    return scores_dir / (key.replace("|", "__") + ".npz")


def _save_scores(scores_dir: Path, spec: RunSpec, out: Outputs) -> None:
    scores_dir.mkdir(parents=True, exist_ok=True)
    arrays = {}
    for scorer, (tune, test) in out.scores.items():
        arrays[f"{scorer}_tune"] = tune.astype("float64")
        arrays[f"{scorer}_test"] = test.astype("float64")
    path = scores_path(scores_dir, spec.key)
    tmp = path.with_name(path.stem + ".tmp.npz")
    np.savez_compressed(tmp, **arrays)
    tmp.replace(path)


def run_stage(cfg: Mapping[str, Any], stage: str, device: str | torch.device | None = None) -> dict:
    """Run every unfinished spec of ``stage``. Failures are logged to ``errors.jsonl`` and reported."""
    paths = resolve_paths(cfg)
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    specs = build_matrix(cfg, stage)
    done = completed_keys(paths["runs"], config_hash(cfg))
    pending = [s for s in specs if s.key not in done]
    log.info("stage %s: %d runs, %d already done, %d to run on %s",
             stage, len(specs), len(specs) - len(pending), len(pending), device)
    contexts: dict[str, Context] = {}
    finished, failed = [], []
    for position, spec in enumerate(pending, start=1):
        try:
            if spec.condition not in contexts:
                contexts[spec.condition] = Context.create(
                    load_dataset(cfg, spec.condition), cfg, device, paths["ckpt_dir"]
                )
            ctx = contexts[spec.condition]
            started = time.perf_counter()
            out = run_model(spec.model, spec.variant, spec.budget, spec.seed, ctx)
            record = build_record(spec, out, ctx.ds, cfg, device, time.perf_counter() - started)
            if cfg["eval"]["save_scores"]:
                _save_scores(paths["scores_dir"], spec, out)
            _append_jsonl(paths["runs"], record)
            finished.append(spec.key)
            first = next(iter(record["zero_day"].values()))
            log.info("[%d/%d] %s  AUROC=%.4f  %.1fs", position, len(pending), spec.key,
                     first["auroc"], record["timing"]["wall_seconds"])
        except Exception as error:  # keep the matrix going; the failure is recorded and returned
            failed.append(spec.key)
            log.error("[%d/%d] %s FAILED: %s", position, len(pending), spec.key, error)
            _append_jsonl(paths["errors"], {"key": spec.key, "error": repr(error),
                                            "traceback": traceback.format_exc(),
                                            "at": time.strftime("%Y-%m-%dT%H:%M:%S%z")})
    return {"stage": stage, "total": len(specs), "skipped": len(specs) - len(pending),
            "finished": finished, "failed": failed}
