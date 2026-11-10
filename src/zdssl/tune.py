"""Hyperparameter selection on stand-in zero-days. The locked test day is never read here.

For each candidate configuration, one catalogued attack class at a time is removed
from the training split entirely (pretraining pool and labels). The model is then
scored on separating tuning-split benign flows from that unseen class. The mean
AUROC over held-out classes and seeds is the selection criterion.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

from .config import config_hash, deep_merge, resolve_paths, validate_config
from .dataset import Dataset, load_dataset
from .encoder import build_encoder
from .pretrain import encode, pretrain_multitask, set_seed
from .probe import predict_proba, train_probe
from .runner import _append_jsonl
from .scoring import knn_distance, latent_scores, mahalanobis_distance, msp_score
from .splits import budget_subset

log = logging.getLogger(__name__)

# Candidate settings: the paper's configuration plus one-factor changes around it.
CANDIDATES: dict[str, dict] = {
    "base": {},
    "epochs30": {"pretrain": {"epochs": 30, "warmup_epochs": 15}},
    "no_novelty": {"pretrain": {"variants": {"F": [0.5, 0.5, 0.0]}}},
    "novelty1.0": {"pretrain": {"variants": {"F": [0.5, 0.5, 1.0]}}},
    "mask0.4": {"pretrain": {"p_mask": 0.4}},
    "temp0.5": {"pretrain": {"temperature": 0.5}},
    "no_input_norm": {"model": {"input_norm": False}},
}


def dev_split(ds: Dataset, held_out: int) -> dict[str, np.ndarray]:
    """Training data without the held-out class, and a dev set of tuning benign vs that class."""
    if held_out == ds.benign_index:
        raise ValueError("the benign class cannot be the stand-in zero-day")
    keep = ds.y_train != held_out
    benign = ds.x_tune[ds.y_tune == ds.benign_index]
    novel = np.concatenate([ds.x_train[~keep], ds.x_tune[ds.y_tune == held_out]])
    if len(novel) == 0:
        raise ValueError(f"class {held_out} has no flows to hold out")
    return {
        "x_train": ds.x_train[keep], "y_train": ds.y_train[keep],
        "x_dev": np.concatenate([benign, novel]),
        "is_novel": np.concatenate([np.zeros(len(benign), dtype="int64"), np.ones(len(novel), dtype="int64")]),
    }


def dev_aurocs(ds: Dataset, cfg: Mapping[str, Any], held_out: int, seed: int, device: torch.device) -> dict:
    """Dev AUROC of every scorer for one held-out class and seed."""
    split = dev_split(ds, held_out)
    x_pool = torch.tensor(split["x_train"], device=device)
    y_pool = torch.tensor(split["y_train"], device=device)
    x_dev = torch.tensor(split["x_dev"], device=device)
    weights = tuple(cfg["pretrain"]["variants"]["F"])
    trained = pretrain_multitask(x_pool, weights, cfg, seed)
    encoder = build_encoder(x_pool.shape[1], cfg["model"])
    encoder.load_state_dict(trained["state"]["encoder"])
    encoder = encoder.to(device).eval()

    idx_np = budget_subset(split["y_train"], cfg["matrix"]["primary_budget"], seed)
    idx = torch.as_tensor(idx_np, device=device)
    set_seed(seed)
    z_ref = encode(encoder, x_pool[idx])
    probe = train_probe(z_ref, y_pool[idx], ds.n_classes, cfg)
    z_dev = encode(encoder, x_dev)
    ref_y = split["y_train"][idx_np]
    scores = latent_scores(z_dev.cpu().numpy(), trained["centroid"].numpy(), z_ref.cpu().numpy(),
                           ref_y, ds.benign_index, seed)
    scores["msp"] = msp_score(predict_proba(probe, z_dev).cpu().numpy())
    # No-encoder references: the same scorers on the scaled input features.
    raw_benign = split["x_train"][idx_np][ref_y == ds.benign_index]
    scores["raw_knn_benign"] = knn_distance(split["x_dev"], raw_benign, seed=seed)
    scores["raw_maha_benign"] = mahalanobis_distance(split["x_dev"], raw_benign)
    final = trained["history"][-1]
    return {
        "auroc": {name: float(roc_auc_score(split["is_novel"], value)) for name, value in scores.items()},
        "final_loss": {k: final[k] for k in ("rec", "con", "nov", "total")},
        "loss_by_epoch": [round(h["total"], 5) for h in trained["history"]],
        "pretrain_seconds": trained["seconds"],
        "n_dev_novel": int(split["is_novel"].sum()),
    }


def tune_path(cfg: Mapping[str, Any]) -> Path:
    return resolve_paths(cfg)["results_dir"] / "tune.jsonl"


def run_tuning(cfg: Mapping[str, Any], held_out_classes: list[str], seeds: list[int],
               candidates: Mapping[str, dict] | None = None, device: str | None = None) -> pd.DataFrame:
    """Evaluate every candidate; resumable through ``results/tune.jsonl``. Returns the summary table."""
    candidates = dict(candidates or CANDIDATES)
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    ds = load_dataset(cfg, "B")
    unknown = [c for c in held_out_classes if c not in ds.classes]
    if unknown:
        raise ValueError(f"not catalogued classes: {unknown}; choose from {list(ds.classes)}")
    path = tune_path(cfg)
    done = set()
    if path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                done.add(json.loads(line)["key"])
            except (json.JSONDecodeError, KeyError):
                continue
    for name, override in candidates.items():
        candidate_cfg = deep_merge(cfg, override)
        validate_config(candidate_cfg)
        for held_name in held_out_classes:
            for seed in seeds:
                key = f"{name}|{config_hash(candidate_cfg)}|{held_name}|{seed}"
                if key in done:
                    continue
                started = time.perf_counter()
                result = dev_aurocs(ds, candidate_cfg, ds.classes.index(held_name), seed, dev)
                _append_jsonl(path, {"key": key, "candidate": name, "override": override,
                                     "held_out": held_name, "seed": seed, **result})
                best = max(result["auroc"], key=result["auroc"].get)
                log.info("tune %s held-out=%s seed=%d: best %s=%.4f (decoupled=%.4f) %.0fs", name, held_name,
                         seed, best, result["auroc"][best], result["auroc"]["decoupled"],
                         time.perf_counter() - started)
    return summarize_tuning(path)


def summarize_tuning(path: Path) -> pd.DataFrame:
    """Mean dev AUROC per candidate (rows) and scorer (columns), plus the per-class minimum of the best scorer."""
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        for scorer, value in record["auroc"].items():
            rows.append({"candidate": record["candidate"], "held_out": record["held_out"],
                         "seed": record["seed"], "scorer": scorer, "auroc": value})
    frame = pd.DataFrame(rows)
    return frame.pivot_table(index="candidate", columns="scorer", values="auroc", aggfunc="mean")


def per_class_table(path: Path) -> pd.DataFrame:
    """Mean dev AUROC per (candidate, scorer) broken down by held-out class."""
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        for scorer, value in record["auroc"].items():
            rows.append({"candidate": record["candidate"], "held_out": record["held_out"],
                         "scorer": scorer, "auroc": value})
    frame = pd.DataFrame(rows)
    return frame.pivot_table(index=["candidate", "scorer"], columns="held_out", values="auroc", aggfunc="mean")
