"""One function per model family: train at a budget and score the tuning, closed-world and test sets."""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from .baselines import fit_tree, serialized_mb, train_svdd, tree_proba, tree_size
from .dataset import Dataset
from .encoder import ScarfEncoder, build_encoder, count_parameters
from .pretrain import encode, load_or_pretrain, set_seed
from .probe import finetune, predict_proba, train_probe
from .scoring import knn_distance, latent_scores, mahalanobis_distance, msp_score
from .splits import budget_subset

SSL_MODELS = ("S4_E06", "S4_E05", "SCARF")
TREE_MODELS = ("RF", "XGB")


@dataclass(frozen=True)
class Outputs:
    scores: dict[str, tuple[np.ndarray, np.ndarray]]  # scorer -> (tuning scores, test scores)
    cw_pred: np.ndarray | None                         # closed-world predictions, None for one-class
    params: dict
    timing: dict
    n_train: int


@dataclass(frozen=True)
class Context:
    ds: Dataset
    cfg: Mapping[str, Any]
    device: torch.device
    ckpt_dir: Path
    tensors: Mapping[str, torch.Tensor]

    @classmethod
    def create(cls, ds: Dataset, cfg: Mapping[str, Any], device: torch.device, ckpt_dir: Path) -> "Context":
        names = ("x_train", "y_train", "x_tune", "x_cw", "x_test")
        tensors = {n: torch.tensor(getattr(ds, n), device=device) for n in names}
        return cls(ds, cfg, device, ckpt_dir, tensors)


class _Clock:
    """Wall-clock timer that waits for pending GPU work."""

    def __init__(self, device: torch.device) -> None:
        self.device = device
        self.start = self._now()

    def _now(self) -> float:
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        return time.perf_counter()

    def elapsed(self) -> float:
        return self._now() - self.start


def _reset_peak(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()


def _peak_mb(device: torch.device) -> float | None:
    return torch.cuda.max_memory_allocated() / 1e6 if device.type == "cuda" else None


def _load_encoder(kind: str, checkpoint: dict, ctx: Context) -> torch.nn.Module:
    model_cfg, in_dim = ctx.cfg["model"], checkpoint["in_dim"]
    if kind == "SCARF":
        encoder = ScarfEncoder(in_dim, ctx.cfg["scarf"]["hidden"], model_cfg["latent_dim"])
    else:
        encoder = build_encoder(in_dim, model_cfg)
    encoder.load_state_dict(checkpoint["state"]["encoder"])
    return encoder.to(ctx.device).eval()


def _to_np(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().numpy()


def run_ssl(model: str, variant: str, budget: float, seed: int, ctx: Context) -> Outputs:
    """S4_E06 / ablation variants / SCARF use a frozen encoder + probe; S4_E05 fine-tunes end to end."""
    t, cfg, ds = ctx.tensors, ctx.cfg, ctx.ds
    kind = "SCARF" if model == "SCARF" else variant
    checkpoint = load_or_pretrain(kind, t["x_train"], cfg, seed, ctx.ckpt_dir, ds.condition)
    encoder = _load_encoder(kind, checkpoint, ctx)
    idx = torch.as_tensor(budget_subset(ds.y_train, budget, seed), device=ctx.device)
    x_sub, y_sub = t["x_train"][idx], t["y_train"][idx]

    set_seed(seed)
    _reset_peak(ctx.device)
    clock = _Clock(ctx.device)
    if model == "S4_E05":
        encoder, head = finetune(copy.deepcopy(encoder), x_sub, y_sub, ds.n_classes, cfg)
    else:
        z_sub = encode(encoder, x_sub)
        head = train_probe(z_sub, y_sub, ds.n_classes, cfg)
    adapt_seconds = clock.elapsed()

    centroid = checkpoint["centroid"].numpy()
    batch = cfg["eval"]["batch_size"]
    clock = _Clock(ctx.device)
    z_test = encode(encoder, t["x_test"], batch)
    proba_test = predict_proba(head, z_test, batch)
    distance_test = ((z_test - checkpoint["centroid"].to(ctx.device)) ** 2).sum(dim=1)
    infer_seconds = clock.elapsed()
    del distance_test
    z_tune, z_cw = encode(encoder, t["x_tune"], batch), encode(encoder, t["x_cw"], batch)

    scores = {"msp": (msp_score(_to_np(predict_proba(head, z_tune, batch))), msp_score(_to_np(proba_test)))}
    if model == "S4_E06":  # frozen encoder: latent-space scores are defined
        ref_z, ref_y = _to_np(z_sub), ds.y_train[_to_np(idx)]
        on_tune = latent_scores(_to_np(z_tune), centroid, ref_z, ref_y, ds.benign_index, seed)
        on_test = latent_scores(_to_np(z_test), centroid, ref_z, ref_y, ds.benign_index, seed)
        scores.update({name: (on_tune[name], on_test[name]) for name in on_tune})
        # No-encoder references: the same reference flows, scored in the scaled input space.
        raw_benign = ds.x_train[_to_np(idx)][ref_y == ds.benign_index]
        scores["raw_knn_benign"] = (knn_distance(ds.x_tune, raw_benign, seed=seed),
                                    knn_distance(ds.x_test, raw_benign, seed=seed))
        scores["raw_maha_benign"] = (mahalanobis_distance(ds.x_tune, raw_benign),
                                     mahalanobis_distance(ds.x_test, raw_benign))
    deployed = {"encoder": encoder.state_dict(), "head": head.state_dict()}
    return Outputs(
        scores=scores,
        cw_pred=_to_np(predict_proba(head, z_cw, batch).argmax(dim=1)),
        params={**{f"pretrain_{k}": v for k, v in checkpoint["params"].items()},
                "probe": count_parameters(head),
                "deployed": count_parameters(encoder) + count_parameters(head),
                "trained_total": sum(checkpoint["params"].values()) + count_parameters(head)},
        timing={"pretrain_seconds": checkpoint["seconds"], "adapt_seconds": adapt_seconds,
                "infer_ms_per_flow": 1000 * infer_seconds / len(ds.x_test),
                "peak_memory_mb": _peak_mb(ctx.device), "checkpoint_mb": serialized_mb(deployed)},
        n_train=int(len(idx)),
    )


def run_svdd(budget: float, seed: int, ctx: Context) -> Outputs:
    t, ds = ctx.tensors, ctx.ds
    idx = budget_subset(ds.y_train, budget, seed)
    benign = torch.as_tensor(idx[ds.y_train[idx] == ds.benign_index], device=ctx.device)
    set_seed(seed)
    _reset_peak(ctx.device)
    clock = _Clock(ctx.device)
    net, center = train_svdd(t["x_train"][benign], ctx.cfg)
    adapt_seconds = clock.elapsed()

    def score(x: torch.Tensor) -> torch.Tensor:
        return ((encode(net, x, ctx.cfg["eval"]["batch_size"]) - center) ** 2).sum(dim=1)

    clock = _Clock(ctx.device)
    test_scores = score(t["x_test"])
    infer_seconds = clock.elapsed()
    return Outputs(
        scores={"svdd": (_to_np(score(t["x_tune"])).astype("float64"), _to_np(test_scores).astype("float64"))},
        cw_pred=None,
        params={"deployed": count_parameters(net)},
        timing={"adapt_seconds": adapt_seconds, "infer_ms_per_flow": 1000 * infer_seconds / len(ds.x_test),
                "peak_memory_mb": _peak_mb(ctx.device),
                "checkpoint_mb": serialized_mb({"net": net.state_dict(), "center": center})},
        n_train=int(len(benign)),
    )


def run_tree(model: str, budget: float, seed: int, ctx: Context) -> Outputs:
    ds = ctx.ds
    idx = budget_subset(ds.y_train, budget, seed)
    started = time.perf_counter()
    fitted, present = fit_tree(model, ds.x_train[idx], ds.y_train[idx], ctx.cfg, seed)
    adapt_seconds = time.perf_counter() - started
    started = time.perf_counter()
    proba_test = tree_proba(fitted, present, ds.x_test, ds.n_classes)
    infer_seconds = time.perf_counter() - started
    proba_tune = tree_proba(fitted, present, ds.x_tune, ds.n_classes)
    return Outputs(
        scores={"msp": (msp_score(proba_tune), msp_score(proba_test))},
        cw_pred=tree_proba(fitted, present, ds.x_cw, ds.n_classes).argmax(axis=1),
        params={"nodes" if model == "RF" else "trees": tree_size(model, fitted)},
        timing={"adapt_seconds": adapt_seconds, "infer_ms_per_flow": 1000 * infer_seconds / len(ds.x_test),
                "peak_memory_mb": None, "checkpoint_mb": serialized_mb(fitted)},
        n_train=int(len(idx)),
    )


def run_model(model: str, variant: str, budget: float, seed: int, ctx: Context) -> Outputs:
    if model in SSL_MODELS:
        return run_ssl(model, variant, budget, seed, ctx)
    if model == "DeepSVDD":
        return run_svdd(budget, seed, ctx)
    if model in TREE_MODELS:
        return run_tree(model, budget, seed, ctx)
    raise ValueError(f"unknown model {model!r}")
