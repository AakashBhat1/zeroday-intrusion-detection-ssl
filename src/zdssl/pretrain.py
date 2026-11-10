"""Self-supervised pretraining (multi-task S4 variants and SCARF) with Drive checkpoints.

The pretraining pool is the training split with labels hidden. Callers pass that
tensor only; validation and test flows never reach this module.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from torch import nn

from .config import config_hash
from .encoder import ProjectionHead, ScarfEncoder, build_multitask, count_parameters
from .losses import (corrupt_view, novelty_loss, nt_xent, reconstruction_loss,
                     sample_marginals, scarf_info_nce)

log = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def iterate_batches(n: int, batch_size: int, device: torch.device, min_size: int = 1):
    """Yield shuffled index batches for one epoch."""
    order = torch.randperm(n, device=device)
    for start in range(0, n, batch_size):
        batch = order[start:start + batch_size]
        if len(batch) >= min_size:
            yield batch


@torch.no_grad()
def encode(encoder: nn.Module, x: torch.Tensor, batch_size: int = 8192) -> torch.Tensor:
    """Latents for ``x`` with the encoder in eval mode."""
    encoder.eval()
    return torch.cat([encoder(x[i:i + batch_size]) for i in range(0, len(x), batch_size)])


def pretrain_multitask(x_pool: torch.Tensor, weights: tuple[float, float, float],
                       cfg: Mapping[str, Any], seed: int) -> dict:
    """Train encoder + heads with ``weights`` = (lambda_rec, lambda_con, lambda_nov).

    The centroid is the mean latent of the whole pool. With a novelty weight it is
    fixed after the warm-up epochs and the novelty loss is active from the next
    epoch; without one it is computed after the last epoch (used for scoring only).
    """
    pre, device = cfg["pretrain"], x_pool.device
    lam_rec, lam_con, lam_nov = weights
    if len(x_pool) < pre["centroid_min_flows"]:
        raise ValueError(f"SSL pool has {len(x_pool)} flows, fewer than centroid_min_flows")
    set_seed(seed)
    modules = {k: m.to(device) for k, m in build_multitask(x_pool.shape[1], cfg["model"]).items()}
    params = [p for m in modules.values() for p in m.parameters()]
    optimizer = torch.optim.AdamW(params, lr=pre["lr"], weight_decay=pre["weight_decay"])
    encoder, recon, proj = modules["encoder"], modules["recon"], modules["proj"]
    centroid: torch.Tensor | None = None
    history = []
    started = time.perf_counter()
    for epoch in range(1, pre["epochs"] + 1):
        for module in modules.values():
            module.train()
        use_nov = lam_nov > 0 and centroid is not None
        sums = {"rec": 0.0, "con": 0.0, "nov": 0.0, "total": 0.0}
        steps = 0
        for batch in iterate_batches(len(x_pool), pre["batch_size"], device, min_size=2):
            x = x_pool[batch]
            views = [corrupt_view(x, x_pool, pre["p_mask"], pre["jitter_sigma"]) for _ in range(2)]
            latents = [encoder(view) for view, _ in views]
            zero = x.new_zeros(())
            loss_rec = loss_con = loss_nov = zero
            if lam_rec > 0:
                loss_rec = sum(
                    reconstruction_loss(*recon(z), x, mask, pre["gamma_mask"])
                    for z, (_, mask) in zip(latents, views)
                ) / 2
            if lam_con > 0:
                loss_con = nt_xent(proj(latents[0]), proj(latents[1]), pre["temperature"])
            if use_nov:
                loss_nov = sum(novelty_loss(z, centroid) for z in latents) / 2
            loss = lam_rec * loss_rec + lam_con * loss_con + (lam_nov * loss_nov if use_nov else zero)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            steps += 1
            for name, value in (("rec", loss_rec), ("con", loss_con), ("nov", loss_nov), ("total", loss)):
                sums[name] += float(value.detach())
        if steps == 0:
            raise ValueError("SSL pool is smaller than two flows")
        history.append({"epoch": epoch, "novelty_active": use_nov,
                        **{k: v / steps for k, v in sums.items()}})
        if lam_nov > 0 and epoch == pre["warmup_epochs"]:
            centroid = encode(encoder, x_pool).mean(dim=0)
    if centroid is None:
        centroid = encode(encoder, x_pool).mean(dim=0)
    return {
        "state": {k: m.state_dict() for k, m in modules.items()},
        "centroid": centroid.detach().cpu(),
        "history": history,
        "seconds": time.perf_counter() - started,
        "params": {k: count_parameters(m) for k, m in modules.items()},
        "in_dim": x_pool.shape[1],
    }


def pretrain_scarf(x_pool: torch.Tensor, cfg: Mapping[str, Any], seed: int) -> dict:
    """SCARF: InfoNCE between a clean flow and a copy with random features resampled from marginals."""
    sc, device = cfg["scarf"], x_pool.device
    latent = cfg["model"]["latent_dim"]
    set_seed(seed)
    encoder = ScarfEncoder(x_pool.shape[1], sc["hidden"], latent).to(device)
    head = ProjectionHead(latent, sc["hidden"], cfg["model"]["proj_dim"]).to(device)
    optimizer = torch.optim.Adam([*encoder.parameters(), *head.parameters()],
                                 lr=sc["lr"], weight_decay=sc["weight_decay"])
    history = []
    started = time.perf_counter()
    for epoch in range(1, sc["epochs"] + 1):
        encoder.train()
        head.train()
        total, steps = 0.0, 0
        for batch in iterate_batches(len(x_pool), sc["batch_size"], device, min_size=2):
            x = x_pool[batch]
            mask = torch.rand_like(x) < sc["corruption_rate"]
            corrupted = torch.where(mask, sample_marginals(x_pool, x.shape), x)
            loss = scarf_info_nce(head(encoder(x)), head(encoder(corrupted)), sc["temperature"])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total, steps = total + float(loss.detach()), steps + 1
        if steps == 0:
            raise ValueError("SSL pool is smaller than two flows")
        history.append({"epoch": epoch, "total": total / steps})
    return {
        "state": {"encoder": encoder.state_dict(), "proj": head.state_dict()},
        "centroid": encode(encoder, x_pool).mean(dim=0).cpu(),
        "history": history,
        "seconds": time.perf_counter() - started,
        "params": {"encoder": count_parameters(encoder), "proj": count_parameters(head)},
        "in_dim": x_pool.shape[1],
    }


def checkpoint_path(ckpt_dir: Path, kind: str, condition: str, seed: int) -> Path:
    return ckpt_dir / f"{kind}_cond{condition}_seed{seed}.pt"


def load_or_pretrain(kind: str, x_pool: torch.Tensor, cfg: Mapping[str, Any], seed: int,
                     ckpt_dir: Path, condition: str) -> dict:
    """Return the checkpoint for ``kind`` (``SCARF`` or an S4 variant letter), training it if absent."""
    path = checkpoint_path(ckpt_dir, kind, condition, seed)
    settings = config_hash(cfg, ("model", "pretrain", "scarf", "features", "splits"))
    if path.is_file():
        try:
            checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        except Exception as error:  # half-synced or truncated file: train it again
            log.warning("checkpoint %s is unreadable (%s); retraining", path, error)
            checkpoint = None
        if checkpoint is not None:
            if checkpoint["in_dim"] != x_pool.shape[1] or checkpoint["pool_size"] != len(x_pool):
                raise ValueError(f"checkpoint {path} was trained on different data; delete it to retrain")
            if checkpoint.get("settings") != settings:
                raise ValueError(f"checkpoint {path} was trained with different settings; delete it to retrain")
            return checkpoint
    if kind == "SCARF":
        checkpoint = pretrain_scarf(x_pool, cfg, seed)
    else:
        weights = tuple(cfg["pretrain"]["variants"][kind])
        checkpoint = pretrain_multitask(x_pool, weights, cfg, seed)
    checkpoint = {**checkpoint, "kind": kind, "seed": seed, "pool_size": len(x_pool), "settings": settings}
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    torch.save(checkpoint, tmp)
    tmp.replace(path)
    log.info("pretrained %s seed %d in %.1fs", kind, seed, checkpoint["seconds"])
    return checkpoint
