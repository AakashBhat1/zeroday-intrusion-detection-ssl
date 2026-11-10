"""Downstream adaptation: frozen-encoder probe and end-to-end fine-tuning."""

from __future__ import annotations

from typing import Any, Mapping

import torch
import torch.nn.functional as F
from torch import nn

from .encoder import Probe
from .pretrain import iterate_batches


def class_weights(y: torch.Tensor, n_classes: int, power: float) -> torch.Tensor:
    """w_c proportional to (N / N_c) ** power, mean 1 over the classes present; 0 for absent classes."""
    counts = torch.bincount(y, minlength=n_classes).float()
    present = counts > 0
    weights = torch.zeros(n_classes, device=y.device)
    weights[present] = (counts.sum() / counts[present]) ** power
    return weights / weights[present].mean()


def train_probe(z_train: torch.Tensor, y_train: torch.Tensor, n_classes: int,
                cfg: Mapping[str, Any]) -> Probe:
    """Train the probe on fixed latents (the encoder is frozen by construction)."""
    pc, device = cfg["probe"], z_train.device
    probe = Probe(z_train.shape[1], cfg["model"]["probe_hidden"], n_classes).to(device)
    optimizer = torch.optim.AdamW(probe.parameters(), lr=pc["lr"], weight_decay=pc["weight_decay"])
    weights = class_weights(y_train, n_classes, pc["class_weight_power"])
    for _ in range(pc["epochs"]):
        probe.train()
        for batch in iterate_batches(len(z_train), pc["batch_size"], device):
            loss = F.cross_entropy(probe(z_train[batch]), y_train[batch], weight=weights)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
    return probe.eval()


def finetune(encoder: nn.Module, x_train: torch.Tensor, y_train: torch.Tensor, n_classes: int,
             cfg: Mapping[str, Any]) -> tuple[nn.Module, Probe]:
    """End-to-end adaptation with separate learning rates for encoder and head."""
    fc, device = cfg["finetune"], x_train.device
    head = Probe(cfg["model"]["latent_dim"], cfg["model"]["probe_hidden"], n_classes).to(device)
    optimizer = torch.optim.AdamW(
        [{"params": encoder.parameters(), "lr": fc["lr_encoder"]},
         {"params": head.parameters(), "lr": fc["lr_head"]}],
        weight_decay=fc["weight_decay"],
    )
    weights = class_weights(y_train, n_classes, cfg["probe"]["class_weight_power"])
    for _ in range(fc["epochs"]):
        encoder.train()
        head.train()
        for batch in iterate_batches(len(x_train), fc["batch_size"], device):
            loss = F.cross_entropy(head(encoder(x_train[batch])), y_train[batch], weight=weights)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
    return encoder.eval(), head.eval()


@torch.no_grad()
def predict_proba(head: nn.Module, z: torch.Tensor, batch_size: int = 8192) -> torch.Tensor:
    head.eval()
    return torch.cat([F.softmax(head(z[i:i + batch_size]), dim=1) for i in range(0, len(z), batch_size)])
