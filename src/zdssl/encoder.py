"""Network definitions.

With 68 inputs the default sizes give exactly the component counts printed in the
paper: encoder 316,424; reconstruction head 102,024; projection head 49,984;
probe 9,294. Counts are always measured and reported, never assumed.
"""

from __future__ import annotations

from typing import Sequence

import torch
from torch import nn


def count_parameters(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


class ResidualBlock(nn.Module):
    def __init__(self, width: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(width, width), nn.LayerNorm(width), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(width, width), nn.LayerNorm(width),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class FoundationEncoder(nn.Module):
    """in_dim -> width (LayerNorm) -> residual blocks -> latent_dim."""

    def __init__(self, in_dim: int, width: int = 256, latent_dim: int = 128,
                 n_blocks: int = 2, dropout: float = 0.1, input_norm: bool = True) -> None:
        super().__init__()
        # The input LayerNorm normalises across a flow's features, so it discards a
        # shift common to all features; it is what the paper's parameter count implies.
        first = [nn.LayerNorm(in_dim)] if input_norm else []
        self.stem = nn.Sequential(*first, nn.Linear(in_dim, width), nn.LayerNorm(width), nn.GELU())
        self.blocks = nn.Sequential(*[ResidualBlock(width, dropout) for _ in range(n_blocks)])
        self.out = nn.Linear(width, latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.out(self.blocks(self.stem(x)))


class ReconstructionHead(nn.Module):
    """Predicts the clean features and the corruption-mask logits from a latent."""

    def __init__(self, latent_dim: int, hidden: int, out_dim: int) -> None:
        super().__init__()
        self.features = self._branch(latent_dim, hidden, out_dim)
        self.mask = self._branch(latent_dim, hidden, out_dim)

    @staticmethod
    def _branch(latent_dim: int, hidden: int, out_dim: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Linear(latent_dim, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Linear(hidden, out_dim)
        )

    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.features(z), self.mask(z)


class ProjectionHead(nn.Module):
    def __init__(self, latent_dim: int, hidden: int, out_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Linear(hidden, out_dim)
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class Probe(nn.Module):
    """Lightweight classifier head: latent_dim -> hidden (LayerNorm, ReLU) -> n_classes."""

    def __init__(self, latent_dim: int, hidden: int, n_classes: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden), nn.LayerNorm(hidden), nn.ReLU(), nn.Linear(hidden, n_classes)
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class ScarfEncoder(nn.Module):
    """Plain ReLU MLP encoder, as in SCARF."""

    def __init__(self, in_dim: int, hidden: int, latent_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, latent_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SvddNet(nn.Module):
    """Bias-free MLP, as Deep SVDD requires (no bias terms, no affine normalisation)."""

    def __init__(self, in_dim: int, hidden: Sequence[int], out_dim: int) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        previous = in_dim
        for width in hidden:
            layers += [nn.Linear(previous, width, bias=False), nn.LeakyReLU(0.1)]
            previous = width
        layers.append(nn.Linear(previous, out_dim, bias=False))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def build_encoder(in_dim: int, model_cfg: dict) -> FoundationEncoder:
    return FoundationEncoder(in_dim, model_cfg["width"], model_cfg["latent_dim"], model_cfg["n_blocks"],
                             model_cfg["dropout"], model_cfg["input_norm"])


def build_multitask(in_dim: int, model_cfg: dict) -> dict[str, nn.Module]:
    """Encoder plus the two pretraining heads."""
    latent = model_cfg["latent_dim"]
    return {
        "encoder": build_encoder(in_dim, model_cfg),
        "recon": ReconstructionHead(latent, model_cfg["head_hidden"], in_dim),
        "proj": ProjectionHead(latent, model_cfg["head_hidden"], model_cfg["proj_dim"]),
    }
