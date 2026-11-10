"""View corruption and the self-supervised losses."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def sample_marginals(pool: torch.Tensor, shape: tuple[int, int]) -> torch.Tensor:
    """Draw each coordinate independently from that feature's empirical marginal in ``pool``."""
    rows = torch.randint(0, pool.shape[0], shape, device=pool.device)
    return torch.gather(pool, 0, rows)


def corrupt_view(x: torch.Tensor, pool: torch.Tensor, p_mask: float, sigma: float
                 ) -> tuple[torch.Tensor, torch.Tensor]:
    """Paper Sec. 3.2: masked coordinates get a marginal draw, the others Gaussian jitter.

    Returns the corrupted view and the binary mask (1 = replaced).
    """
    mask = torch.rand_like(x) < p_mask
    jittered = x + sigma * torch.randn_like(x)
    return torch.where(mask, sample_marginals(pool, x.shape), jittered), mask.float()


def nt_xent(p1: torch.Tensor, p2: torch.Tensor, temperature: float) -> torch.Tensor:
    """Paper Eq. 1: InfoNCE over the 2N projections, self-similarity excluded."""
    n = p1.shape[0]
    z = F.normalize(torch.cat([p1, p2]), dim=1)
    sim = (z @ z.T) / temperature
    sim = sim.masked_fill(torch.eye(2 * n, dtype=torch.bool, device=z.device), float("-inf"))
    targets = torch.cat([torch.arange(n, 2 * n), torch.arange(0, n)]).to(z.device)
    return F.cross_entropy(sim, targets)


def scarf_info_nce(anchor: torch.Tensor, positive: torch.Tensor, temperature: float) -> torch.Tensor:
    """SCARF loss: each clean anchor must pick its own corrupted view among the batch."""
    logits = (F.normalize(anchor, dim=1) @ F.normalize(positive, dim=1).T) / temperature
    return F.cross_entropy(logits, torch.arange(anchor.shape[0], device=anchor.device))


def reconstruction_loss(x_hat: torch.Tensor, mask_logits: torch.Tensor, x: torch.Tensor,
                        mask: torch.Tensor, gamma: float) -> torch.Tensor:
    """Paper Eq. 2: feature MSE plus gamma times mask BCE, both averaged over coordinates."""
    return F.mse_loss(x_hat, x) + gamma * F.binary_cross_entropy_with_logits(mask_logits, mask)


def novelty_loss(z: torch.Tensor, centroid: torch.Tensor) -> torch.Tensor:
    """Mean squared Euclidean distance to the fixed centroid."""
    return ((z - centroid) ** 2).sum(dim=1).mean()
