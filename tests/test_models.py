from __future__ import annotations

import numpy as np
import pytest
import torch

from zdssl.baselines import fit_tree, train_svdd, tree_proba
from zdssl.config import load_config
from zdssl.encoder import Probe, SvddNet, build_multitask, count_parameters
from zdssl.losses import corrupt_view, novelty_loss, nt_xent, reconstruction_loss, scarf_info_nce
from zdssl.pretrain import load_or_pretrain, pretrain_multitask, set_seed
from zdssl.probe import class_weights, train_probe

from conftest import CONFIG


def test_parameter_counts_match_the_paper_components():
    cfg = load_config(CONFIG)
    modules = build_multitask(68, cfg["model"])
    assert count_parameters(modules["encoder"]) == 316_424
    assert count_parameters(modules["recon"]) == 102_024
    assert count_parameters(modules["proj"]) == 49_984
    assert count_parameters(Probe(128, 64, 14)) == 9_294
    assert all(p.ndim == 2 for p in SvddNet(68, [256, 128], 64).parameters())  # weights only, no biases


def test_corrupt_view_mask_rate_and_marginals():
    set_seed(0)
    pool = torch.arange(50, dtype=torch.float32).repeat(4, 1).T.contiguous() + 1000  # values >= 1000
    x = torch.zeros(4000, 4)
    view, mask = corrupt_view(x, pool, p_mask=0.2, sigma=0.02)
    assert mask.mean().item() == pytest.approx(0.2, abs=0.02)
    assert (view[mask.bool()] >= 1000).all()          # masked coordinates come from the pool
    assert view[~mask.bool()].abs().max() < 0.2       # the rest is the input plus small jitter
    assert view[~mask.bool()].std().item() == pytest.approx(0.02, rel=0.1)


def test_losses_behave():
    set_seed(0)
    a = torch.randn(64, 16)
    assert nt_xent(a, a, 0.1) < nt_xent(a, torch.randn(64, 16), 0.1)
    assert scarf_info_nce(a, a, 1.0) < scarf_info_nce(a, torch.randn(64, 16), 1.0)
    assert novelty_loss(torch.ones(3, 2), torch.zeros(2)).item() == pytest.approx(2.0)
    x, mask = torch.randn(8, 5), torch.ones(8, 5)
    perfect = reconstruction_loss(x, 20 * torch.ones(8, 5), x, mask, gamma=1.0)
    assert perfect.item() == pytest.approx(0.0, abs=1e-6)


def test_class_weights_power_smoothing():
    y = torch.tensor([0] * 90 + [1] * 10)
    w = class_weights(y, n_classes=3, power=0.5)
    assert w[2] == 0 and w[1] / w[0] == pytest.approx(3.0)  # sqrt(90 / 10)
    assert w[:2].mean().item() == pytest.approx(1.0)


def _tiny_cfg():
    return load_config(CONFIG, {
        "pretrain": {"epochs": 3, "warmup_epochs": 1, "batch_size": 64, "centroid_min_flows": 50},
        "probe": {"epochs": 30}, "svdd": {"epochs": 3},
        "trees": {"rf": {"n_estimators": 5, "max_depth": 4}, "xgb": {"n_estimators": 3}},
    })


def test_pretraining_warmup_centroid_and_determinism(tmp_path):
    cfg = _tiny_cfg()
    set_seed(1)
    pool = torch.randn(400, 10)
    full = pretrain_multitask(pool, (0.5, 0.5, 0.2), cfg, seed=3)
    assert [h["novelty_active"] for h in full["history"]] == [False, True, True]
    assert full["history"][0]["nov"] == 0.0 and full["history"][1]["nov"] > 0
    assert full["centroid"].shape == (128,)
    again = pretrain_multitask(pool, (0.5, 0.5, 0.2), cfg, seed=3)
    assert torch.equal(full["centroid"], again["centroid"])
    rec_only = pretrain_multitask(pool, (0.5, 0.0, 0.0), cfg, seed=3)
    assert all(h["con"] == 0.0 and not h["novelty_active"] for h in rec_only["history"])
    with pytest.raises(ValueError):
        pretrain_multitask(pool[:10], (0.5, 0.5, 0.2), cfg, seed=3)

    first = load_or_pretrain("F", pool, cfg, 3, tmp_path, "B")
    assert (tmp_path / "F_condB_seed3.pt").is_file()
    cached = load_or_pretrain("F", pool, cfg, 3, tmp_path, "B")
    assert torch.equal(first["centroid"], cached["centroid"]) and cached["seconds"] == first["seconds"]
    with pytest.raises(ValueError, match="different data"):
        load_or_pretrain("F", pool[:300], cfg, 3, tmp_path, "B")
    assert load_or_pretrain("SCARF", pool, cfg, 3, tmp_path, "B")["kind"] == "SCARF"


def test_probe_learns_separable_classes():
    cfg = _tiny_cfg()
    set_seed(0)
    y = torch.arange(600) % 3
    z = torch.randn(600, 128) * 0.1 + torch.nn.functional.one_hot(y, 128).float() * 3
    probe = train_probe(z, y, 3, cfg)
    assert (probe(z).argmax(dim=1) == y).float().mean() > 0.95


def test_svdd_scores_are_finite_and_small():
    cfg = _tiny_cfg()
    set_seed(0)
    benign = torch.randn(500, 10).clamp(-10, 10)
    net, center = train_svdd(benign, cfg)
    scores = ((net(benign) - center) ** 2).sum(dim=1)
    assert torch.isfinite(scores).all() and scores.max() < 1e4
    with pytest.raises(ValueError):
        train_svdd(benign[:0], cfg)


@pytest.mark.parametrize("name", ["RF", "XGB"])
def test_tree_proba_handles_missing_classes(name):
    cfg = _tiny_cfg()
    rng = np.random.default_rng(0)
    y = np.repeat([0, 2, 5], 40)  # classes 1, 3, 4 absent from this budget
    x = rng.normal(size=(120, 6)).astype("float32") + y[:, None]
    model, present = fit_tree(name, x, y, cfg, seed=0)
    proba = tree_proba(model, present, x, n_classes=6)
    assert proba.shape == (120, 6) and np.allclose(proba.sum(axis=1), 1.0, atol=1e-5)
    assert (proba[:, [1, 3, 4]] == 0).all()
    assert (proba.argmax(axis=1) == y).mean() > 0.9
    with pytest.raises(ValueError):
        fit_tree("SVM", x, y, cfg, seed=0)
