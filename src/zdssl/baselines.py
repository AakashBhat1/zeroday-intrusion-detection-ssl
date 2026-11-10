"""Baselines: Deep SVDD, Random Forest, XGBoost."""

from __future__ import annotations

import io
from typing import Any, Mapping

import joblib
import numpy as np
import torch
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from .encoder import SvddNet
from .pretrain import encode, iterate_batches


def train_svdd(x_benign: torch.Tensor, cfg: Mapping[str, Any]) -> tuple[SvddNet, torch.Tensor]:
    """One-class Deep SVDD on benign flows. The center comes from the initial forward pass."""
    sv, device = cfg["svdd"], x_benign.device
    if len(x_benign) == 0:
        raise ValueError("Deep SVDD needs at least one benign training flow")
    net = SvddNet(x_benign.shape[1], sv["hidden"], sv["out_dim"]).to(device)
    center = encode(net, x_benign).mean(dim=0)
    eps = sv["center_eps"]  # keep the center away from zero so the trivial solution is not free
    center = torch.where(center.abs() < eps, torch.where(center < 0, -eps, eps), center)
    optimizer = torch.optim.Adam(net.parameters(), lr=sv["lr"], weight_decay=sv["weight_decay"])
    for _ in range(sv["epochs"]):
        net.train()
        for batch in iterate_batches(len(x_benign), sv["batch_size"], device):
            loss = ((net(x_benign[batch]) - center) ** 2).sum(dim=1).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
    return net.eval(), center


def fit_tree(model_name: str, x: np.ndarray, y: np.ndarray, cfg: Mapping[str, Any], seed: int):
    """Fit RF or XGB on the classes present. Returns (model, present class ids)."""
    present = np.unique(y)
    y_local = np.searchsorted(present, y)
    if model_name == "RF":
        model = RandomForestClassifier(random_state=seed, n_jobs=-1, **cfg["trees"]["rf"])
    elif model_name == "XGB":
        model = XGBClassifier(objective="multi:softprob", tree_method="hist", random_state=seed,
                              n_jobs=-1, **cfg["trees"]["xgb"])
    else:
        raise ValueError(f"unknown tree model {model_name}")
    model.fit(x, y_local)
    return model, present


def tree_proba(model, present: np.ndarray, x: np.ndarray, n_classes: int) -> np.ndarray:
    """Class probabilities over all ``n_classes`` (absent classes get probability 0)."""
    local = np.asarray(model.predict_proba(x), dtype="float64")
    if local.ndim == 1:  # binary XGBoost returns one column per row
        local = np.column_stack([1 - local, local])
    full = np.zeros((len(x), n_classes))
    full[:, present[: local.shape[1]]] = local
    return full


def tree_size(model_name: str, model) -> int:
    """Model size: total nodes for RF, number of trees for XGB."""
    if model_name == "RF":
        return int(sum(est.tree_.node_count for est in model.estimators_))
    return int(len(model.get_booster().get_dump()))


def serialized_mb(obj) -> float:
    """Size of ``obj`` on disk, in MB (joblib for sklearn/xgboost, torch.save for state dicts)."""
    buffer = io.BytesIO()
    if isinstance(obj, dict):
        torch.save(obj, buffer)
    else:
        joblib.dump(obj, buffer)
    return buffer.getbuffer().nbytes / 1e6
