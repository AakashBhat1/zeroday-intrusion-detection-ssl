"""Zero-day scores. Higher always means more novel."""

from __future__ import annotations

import numpy as np


def centroid_distance(z: np.ndarray, centroid: np.ndarray) -> np.ndarray:
    """Decoupled score: squared Euclidean distance from the latent to the centroid."""
    return ((np.asarray(z, dtype="float64") - np.asarray(centroid, dtype="float64")) ** 2).sum(axis=1)


def msp_score(proba: np.ndarray) -> np.ndarray:
    """Maximum-softmax-probability rejection score: 1 - max class probability."""
    return 1.0 - np.asarray(proba, dtype="float64").max(axis=1)


def _shrunk_precision(centered: np.ndarray, shrinkage: float) -> np.ndarray:
    """Inverse covariance with a ridge of ``shrinkage`` times the mean variance (keeps it invertible)."""
    dim = centered.shape[1]
    cov = centered.T @ centered / max(len(centered) - 1, 1)
    ridge = shrinkage * max(float(np.trace(cov)) / dim, 1e-12)
    return np.linalg.inv(cov + ridge * np.eye(dim))


def _quadratic_form(diff: np.ndarray, precision: np.ndarray) -> np.ndarray:
    """Row-wise diff @ precision @ diff (BLAS matmul; a three-operand einsum is far slower)."""
    return ((diff @ precision) * diff).sum(axis=1)


def mahalanobis_distance(z: np.ndarray, reference: np.ndarray, shrinkage: float = 0.01) -> np.ndarray:
    """Squared Mahalanobis distance from each latent to the reference set's mean."""
    z, reference = np.asarray(z, dtype="float64"), np.asarray(reference, dtype="float64")
    mean = reference.mean(axis=0)
    precision = _shrunk_precision(reference - mean, shrinkage)
    diff = z - mean
    return _quadratic_form(diff, precision)


def class_mahalanobis_min(z: np.ndarray, reference: np.ndarray, labels: np.ndarray,
                          shrinkage: float = 0.01) -> np.ndarray:
    """Smallest squared Mahalanobis distance to any known class mean (shared within-class covariance)."""
    z, reference = np.asarray(z, dtype="float64"), np.asarray(reference, dtype="float64")
    classes = np.unique(labels)
    means = np.stack([reference[labels == c].mean(axis=0) for c in classes])
    centered = reference - means[np.searchsorted(classes, labels)]
    precision = _shrunk_precision(centered, shrinkage)
    best = np.full(len(z), np.inf)
    for mean in means:
        diff = z - mean
        best = np.minimum(best, _quadratic_form(diff, precision))
    return best


def knn_distance(z: np.ndarray, reference: np.ndarray, k: int = 10, max_reference: int = 10000,
                 seed: int = 0, chunk: int = 4096) -> np.ndarray:
    """Euclidean distance to the k-th nearest reference latent."""
    import torch

    ref = np.asarray(reference, dtype="float32")
    if len(ref) > max_reference:
        ref = ref[np.random.default_rng(seed).choice(len(ref), size=max_reference, replace=False)]
    k = min(k, len(ref))
    ref_t = torch.from_numpy(ref)
    out = []
    for start in range(0, len(z), chunk):
        part = torch.from_numpy(np.asarray(z[start:start + chunk], dtype="float32"))
        out.append(torch.cdist(part, ref_t).kthvalue(k, dim=1).values)
    return torch.cat(out).numpy().astype("float64")


LATENT_SCORERS = ("decoupled", "centroid_benign", "maha_benign", "maha_class", "knn_benign")


def latent_scores(z: np.ndarray, pool_centroid: np.ndarray, ref_z: np.ndarray, ref_y: np.ndarray,
                  benign_index: int, seed: int = 0) -> dict[str, np.ndarray]:
    """Every latent-space zero-day score for ``z``.

    ``decoupled`` uses the unlabelled pool centroid (the paper's original score). The
    others use the labelled flows of the current budget (``ref_z``, ``ref_y``) as reference.
    """
    benign = ref_z[ref_y == benign_index]
    if len(benign) < 2:
        raise ValueError("latent scorers need at least two labelled benign flows")
    return {
        "decoupled": centroid_distance(z, pool_centroid),
        "centroid_benign": centroid_distance(z, benign.mean(axis=0)),
        "maha_benign": mahalanobis_distance(z, benign),
        "maha_class": class_mahalanobis_min(z, ref_z, ref_y),
        "knn_benign": knn_distance(z, benign, seed=seed),
    }
