from __future__ import annotations
import numpy as np
from sklearn.datasets import make_blobs, make_classification
from dataclasses import dataclass

@dataclass
class DataSpec:
    n_samples: int
    n_features: int
    centers: int
    compactness: float
    seed: int

def gen_blobs(spec: DataSpec):
    X, y = make_blobs(
        n_samples=spec.n_samples,
        n_features=spec.n_features,
        centers=spec.centers,
        cluster_std=spec.compactness,
        shuffle=True,
        random_state=spec.seed
    )
    return X.astype(np.float32), y

def gen_mixed(spec: DataSpec):
    X, y = make_classification(
        n_samples=spec.n_samples,
        n_features=spec.n_features,
        n_informative=max(2, spec.n_features // 2),
        n_redundant=spec.n_features // 4,
        n_clusters_per_class=1,
        n_classes=min(8, spec.centers),
        class_sep=1.0 / max(0.2, spec.compactness),
        random_state=spec.seed
    )
    return X.astype(np.float32), y

DATASETS = {"blobs": gen_blobs, "mixed": gen_mixed}
