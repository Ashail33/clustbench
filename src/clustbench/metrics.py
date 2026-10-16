from __future__ import annotations
import numpy as np
from scipy.spatial.distance import cdist, pdist
from sklearn.metrics import silhouette_score, davies_bouldin_score, adjusted_rand_score, normalized_mutual_info_score


def compactness(X: np.ndarray, labels: np.ndarray) -> float:
    comps = []
    for lbl in set(labels):
        if lbl == -1:
            continue
        members = X[labels == lbl]
        if len(members) <= 1:
            continue
        c = members.mean(axis=0)
        comps.append(((members - c) ** 2).sum(axis=1).mean())
    return float(np.mean(comps)) if comps else float("nan")


def separation(X: np.ndarray, labels: np.ndarray) -> float:
    cents = []
    for lbl in set(labels):
        if lbl == -1:
            continue
        members = X[labels == lbl]
        if len(members) == 0:
            continue
        cents.append(members.mean(axis=0))
    if len(cents) < 2:
        return float("nan")
    cents = np.vstack(cents)
    d2 = np.sum((cents[:, None, :] - cents[None, :, :]) ** 2, axis=2)
    tri = d2[np.triu_indices_from(d2, k=1)]
    return float(tri.mean())


def dunn_index(X: np.ndarray, labels: np.ndarray) -> float:
    valid = [l for l in set(labels) if l != -1]
    if len(valid) < 2:
        return float("nan")
    clusters = [X[labels == l] for l in valid]
    diam = 0.0
    for C in clusters:
        if len(C) < 2:
            continue
        m = min(len(C), 2000)
        CC = C[np.random.choice(len(C), size=m, replace=False)]
        # pdist/cdist work in O(m^2) memory; the broadcast difference tensor
        # this replaced was O(m^2 d) and reached 16 GB at m=2000, d=500.
        diam = max(diam, float(pdist(CC).max()) if len(CC) > 1 else 0.0)
    if diam == 0:
        return float("nan")
    mind = np.inf
    for i in range(len(clusters)):
        for j in range(i + 1, len(clusters)):
            A, B = clusters[i], clusters[j]
            a = A[np.random.choice(len(A), size=min(len(A), 2000), replace=False)]
            b = B[np.random.choice(len(B), size=min(len(B), 2000), replace=False)]
            mind = min(mind, float(cdist(a, b).min()))
    return float(mind / diam)


def bundle_scores(X, labels, y_true=None):
    out = {
        "compactness": compactness(X, labels),
        "separation": separation(X, labels),
        "davies_bouldin": davies_bouldin_score(X, labels) if len(set(labels)) > 1 else float("nan"),
        "dunn": dunn_index(X, labels),
    }
    if y_true is not None:
        silhouette_kwargs = {}
        if X.shape[0] > 5000:
            silhouette_kwargs = {"sample_size": 5000, "random_state": 42}
        out.update(
            {
                "ari": adjusted_rand_score(y_true, labels),
                "nmi": normalized_mutual_info_score(y_true, labels),
                "silhouette": silhouette_score(X, labels, **silhouette_kwargs)
                if len(set(labels)) > 1
                else float("nan"),
            }
        )
    if y_true is not None:
        out.update(inlier_scores(labels, y_true))
    return out


def inlier_scores(labels, y_true):
    """ARI / NMI restricted to ground-truth inliers (``y_true != -1``).

    Generators label injected outliers and noise points ``-1``. The plain ARI
    and NMI treat those points as one extra ground-truth class, which rewards
    an algorithm for lumping all outliers together and penalises it for
    assigning them to their nearest cluster. Scoring only the inliers removes
    that convention from the comparison. Returns ``None`` values when there
    are no injected outliers (the scores would equal ARI / NMI) or fewer than
    two inliers.
    """
    y_true = np.asarray(y_true)
    labels = np.asarray(labels)
    mask = y_true != -1
    if mask.all() or mask.sum() < 2:
        return {"ari_inliers": None, "nmi_inliers": None}
    return {
        "ari_inliers": float(adjusted_rand_score(y_true[mask], labels[mask])),
        "nmi_inliers": float(normalized_mutual_info_score(y_true[mask], labels[mask])),
    }
