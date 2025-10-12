from __future__ import annotations
import numpy as np
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
        D = np.linalg.norm(CC[:, None, :] - CC[None, :, :], axis=2)
        diam = max(diam, float(D.max()))
    if diam == 0:
        return float("nan")
    mind = np.inf
    for i in range(len(clusters)):
        for j in range(i + 1, len(clusters)):
            A, B = clusters[i], clusters[j]
            a = A[np.random.choice(len(A), size=min(len(A), 2000), replace=False)]
            b = B[np.random.choice(len(B), size=min(len(B), 2000), replace=False)]
            D = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=2)
            mind = min(mind, float(D.min()))
    return float(mind / diam)


def bundle_scores(X, labels, y_true=None):
    out = {
        "compactness": compactness(X, labels),
        "separation": separation(X, labels),
        "davies_bouldin": davies_bouldin_score(X, labels) if len(set(labels)) > 1 else float("nan"),
        "dunn": dunn_index(X, labels),
    }
    if y_true is not None:
        out.update(
            {
                "ari": adjusted_rand_score(y_true, labels),
                "nmi": normalized_mutual_info_score(y_true, labels),
                "silhouette": silhouette_score(X, labels)
                if len(set(labels)) > 1
                else float("nan"),
            }
        )
    return out
