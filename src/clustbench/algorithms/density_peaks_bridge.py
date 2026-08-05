"""Density Peaks Clustering (Rodriguez-Laio, 2014) with a kNN label-spreading
bridge (Zhou et al., 2004) to reassign borderline noise while preserving true
outliers.

Pipeline (each stage becomes a Step in the trajectory):

1. Density: compute (kNN-restricted for large n) pairwise distances and a
   Gaussian local density rho with cutoff d_c set to the 2%-percentile of
   pairwise distances.
2. Parent link: for each point, delta_i = distance to the nearest
   higher-density point, gamma_i = rho_i * delta_i. Cluster centers are the
   top-k gamma (k given) or the elbow on the sorted log-gamma curve.
3. Chain assign: sweep in descending rho, each point inherits its parent's
   label -- linear in n once the parent pointers exist.
4. Halo / noise: rho < median(rho) AND delta > q90(delta) -> candidate noise.
5. Bridge: build a symmetric kNN heat-kernel graph, run one round of
   normalized-Laplacian label spreading (Zhou 2004) seeded from confidently
   assigned points, and re-label candidate-noise iff the winning score
   exceeds tau -- otherwise they stay as -1 (true outliers).
"""

from __future__ import annotations

from typing import Any, Optional
import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.sparse import csr_matrix, eye as speye, diags
from scipy.sparse.linalg import spsolve
from sklearn.neighbors import NearestNeighbors

from .base import Algorithm, AlgoResult, Step, register


# -----------------------------------------------------------------------------
# Utility: pairwise or kNN-restricted distance matrix
# -----------------------------------------------------------------------------
def _distance_matrix(X: np.ndarray, knn_threshold: int = 5000, k_nn: int = 30):
    """Return (D, mode) where D is either a dense NxN distance matrix or a
    CSR sparse matrix with only k_nn neighbors filled in. mode in {'full','knn'}.
    """
    n = X.shape[0]
    if n <= knn_threshold:
        D = squareform(pdist(X.astype(np.float64), metric="euclidean"))
        return D, "full"
    k = min(k_nn, n - 1)
    nn = NearestNeighbors(n_neighbors=k + 1).fit(X)
    dists, idx = nn.kneighbors(X)
    rows = np.repeat(np.arange(n), k + 1)
    D_sparse = csr_matrix((dists.ravel(), (rows, idx.ravel())), shape=(n, n))
    D_sparse = D_sparse.maximum(D_sparse.T)  # symmetric
    return D_sparse, "knn"


def _local_density(D, mode: str, dc: float) -> np.ndarray:
    """Gaussian kernel local density rho."""
    if mode == "full":
        W = np.exp(-((D / dc) ** 2))
        np.fill_diagonal(W, 0.0)
        return W.sum(axis=1)
    # sparse: only near neighbors contribute (others assumed ~0 kernel value)
    D_coo = D.tocoo()
    vals = np.exp(-((D_coo.data / dc) ** 2))
    # exclude self-loops
    keep = D_coo.row != D_coo.col
    W = csr_matrix((vals[keep], (D_coo.row[keep], D_coo.col[keep])), shape=D.shape)
    return np.asarray(W.sum(axis=1)).ravel()


def _delta_and_parent(D, mode: str, rho: np.ndarray):
    """For each point i, distance to (and index of) the nearest point of strictly
    higher density. For the global densest point, delta = max distance and
    parent = -1."""
    n = rho.shape[0]
    delta = np.zeros(n, dtype=np.float64)
    parent = -np.ones(n, dtype=np.int64)
    order = np.argsort(-rho)  # descending density
    if mode == "full":
        big = float(D.max()) if n > 1 else 1.0
        for rank, i in enumerate(order):
            if rank == 0:
                delta[i] = big
                continue
            higher = order[:rank]
            d_row = D[i, higher]
            j_local = int(np.argmin(d_row))
            parent[i] = int(higher[j_local])
            delta[i] = float(d_row[j_local])
        return delta, parent
    # sparse fallback: use only kNN entries; missing higher-density neighbors
    # get delta = max_dist_seen (a large finite value).
    D_lil = D.tolil()
    big = float(D.data.max()) if D.nnz else 1.0
    for rank, i in enumerate(order):
        if rank == 0:
            delta[i] = big
            continue
        cols = np.array(D_lil.rows[i], dtype=np.int64)
        if cols.size == 0:
            delta[i] = big
            continue
        vals = np.array(D_lil.data[i], dtype=np.float64)
        higher_mask = rho[cols] > rho[i]
        if not higher_mask.any():
            delta[i] = big
            continue
        c = cols[higher_mask]
        v = vals[higher_mask]
        j = int(np.argmin(v))
        parent[i] = int(c[j])
        delta[i] = float(v[j])
    return delta, parent


def _pick_centers(gamma: np.ndarray, k: Optional[int]) -> np.ndarray:
    """Top-k gamma if k given; otherwise kneedle-style elbow on sorted
    log(gamma). Always returns at least 2 centers when possible."""
    n = gamma.shape[0]
    order = np.argsort(-gamma)
    if k is not None:
        k = max(1, min(int(k), n))
        return order[:k]
    if n < 2:
        return order[:1]
    g_sorted = gamma[order]
    logg = np.log1p(g_sorted - g_sorted.min() + 1e-12)
    # Kneedle: normalize, look at gap to chord
    x = np.linspace(0.0, 1.0, len(logg))
    y = (logg - logg.min()) / (logg.max() - logg.min() + 1e-12)
    chord = 1.0 - x  # descending, so chord goes from 1 to 0
    diff = y - chord
    k_hat = int(np.argmax(diff)) + 1
    k_hat = max(2, min(k_hat, min(20, n)))
    return order[:k_hat]


def _chain_assign(rho: np.ndarray, parent: np.ndarray, centers: np.ndarray) -> np.ndarray:
    """Assign each non-center point the label of its parent, walked greedily
    in descending-density order (linear time)."""
    n = rho.shape[0]
    labels = -np.ones(n, dtype=np.int64)
    for c_idx, c in enumerate(centers):
        labels[int(c)] = c_idx
    for i in np.argsort(-rho):
        if labels[i] != -1:
            continue
        p = parent[i]
        if p == -1 or labels[p] == -1:
            # shouldn't happen if we go in descending rho -- fallback: nearest center
            labels[i] = 0
        else:
            labels[i] = labels[p]
    return labels


def _label_spread(X: np.ndarray, seed_labels: np.ndarray, k_graph: int = 15,
                  alpha: float = 0.9) -> tuple[np.ndarray, np.ndarray]:
    """One-shot Zhou et al. label spreading on a symmetric kNN heat-kernel graph.
    Points with seed_labels == -1 are unlabeled. Returns (best_label, max_score)."""
    n = X.shape[0]
    k = min(k_graph, max(2, n - 1))
    nn = NearestNeighbors(n_neighbors=k + 1).fit(X)
    d, idx = nn.kneighbors(X)
    sigma = float(np.median(d[:, 1:])) + 1e-12
    w = np.exp(-((d / sigma) ** 2))
    rows = np.repeat(np.arange(n), k + 1)
    W = csr_matrix((w.ravel(), (rows, idx.ravel())), shape=(n, n))
    W = W.maximum(W.T)
    W = W - diags(W.diagonal())  # zero diagonal
    deg = np.asarray(W.sum(axis=1)).ravel()
    d_inv_sqrt = 1.0 / np.sqrt(np.maximum(deg, 1e-12))
    D_inv = diags(d_inv_sqrt)
    S = D_inv @ W @ D_inv
    # F* = (1-alpha) * (I - alpha S)^{-1} Y
    seeded = seed_labels >= 0
    classes = np.unique(seed_labels[seeded])
    if classes.size == 0:
        return -np.ones(n, dtype=np.int64), np.zeros(n)
    Y = np.zeros((n, classes.size), dtype=np.float64)
    for j, c in enumerate(classes):
        Y[(seed_labels == c) & seeded, j] = 1.0
    A = (speye(n) - alpha * S).tocsc()
    F = np.column_stack([spsolve(A, Y[:, j]) for j in range(classes.size)])
    F *= (1.0 - alpha)
    # Row-normalize scores so tau is meaningful across densities
    row_sum = F.sum(axis=1, keepdims=True)
    row_sum = np.where(row_sum <= 0, 1.0, row_sum)
    F_norm = F / row_sum
    winner = np.argmax(F_norm, axis=1)
    best_score = F_norm[np.arange(n), winner]
    return classes[winner].astype(np.int64), best_score


def _modularity(labels: np.ndarray, X: np.ndarray, k_graph: int = 15) -> float:
    """Newman modularity on a symmetric kNN graph. Cheap surrogate cost."""
    n = X.shape[0]
    k = min(k_graph, max(2, n - 1))
    nn = NearestNeighbors(n_neighbors=k + 1).fit(X)
    _, idx = nn.kneighbors(X)
    rows = np.repeat(np.arange(n), k + 1)
    A = csr_matrix((np.ones(rows.size), (rows, idx.ravel())), shape=(n, n))
    A = A.maximum(A.T)
    A.setdiag(0)
    A.eliminate_zeros()
    m2 = A.sum()
    if m2 == 0:
        return 0.0
    deg = np.asarray(A.sum(axis=1)).ravel()
    A_coo = A.tocoo()
    same = labels[A_coo.row] == labels[A_coo.col]
    edge_term = float(A_coo.data[same].sum())
    # k_i k_j / 2m over same-community pairs (label -1 excluded)
    valid_labels = np.unique(labels[labels >= 0])
    deg_term = 0.0
    for c in valid_labels:
        deg_term += float(deg[labels == c].sum() ** 2)
    return (edge_term - deg_term / m2) / m2


@register
class Density_peaks_bridge(Algorithm):
    """Rodriguez-Laio density peaks + kNN label-spreading bridge."""

    def __init__(
        self,
        knn_threshold: int = 5000,
        k_nn_rho: int = 30,
        d_c_percentile: float = 2.0,
        k_graph: int = 15,
        alpha: float = 0.9,
        tau: float = 0.6,
        **kwargs: Any,
    ) -> None:
        self.name = "density_peaks_bridge"
        self.knn_threshold = knn_threshold
        self.k_nn_rho = k_nn_rho
        self.d_c_percentile = d_c_percentile
        self.k_graph = k_graph
        self.alpha = alpha
        self.tau = tau

    def fit_predict(self, X: np.ndarray, k: Optional[int] = None) -> AlgoResult:
        X = np.asarray(X, dtype=np.float64)
        n = X.shape[0]
        traj: list[Step] = []

        # --- Stage 1: density -------------------------------------------------
        D, mode = _distance_matrix(X, self.knn_threshold, self.k_nn_rho)
        if mode == "full":
            all_d = D[np.triu_indices(n, k=1)]
        else:
            all_d = D.data[D.data > 0]
        d_c = float(np.percentile(all_d, self.d_c_percentile)) if all_d.size else 1.0
        d_c = max(d_c, 1e-9)
        rho = _local_density(D, mode, d_c)

        # --- Stage 2: parent link + centers -----------------------------------
        delta, parent = _delta_and_parent(D, mode, rho)
        gamma = rho * delta
        centers = _pick_centers(gamma, k)
        cost1 = float(-rho[centers].sum())
        traj.append(Step(
            step_idx=0, cost=cost1, delta_cost=None, accepted=True,
            action={"type": "select_centers", "n_centers": int(centers.size)},
            state={"d_c": d_c, "centers": centers.tolist()},
        ))

        # --- Stage 3: chain assign --------------------------------------------
        labels = _chain_assign(rho, parent, centers)

        # --- Stage 4: candidate noise -----------------------------------------
        rho_med = float(np.median(rho))
        delta_q90 = float(np.quantile(delta, 0.90))
        candidate_noise = (rho < rho_med) & (delta > delta_q90)
        # Confident seeds are everything else that already has a label
        seed_labels = labels.copy()
        seed_labels[candidate_noise] = -1
        n_candidates = int(candidate_noise.sum())
        traj.append(Step(
            step_idx=1, cost=cost1, delta_cost=0.0, accepted=True,
            action={"type": "mark_noise", "n_candidate_noise": n_candidates},
            state={"rho_median": rho_med, "delta_q90": delta_q90},
        ))

        # --- Stage 5: bridge (label spreading) --------------------------------
        n_noise = n_candidates
        if n_candidates > 0 and (seed_labels >= 0).any():
            spread_labels, spread_scores = _label_spread(
                X, seed_labels, k_graph=self.k_graph, alpha=self.alpha
            )
            accept = candidate_noise & (spread_scores >= self.tau)
            labels_final = labels.copy()
            labels_final[candidate_noise] = -1  # tentatively noise
            labels_final[accept] = spread_labels[accept]
            n_noise = int((labels_final == -1).sum())
        else:
            labels_final = labels

        cost2 = float(-_modularity(labels_final, X, self.k_graph))
        traj.append(Step(
            step_idx=2, cost=cost2, delta_cost=cost2 - cost1, accepted=True,
            action={"type": "bridge", "tau": self.tau, "n_noise": n_noise},
            state={"d_c": d_c, "centers": centers.tolist(), "n_noise": n_noise},
        ))

        return AlgoResult(
            labels=labels_final.astype(np.int64),
            extra={
                "d_c": d_c,
                "n_centers": int(centers.size),
                "n_noise": n_noise,
                "mode": mode,
            },
            trajectory=traj,
        )
