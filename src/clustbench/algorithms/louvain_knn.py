"""louvain_knn: graph-native clustering for the registry's ``graph_*`` gap.

The benchmark's graph datasets (``graph_karate``, ``graph_sbm``) ship a
16-dim node-feature stack (6 centralities + 10 bottom Laplacian eigvecs).
Spectral / centroid algorithms then run *another* round of dimension-
reduction on top of those features — on a 34-node karate graph this is
curse-of-dimensionality territory, and ARI tanks below 0.15 for almost
every algorithm in the registry (rapid_v2 hits 0.428).

Idea: rebuild the graph **implicitly** from the feature space via a
kNN graph, then optimise community structure on that graph directly via
modularity (Louvain). This is qualitatively different from ``spectral``
(which uses kmeans on eigenvectors): Louvain optimises modularity directly
without needing ``k`` specified, and is designed for graph-native problems.

Algorithm summary
-----------------
1. Build kNN graph on ``X`` (sklearn's ``kneighbors_graph``, symmetrised).
2. Run Louvain modularity optimisation.
3. Reconcile Louvain's natural ``k_found`` with the requested ``k`` via
   merge (smallest into nearest-by-edge-density) or split (spectral
   bipartition on the largest cluster).
4. Edge-case fallbacks: tiny n => kmeans; missing networkx +
   python-louvain => a self-contained numpy Louvain implementation; if
   *that* also explodes => spectral on the kNN affinity.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.neighbors import kneighbors_graph

from .base import Algorithm, AlgoResult, Step, register


# ---------------------------------------------------------------------------
# Self-contained numpy Louvain. Used when neither networkx nor python-louvain
# is installed. Operates on a symmetric sparse adjacency (scipy.sparse).
# ---------------------------------------------------------------------------

def _modularity(adj_lil, labels: np.ndarray, m2: float) -> float:
    """Compute Newman-Girvan modularity on a sparse adjacency.

    ``m2`` is ``2 * |E|`` (sum of edge weights, double-counted because
    the graph is undirected). ``adj_lil`` is a scipy.sparse matrix.
    """
    if m2 <= 0:
        return 0.0
    A = adj_lil.tocsr()
    n = A.shape[0]
    # Per-node weighted degree.
    deg = np.asarray(A.sum(axis=1)).ravel()
    # Sum over same-community edges, minus expected weight under config model.
    same = 0.0
    expected = 0.0
    for c in np.unique(labels):
        members = np.where(labels == c)[0]
        if len(members) == 0:
            continue
        sub = A[members][:, members]
        same += float(sub.sum())
        dsum = float(deg[members].sum())
        expected += (dsum * dsum) / m2
    return (same - expected) / m2


def _numpy_louvain(adj_csr, rng: np.random.Generator, max_passes: int = 30) -> np.ndarray:
    """One-level Louvain (modularity-greedy node moves).

    Not as polished as networkx's multi-level implementation but plenty
    good enough for n<=~2000 and for the registry's gap-filling purpose.
    We loop node-move passes until no node changes community in a pass.

    Modularity convention (undirected graph):
        Q = (1 / 2m) Σ_{ij} [A_{ij} - k_i k_j / (2m)] δ(c_i, c_j)
    where ``2m`` = ``deg.sum()`` (sum of weighted degrees). ΔQ for moving
    node ``i`` from community ``ci`` to community ``c`` is:
        ΔQ = (k_i_in_c - k_i_in_ci) / m - k_i * (Σtot_c - Σtot_ci) / (2 m^2)
    after the standard "remove i from ci first" trick (so Σtot_ci no
    longer includes k_i and k_i_in_ci no longer counts i's self loop).
    """
    A = adj_csr.tocsr()
    n = A.shape[0]
    if n == 0:
        return np.zeros(0, dtype=np.int64)
    deg = np.asarray(A.sum(axis=1)).ravel().astype(np.float64)
    two_m = float(deg.sum())  # 2m for the undirected graph
    if two_m <= 0:
        return np.arange(n, dtype=np.int64)

    labels = np.arange(n, dtype=np.int64)
    comm_deg = deg.copy()  # Σtot per community id (same id space as labels)

    for _pass in range(max_passes):
        changed = 0
        order = rng.permutation(n)
        for i in order:
            ci = int(labels[i])
            start, end = A.indptr[i], A.indptr[i + 1]
            nbr = A.indices[start:end]
            wts = A.data[start:end].astype(np.float64)
            if len(nbr) == 0:
                continue
            # Total edge weight from i into each neighbour community.
            comm_w: Dict[int, float] = {}
            for c, w in zip(labels[nbr], wts):
                cc = int(c)
                comm_w[cc] = comm_w.get(cc, 0.0) + float(w)

            ki = deg[i]
            # "Remove" i from ci before considering moves.
            comm_deg[ci] -= ki
            w_to_ci = comm_w.get(ci, 0.0)

            best_c = ci
            best_gain = 0.0
            # ΔQ(ci -> c) = 2*(k_in_c - k_in_ci)/two_m
            #              - 2*ki*(Σtot_c - Σtot_ci) / two_m^2
            # Constants in front are the same for every candidate c, so
            # they don't affect the argmax but we keep them for the
            # numerical comparison against `best_gain = 0` (= no move).
            inv_two_m = 1.0 / two_m
            inv_two_m_sq = inv_two_m * inv_two_m
            for c, k_in_c in comm_w.items():
                gain = 2.0 * (k_in_c - w_to_ci) * inv_two_m \
                    - 2.0 * ki * (comm_deg[c] - comm_deg[ci]) * inv_two_m_sq
                if gain > best_gain + 1e-12:
                    best_gain = gain
                    best_c = c

            comm_deg[best_c] += ki
            if best_c != ci:
                labels[i] = best_c
                changed += 1
        if changed == 0:
            break

    _, labels = np.unique(labels, return_inverse=True)
    return labels.astype(np.int64)


# ---------------------------------------------------------------------------
# Reconciliation helpers
# ---------------------------------------------------------------------------

def _merge_smallest_into_nearest(
    labels: np.ndarray, adj_csr, target_k: int
) -> Tuple[np.ndarray, int]:
    """Repeatedly merge the smallest cluster into the cluster sharing
    the most kNN edges with it, until ``len(unique(labels)) == target_k``.

    Returns the new labels and the number of merge operations performed.
    """
    n_merges = 0
    while True:
        uniq, counts = np.unique(labels, return_counts=True)
        if len(uniq) <= target_k:
            break
        # Smallest cluster.
        smallest = int(uniq[np.argmin(counts)])
        members = np.where(labels == smallest)[0]
        # Edge-weight to every other cluster.
        sub = adj_csr[members]
        # cross[j] = total edge weight from smallest -> cluster of j
        sums: Dict[int, float] = {}
        rows, cols = sub.nonzero()
        data = sub.data
        for col, w in zip(cols, data):
            c = int(labels[col])
            if c == smallest:
                continue
            sums[c] = sums.get(c, 0.0) + float(w)
        if not sums:
            # Disconnected — merge into the next-smallest cluster (closest by size).
            others = [int(u) for u in uniq if int(u) != smallest]
            if not others:
                break
            target = others[0]
        else:
            target = max(sums.items(), key=lambda kv: kv[1])[0]
        labels = np.where(labels == smallest, target, labels)
        # Densify labels so cluster ids stay 0..k-1.
        _, labels = np.unique(labels, return_inverse=True)
        labels = labels.astype(np.int64)
        n_merges += 1
    return labels, n_merges


def _split_largest_via_spectral(
    labels: np.ndarray, adj_csr, target_k: int, random_state: int
) -> Tuple[np.ndarray, int]:
    """Recursively spectral-bipartition the largest cluster until the
    label count reaches ``target_k``.
    """
    from sklearn.cluster import SpectralClustering

    n_splits = 0
    while True:
        uniq, counts = np.unique(labels, return_counts=True)
        if len(uniq) >= target_k:
            break
        largest = int(uniq[np.argmax(counts)])
        members = np.where(labels == largest)[0]
        if len(members) < 2:
            break
        sub = adj_csr[members][:, members]
        # Symmetrise / clip to non-negative for SpectralClustering's precomputed mode.
        sub = sub.maximum(sub.T)
        try:
            sc = SpectralClustering(
                n_clusters=2,
                affinity="precomputed",
                assign_labels="kmeans",
                random_state=random_state,
            )
            sub_labels = sc.fit_predict(sub.toarray())
        except Exception:
            # Fall back: split by sign of Fiedler vector.
            sub_labels = _fiedler_bipartition(sub.toarray())
        # Map the 2-way split to fresh labels.
        new_id = int(labels.max()) + 1
        # Keep one half as ``largest``, send the other half to ``new_id``.
        flip = sub_labels == 1
        labels = labels.copy()
        labels[members[flip]] = new_id
        _, labels = np.unique(labels, return_inverse=True)
        labels = labels.astype(np.int64)
        n_splits += 1
        if n_splits > target_k * 3:  # guard against pathological loops
            break
    return labels, n_splits


def _fiedler_bipartition(A_dense: np.ndarray) -> np.ndarray:
    """Sign-of-Fiedler-vector bipartition. Used when SpectralClustering fails."""
    deg = A_dense.sum(axis=1)
    L = np.diag(deg) - A_dense
    try:
        vals, vecs = np.linalg.eigh(L)
        if vecs.shape[1] < 2:
            return np.zeros(A_dense.shape[0], dtype=np.int64)
        fiedler = vecs[:, 1]
    except Exception:
        return np.zeros(A_dense.shape[0], dtype=np.int64)
    return (fiedler > 0).astype(np.int64)


# ---------------------------------------------------------------------------
# Main algorithm
# ---------------------------------------------------------------------------

@register
class Louvain_knn(Algorithm):
    """Louvain modularity on a kNN graph rebuilt from the feature space.

    Designed for the benchmark's graph-native datasets (``graph_karate``,
    ``graph_sbm``) where running spectral on top of a 16-dim feature stack
    drowns the Fiedler signal. Also reasonably well-behaved on small
    convex point-cloud problems via the merge/split reconciliation step.
    """

    def __init__(self, random_state: int = 42, **kwargs: Any) -> None:
        self.name = "louvain_knn"
        self.random_state = random_state

    def fit_predict(self, X: np.ndarray, k: Optional[int] = None) -> AlgoResult:
        assert k is not None, "k (number of clusters) must be provided"
        n = X.shape[0]
        rng = np.random.default_rng(self.random_state)
        trajectory: List[Step] = []

        # --- Edge case: tiny n => plain kmeans. -----------------------------
        if n < 8:
            from sklearn.cluster import KMeans

            km = KMeans(n_clusters=min(k, n), n_init=10, random_state=self.random_state)
            labels = km.fit_predict(X).astype(np.int64)
            return AlgoResult(
                labels=labels,
                extra={
                    "k_nn": 0,
                    "k_found": int(len(np.unique(labels))),
                    "final_k": int(len(np.unique(labels))),
                    "modularity": 0.0,
                    "merge_or_split": "none",
                    "fallback": "tiny_n_kmeans",
                },
                trajectory=[Step(step_idx=0, cost=0.0, action={"type": "tiny_n_kmeans"})],
            )

        # --- 1. Build kNN graph. --------------------------------------------
        # Adaptive preprocessing: the graph feature stacks (``gen_graph_*``)
        # carry heterogeneous per-column scales — raw centralities can dwarf
        # the diffusion-weighted Fiedler vector by an order of magnitude,
        # which buries the community signal in Euclidean kNN distance.
        # When the ratio of max to min per-column std is large (>5), L2-
        # normalise each row so kNN distance becomes effectively cosine-like
        # and scale-tolerant. On well-conditioned point clouds (e.g.
        # ``gen_mdcgen``) the ratio is small, so raw features are kept and
        # the algorithm behaves like vanilla Euclidean Louvain.
        X_arr = np.asarray(X, dtype=np.float64)
        col_stds = X_arr.std(axis=0) + 1e-9
        scale_ratio = float(col_stds.max() / col_stds.min())
        if scale_ratio > 5.0:
            row_norms = np.linalg.norm(X_arr, axis=1, keepdims=True) + 1e-9
            X_for_knn = X_arr / row_norms
            preproc = "l2_row_norm"
        else:
            X_for_knn = X_arr
            preproc = "raw"
        k_nn = min(15, max(3, int(round(np.log2(n) * 2))))
        if k_nn >= n:
            k_nn = n - 1
        A_sparse = kneighbors_graph(
            X_for_knn, n_neighbors=k_nn, mode="connectivity", include_self=False
        )
        # Symmetrise.
        A_sym = (A_sparse + A_sparse.T) * 0.5
        A_sym = A_sym.tocsr()
        trajectory.append(
            Step(
                step_idx=0,
                cost=0.0,
                action={
                    "type": "build_knn_graph",
                    "k_nn": k_nn,
                    "n_edges": int(A_sym.nnz / 2),
                    "preproc": preproc,
                    "scale_ratio": scale_ratio,
                },
                state={"n_nodes": int(n)},
            )
        )

        # --- 2. Run Louvain. ------------------------------------------------
        fallback_used = "none"
        louvain_labels: Optional[np.ndarray] = None
        try:
            import networkx as nx  # type: ignore

            G = nx.from_scipy_sparse_array(A_sym) if hasattr(nx, "from_scipy_sparse_array") \
                else nx.from_scipy_sparse_matrix(A_sym)
            try:
                from networkx.algorithms.community import louvain_communities  # type: ignore

                comms = louvain_communities(G, seed=self.random_state, weight="weight")
                louvain_labels = _comms_to_labels(comms, n)
                fallback_used = "networkx_louvain"
            except Exception:
                from networkx.algorithms.community import greedy_modularity_communities  # type: ignore

                comms = greedy_modularity_communities(G, weight="weight")
                louvain_labels = _comms_to_labels(list(comms), n)
                fallback_used = "networkx_greedy_modularity"
        except Exception:
            try:
                import community as community_louvain  # type: ignore
                import networkx as nx  # type: ignore  # python-louvain needs nx

                G = nx.from_scipy_sparse_array(A_sym) if hasattr(nx, "from_scipy_sparse_array") \
                    else nx.from_scipy_sparse_matrix(A_sym)
                part = community_louvain.best_partition(G, random_state=self.random_state)
                louvain_labels = np.array([part[i] for i in range(n)], dtype=np.int64)
                _, louvain_labels = np.unique(louvain_labels, return_inverse=True)
                louvain_labels = louvain_labels.astype(np.int64)
                fallback_used = "python_louvain"
            except Exception:
                try:
                    louvain_labels = _numpy_louvain(A_sym, rng=rng)
                    fallback_used = "numpy_louvain"
                except Exception:
                    # --- Last-resort: spectral on the kNN affinity. ----------
                    try:
                        from sklearn.cluster import SpectralClustering

                        sc = SpectralClustering(
                            n_clusters=k,
                            affinity="precomputed_nearest_neighbors",
                            n_neighbors=k_nn,
                            random_state=self.random_state,
                        )
                        spectral_labels = sc.fit_predict(A_sym.toarray()).astype(np.int64)
                        return AlgoResult(
                            labels=spectral_labels,
                            extra={
                                "k_nn": k_nn,
                                "k_found": int(len(np.unique(spectral_labels))),
                                "final_k": int(len(np.unique(spectral_labels))),
                                "modularity": float("nan"),
                                "merge_or_split": "none",
                                "fallback": "spectral_on_knn",
                            },
                            trajectory=trajectory,
                        )
                    except Exception:
                        # Truly cornered — return one cluster.
                        return AlgoResult(
                            labels=np.zeros(n, dtype=np.int64),
                            extra={
                                "k_nn": k_nn,
                                "k_found": 1,
                                "final_k": 1,
                                "modularity": 0.0,
                                "merge_or_split": "none",
                                "fallback": "everything_failed",
                            },
                            trajectory=trajectory,
                        )

        assert louvain_labels is not None
        m2 = float(np.asarray(A_sym.sum(axis=1)).sum())
        mod_initial = _modularity(A_sym, louvain_labels, m2)
        k_found = int(len(np.unique(louvain_labels)))
        trajectory.append(
            Step(
                step_idx=1,
                cost=-mod_initial,
                action={"type": "run_louvain", "engine": fallback_used},
                state={"k_found": k_found, "modularity": mod_initial},
            )
        )

        # --- 3. Reconcile with target k. ------------------------------------
        merge_or_split = "none"
        if k_found > k:
            new_labels, n_merges = _merge_smallest_into_nearest(louvain_labels, A_sym, k)
            louvain_labels = new_labels
            merge_or_split = f"merge_{n_merges}"
        elif k_found < k:
            new_labels, n_splits = _split_largest_via_spectral(
                louvain_labels, A_sym, k, self.random_state
            )
            louvain_labels = new_labels
            merge_or_split = f"split_{n_splits}"
        final_k = int(len(np.unique(louvain_labels)))
        mod_final = _modularity(A_sym, louvain_labels, m2)
        trajectory.append(
            Step(
                step_idx=2,
                cost=-mod_final,
                delta_cost=(-mod_final) - (-mod_initial),
                action={"type": "reconcile_k", "decision": merge_or_split},
                state={"final_k": final_k, "modularity": mod_final},
            )
        )

        return AlgoResult(
            labels=louvain_labels.astype(np.int64),
            extra={
                "k_nn": k_nn,
                "k_found": k_found,
                "final_k": final_k,
                "modularity": float(mod_final),
                "merge_or_split": merge_or_split,
                "fallback": fallback_used,
                "preproc": preproc,
            },
            trajectory=trajectory,
        )


def _comms_to_labels(comms, n: int) -> np.ndarray:
    """Convert a list of frozenset-of-node-indices to a dense labels array."""
    labels = np.zeros(n, dtype=np.int64)
    for c_id, members in enumerate(comms):
        for v in members:
            labels[int(v)] = c_id
    return labels


# ---------------------------------------------------------------------------
# Inline smoke test.
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from sklearn.metrics import adjusted_rand_score

    from clustbench.datasets import DataSpec, gen_mdcgen
    from clustbench.datasets_domains import gen_graph_karate, gen_graph_sbm

    cases = [
        ("graph_karate",
         gen_graph_karate(DataSpec(n_samples=34, n_features=16, centers=2, compactness=1.0, seed=1)),
         2),
        ("graph_sbm",
         gen_graph_sbm(DataSpec(n_samples=200, n_features=16, centers=4, compactness=1.0, seed=1)),
         4),
        ("mdcgen",
         gen_mdcgen(DataSpec(n_samples=400, n_features=10, centers=3, compactness=1.0, seed=1)),
         3),
    ]
    for name, (X, y), k in cases:
        algo = Louvain_knn()
        res = algo.fit_predict(np.asarray(X), k=k)
        # Drop ground-truth noise label (-1) before scoring.
        mask = y >= 0
        ari = adjusted_rand_score(y[mask], res.labels[mask])
        print(
            f"{name:<14} ARI={ari:+.3f} k_found={res.extra['k_found']} (target k={k}) "
            f"modularity={res.extra['modularity']:+.3f} "
            f"reconcile={res.extra['merge_or_split']:<10} engine={res.extra['fallback']}"
        )
