"""Stage-1 RL environment for compositional clustering (Framing C).

This module defines:

1. A primitive-action ontology (8 actions) shared across clustering families.
2. A copy-on-write ``ClusteringState`` dataclass with a fixed-length feature
   vector for policy consumption.
3. A Gym-style ``ClusteringEnv`` that applies primitives, computes a normalised
   reward, and exposes a discretised legal-action enumeration.
4. ``collect_traces_from_existing_algorithms`` — a behavior-cloning data
   collector that translates trajectories from existing clustbench algorithms
   into this env's vocabulary and writes them to ``runs/rl_traces.parquet``.

The downstream policy/value training (stage 2) will import :class:`Action`,
:class:`ClusteringState`, and :class:`ClusteringEnv` from here.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field, replace
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, DBSCAN, KMeans
from sklearn.manifold import SpectralEmbedding
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors


# ---------------------------------------------------------------------------
# 1. Primitive action ontology
# ---------------------------------------------------------------------------


class Action(Enum):
    """The 8 primitive actions a policy can compose into a pipeline."""

    KPP_INIT = "kpp_init"
    ASSIGN_TO_CENTERS = "assign_to_centers"
    UPDATE_CENTERS = "update_centers"
    MEDOID_SWAP = "medoid_swap"
    WARD_MERGE = "ward_merge"
    DENSITY_PARTITION = "density_partition"
    EIGEN_EMBED = "eigen_embed"
    OUTLIER_TRIM = "outlier_trim"


# Per-action parameter schema. Used both for validation and for action_space
# enumeration. ``type`` is the python type; ``default`` is used when the
# policy emits a partial dict.
ACTION_PARAM_SCHEMA: Dict[Action, Dict[str, Dict[str, Any]]] = {
    Action.KPP_INIT: {
        "k": {"type": int, "default": None},
        "seed": {"type": int, "default": 0},
    },
    Action.ASSIGN_TO_CENTERS: {},
    Action.UPDATE_CENTERS: {},
    Action.MEDOID_SWAP: {
        "out_idx": {"type": int, "default": 0},
        "in_idx": {"type": int, "default": 0},
    },
    Action.WARD_MERGE: {},
    Action.DENSITY_PARTITION: {
        "eps": {"type": float, "default": 0.5},
        "min_samples": {"type": int, "default": 5},
    },
    Action.EIGEN_EMBED: {
        "n_components": {"type": int, "default": 2},
        "n_neighbors": {"type": int, "default": 10},
    },
    Action.OUTLIER_TRIM: {
        "q": {"type": float, "default": 0.05},
        "n_neighbors": {"type": int, "default": 20},
    },
}


_NOISE_LABEL = -1


# ---------------------------------------------------------------------------
# 2. ClusteringState
# ---------------------------------------------------------------------------


# How many features ``to_features()`` emits. Stage-2 policy network expects
# this constant — keep it stable across versions.
N_STATE_FEATURES = 20


@dataclass(frozen=True)
class ClusteringState:
    """Frozen, copy-on-write snapshot of the clustering pipeline state."""

    X: np.ndarray
    embedding: np.ndarray
    labels: np.ndarray
    centers: Optional[np.ndarray]
    k_target: int
    step_idx: int
    cost: float
    silhouette: float

    # Caching/bookkeeping. ``cost_initial`` lets reward normalisation be
    # stateless across env.step calls.
    cost_initial: float = 0.0

    def replace(self, **kwargs: Any) -> "ClusteringState":
        """Copy-on-write update — wraps dataclasses.replace."""
        return replace(self, **kwargs)

    # ------------------------------------------------------------------
    # Fixed-length feature vector
    # ------------------------------------------------------------------
    def to_features(self) -> np.ndarray:
        """Return a fixed-length feature vector (length ``N_STATE_FEATURES``).

        Never raises. Degenerate fields (single cluster, all noise, NaN cost,
        etc.) collapse to zero. The vector is what the stage-2 policy
        consumes; do *not* change its length or ordering without bumping
        ``N_STATE_FEATURES`` and retraining.
        """
        feats = np.zeros(N_STATE_FEATURES, dtype=np.float32)
        try:
            n = int(self.labels.shape[0])
            if n == 0:
                return feats

            uniq = np.unique(self.labels[self.labels != _NOISE_LABEL])
            n_clusters = int(uniq.size)
            outlier_frac = float(np.mean(self.labels == _NOISE_LABEL))

            feats[0] = float(self.step_idx)
            feats[1] = float(self.k_target)
            feats[2] = float(n_clusters)
            feats[3] = float(n_clusters) / max(1.0, float(self.k_target))
            feats[4] = outlier_frac
            feats[5] = float(self.silhouette) if np.isfinite(self.silhouette) else 0.0

            # Cost normalisation relative to initial cost.
            cost = float(self.cost) if np.isfinite(self.cost) else 0.0
            c0 = float(self.cost_initial) if self.cost_initial > 0 else 1.0
            feats[6] = cost / c0
            feats[7] = 1.0 if self.centers is not None else 0.0
            feats[8] = float(self.embedding.shape[1])

            # Cluster-size coefficient of variation.
            if n_clusters >= 1:
                sizes = np.array(
                    [int(np.sum(self.labels == lab)) for lab in uniq], dtype=np.float64
                )
                if sizes.sum() > 0:
                    mu = sizes.mean()
                    sd = sizes.std()
                    feats[9] = float(sd / mu) if mu > 0 else 0.0
                    feats[10] = float(sizes.min()) / max(1.0, float(sizes.max()))

            # Mean intra- and inter-cluster distance on the embedding.
            try:
                emb = self.embedding
                if n_clusters >= 2 and n <= 2000:
                    centroids = []
                    intras = []
                    for lab in uniq:
                        mask = self.labels == lab
                        pts = emb[mask]
                        if pts.shape[0] == 0:
                            continue
                        c = pts.mean(axis=0)
                        centroids.append(c)
                        if pts.shape[0] >= 2:
                            intras.append(float(np.mean(np.linalg.norm(pts - c, axis=1))))
                    if centroids:
                        C = np.stack(centroids, axis=0)
                        if intras:
                            feats[11] = float(np.mean(intras))
                        if C.shape[0] >= 2:
                            # pairwise centroid distances
                            d = np.linalg.norm(C[:, None, :] - C[None, :, :], axis=2)
                            iu = np.triu_indices(C.shape[0], k=1)
                            feats[12] = float(d[iu].mean()) if iu[0].size else 0.0
                            feats[13] = float(d[iu].min()) if iu[0].size else 0.0
            except Exception:
                pass

            # Embedding scale (max range across features) — informs whether
            # EIGEN_EMBED has fired.
            try:
                rng = float(np.ptp(self.embedding))
                feats[14] = rng
            except Exception:
                pass

            # Booleans helpful for the policy.
            feats[15] = 1.0 if n_clusters == self.k_target else 0.0
            feats[16] = 1.0 if n_clusters > self.k_target else 0.0
            feats[17] = 1.0 if n_clusters < self.k_target else 0.0
            feats[18] = float(n)
            feats[19] = float(self.X.shape[1])
        except Exception:
            return np.zeros(N_STATE_FEATURES, dtype=np.float32)
        # Clean NaNs/infs that may slip through.
        feats = np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)
        return feats.astype(np.float32)


# ---------------------------------------------------------------------------
# Helpers — silhouette, cost
# ---------------------------------------------------------------------------


def _safe_silhouette(X: np.ndarray, labels: np.ndarray) -> float:
    """Silhouette wrapper that returns NaN on degenerate inputs.

    sklearn raises if there's only one cluster or every point is noise.
    Returning NaN keeps the env total-ordering simple — the policy never
    sees a stack trace.
    """
    try:
        # Treat -1 as its own bucket for silhouette, but if everything is
        # one effective cluster, bail out.
        if labels.shape[0] < 3:
            return float("nan")
        non_noise_mask = labels != _NOISE_LABEL
        if non_noise_mask.sum() < 3:
            return float("nan")
        sub_X = X[non_noise_mask]
        sub_lab = labels[non_noise_mask]
        uniq = np.unique(sub_lab)
        if uniq.size < 2:
            return float("nan")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return float(silhouette_score(sub_X, sub_lab))
    except Exception:
        return float("nan")


def _inertia_from_labels_centers(
    X: np.ndarray, labels: np.ndarray, centers: Optional[np.ndarray]
) -> float:
    """Compute the inertia (sum sq distance to assigned center).

    When ``centers`` is None we fall back to per-cluster means computed
    from labels. Noise points (label=-1) are excluded.
    """
    if X.shape[0] == 0:
        return 0.0
    mask = labels != _NOISE_LABEL
    if mask.sum() == 0:
        return 0.0
    Xm = X[mask]
    lm = labels[mask]
    uniq = np.unique(lm)
    if centers is None or centers.shape[0] < uniq.size:
        # Recompute centers from labels.
        centers_local = np.stack(
            [Xm[lm == lab].mean(axis=0) for lab in uniq], axis=0
        )
        # Remap labels to row-index in centers_local.
        lab_to_row = {int(lab): i for i, lab in enumerate(uniq)}
        rows = np.array([lab_to_row[int(l)] for l in lm])
        diff = Xm - centers_local[rows]
    else:
        # ``centers`` indexed by label value.
        # Defensive: clamp out-of-range labels.
        valid = lm < centers.shape[0]
        if valid.sum() == 0:
            return 0.0
        Xm = Xm[valid]
        lm = lm[valid]
        diff = Xm - centers[lm]
    return float(np.sum(diff * diff))


# ---------------------------------------------------------------------------
# 3. Primitive action implementations
# ---------------------------------------------------------------------------


def _kpp_once(X: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    """One k-means++ initialisation pass — picks ``k`` rows of ``X``."""
    n = X.shape[0]
    first = int(rng.integers(n))
    centers = [X[first]]
    for _ in range(1, k):
        C = np.stack(centers, axis=0)
        d2 = np.min(
            np.sum((X[:, None, :] - C[None, :, :]) ** 2, axis=2), axis=1
        )
        total = float(d2.sum())
        if total <= 0:
            idx = int(rng.integers(n))
        else:
            probs = d2 / total
            idx = int(rng.choice(n, p=probs))
        centers.append(X[idx])
    return np.stack(centers, axis=0)


def _act_kpp_init(state: ClusteringState, params: Dict[str, Any]) -> ClusteringState:
    """k-means++ centroid initialisation on the current embedding.

    Runs several random restarts internally (like sklearn's ``n_init``) so a
    single KPP_INIT action gives a robust initial set rather than one
    coin-flip on the seed. The best init by initial assignment inertia
    wins.
    """
    k = int(params.get("k", state.k_target) or state.k_target)
    seed = int(params.get("seed", 0))
    n_restarts = int(params.get("n_restarts", 5))
    X = state.embedding
    n = X.shape[0]
    if n == 0 or k <= 0:
        return state.replace()
    k = min(k, n)
    rng = np.random.default_rng(seed)
    best_centers = None
    best_cost = np.inf
    for _ in range(max(1, n_restarts)):
        centers = _kpp_once(X, k, rng)
        # Initial assignment inertia under these centers.
        D = np.sum((X[:, None, :] - centers[None, :, :]) ** 2, axis=2)
        cost = float(D.min(axis=1).sum())
        if cost < best_cost:
            best_cost = cost
            best_centers = centers
    new_centers = best_centers.astype(np.float32)
    return state.replace(centers=new_centers)


def _act_assign_to_centers(
    state: ClusteringState, params: Dict[str, Any]
) -> ClusteringState:
    """Recompute labels as argmin distance to current centers."""
    if state.centers is None or state.centers.shape[0] == 0:
        return state.replace()
    X = state.embedding
    C = state.centers
    D = np.sum((X[:, None, :] - C[None, :, :]) ** 2, axis=2)
    labels = D.argmin(axis=1).astype(np.int64)
    return state.replace(labels=labels)


def _act_update_centers(
    state: ClusteringState, params: Dict[str, Any]
) -> ClusteringState:
    """Recompute centers as the mean of points assigned to each label."""
    X = state.embedding
    labels = state.labels
    mask = labels != _NOISE_LABEL
    if mask.sum() == 0:
        return state.replace()
    if state.centers is not None:
        k = state.centers.shape[0]
    else:
        k = max(int(labels[mask].max()) + 1, state.k_target)
    new_centers = np.zeros((k, X.shape[1]), dtype=np.float32)
    for j in range(k):
        members = X[(labels == j) & mask]
        if members.shape[0] > 0:
            new_centers[j] = members.mean(axis=0)
        elif state.centers is not None and j < state.centers.shape[0]:
            # Preserve previous center if no members (empty cluster).
            new_centers[j] = state.centers[j]
    return state.replace(centers=new_centers)


def _act_medoid_swap(
    state: ClusteringState, params: Dict[str, Any]
) -> ClusteringState:
    """CLARANS-style medoid swap. ``out_idx`` is the medoid row in ``centers``,
    ``in_idx`` is the data row to promote. Out-of-range indices no-op.
    """
    if state.centers is None or state.centers.shape[0] == 0:
        return state.replace()
    out_idx = int(params.get("out_idx", 0))
    in_idx = int(params.get("in_idx", 0))
    n = state.embedding.shape[0]
    k = state.centers.shape[0]
    if not (0 <= out_idx < k and 0 <= in_idx < n):
        return state.replace()
    new_centers = state.centers.copy()
    new_centers[out_idx] = state.embedding[in_idx]
    return state.replace(centers=new_centers.astype(np.float32))


def _act_ward_merge(
    state: ClusteringState, params: Dict[str, Any]
) -> ClusteringState:
    """Merge the two clusters whose centroid distance is smallest.

    Implemented as a direct centroid merge rather than re-running scipy's
    full linkage — keeps the action O(k^2) and locally reversible.
    """
    labels = state.labels.copy()
    mask = labels != _NOISE_LABEL
    if mask.sum() == 0:
        return state.replace()
    uniq = np.unique(labels[mask])
    if uniq.size <= 1:
        return state.replace()
    # Compute centroids per cluster on the embedding.
    centroids = np.stack(
        [state.embedding[labels == lab].mean(axis=0) for lab in uniq], axis=0
    )
    d = np.sum((centroids[:, None, :] - centroids[None, :, :]) ** 2, axis=2)
    np.fill_diagonal(d, np.inf)
    i, j = np.unravel_index(np.argmin(d), d.shape)
    keep = int(uniq[min(i, j)])
    drop = int(uniq[max(i, j)])
    labels[labels == drop] = keep
    return state.replace(labels=labels, centers=None)


def _act_density_partition(
    state: ClusteringState, params: Dict[str, Any]
) -> ClusteringState:
    """DBSCAN-style density-connected partition.

    ``eps`` defaults to a heuristic 80th-percentile of k-nearest-neighbour
    distances when not provided. The new labels overwrite the previous
    ones; centers are dropped because density components aren't centroidal.
    """
    X = state.embedding
    if X.shape[0] == 0:
        return state.replace()
    min_samples = int(params.get("min_samples", 5))
    eps = params.get("eps", None)
    if eps is None or float(eps) <= 0:
        # Heuristic eps based on k-distance graph.
        try:
            kk = min(min_samples, X.shape[0] - 1)
            if kk < 1:
                return state.replace()
            nn = NearestNeighbors(n_neighbors=kk + 1).fit(X)
            dists, _ = nn.kneighbors(X)
            eps = float(np.percentile(dists[:, -1], 80))
            if eps <= 0:
                eps = 0.5
        except Exception:
            eps = 0.5
    else:
        eps = float(eps)
    try:
        db = DBSCAN(eps=eps, min_samples=min_samples).fit(X)
        labels = db.labels_.astype(np.int64)
    except Exception:
        return state.replace()
    return state.replace(labels=labels, centers=None)


def _act_eigen_embed(
    state: ClusteringState, params: Dict[str, Any]
) -> ClusteringState:
    """Replace the embedding with the leading eigenvectors of the kNN
    Laplacian — the spectral-clustering preconditioning step.
    """
    X = state.embedding
    n = X.shape[0]
    if n < 3:
        return state.replace()
    n_components = int(params.get("n_components", 2))
    n_neighbors = int(params.get("n_neighbors", 10))
    n_components = max(1, min(n_components, n - 1))
    n_neighbors = max(2, min(n_neighbors, n - 1))
    # Idempotency guard: if the embedding is already low-dim and has
    # ~unit-norm rows (the canonical output of EIGEN_EMBED), repeating the
    # action would only re-spectralise noise. Treat as a no-op so the
    # spec's ``EIGEN_EMBED × 3`` pattern remains stable.
    if X.shape[1] <= max(n_components, 4):
        row_norms = np.linalg.norm(X, axis=1)
        if row_norms.size and np.allclose(row_norms, 1.0, atol=1e-3):
            return state.replace()
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            se = SpectralEmbedding(
                n_components=n_components,
                affinity="nearest_neighbors",
                n_neighbors=n_neighbors,
                random_state=0,
            )
            new_emb = se.fit_transform(X).astype(np.float32)
        # Row-normalise (Ng-Jordan-Weiss): unit-length rows make k-means
        # on the embedding equivalent to standard normalised spectral
        # clustering. Without this, k-means can land on degenerate splits.
        norms = np.linalg.norm(new_emb, axis=1, keepdims=True)
        norms = np.where(norms < 1e-12, 1.0, norms)
        new_emb = (new_emb / norms).astype(np.float32)
    except Exception:
        return state.replace()
    # Embedding changed -> centers are stale.
    return state.replace(embedding=new_emb, centers=None)


def _act_outlier_trim(
    state: ClusteringState, params: Dict[str, Any]
) -> ClusteringState:
    """Mark the top-q LOF-scoring points as outliers (label = -1)."""
    X = state.embedding
    n = X.shape[0]
    if n < 5:
        return state.replace()
    q = float(params.get("q", 0.05))
    q = float(np.clip(q, 0.0, 0.5))
    if q <= 0.0:
        return state.replace()
    n_neighbors = int(params.get("n_neighbors", 20))
    n_neighbors = max(2, min(n_neighbors, n - 1))
    try:
        lof = LocalOutlierFactor(n_neighbors=n_neighbors)
        lof.fit_predict(X)
        scores = -lof.negative_outlier_factor_  # higher = more anomalous
    except Exception:
        return state.replace()
    n_trim = max(1, int(round(q * n)))
    thresh_idx = np.argsort(scores)[-n_trim:]
    labels = state.labels.copy()
    labels[thresh_idx] = _NOISE_LABEL
    return state.replace(labels=labels)


PRIMITIVE_FUNCS = {
    Action.KPP_INIT: _act_kpp_init,
    Action.ASSIGN_TO_CENTERS: _act_assign_to_centers,
    Action.UPDATE_CENTERS: _act_update_centers,
    Action.MEDOID_SWAP: _act_medoid_swap,
    Action.WARD_MERGE: _act_ward_merge,
    Action.DENSITY_PARTITION: _act_density_partition,
    Action.EIGEN_EMBED: _act_eigen_embed,
    Action.OUTLIER_TRIM: _act_outlier_trim,
}


# ---------------------------------------------------------------------------
# 4. ClusteringEnv
# ---------------------------------------------------------------------------


@dataclass
class _EpisodeBookkeeping:
    """Internal scratch used by ClusteringEnv across step() calls."""

    recent_deltas: List[float] = field(default_factory=list)
    max_steps: int = 30
    plateau_window: int = 3
    plateau_threshold: float = 1e-4
    silhouette_terminal: float = 0.95


class ClusteringEnv:
    """Gym-style environment for compositional clustering policies."""

    def __init__(
        self,
        max_steps: int = 30,
        plateau_window: int = 3,
        plateau_threshold: float = 1e-4,
        silhouette_terminal: float = 0.95,
        n_medoid_candidates: int = 10,
    ) -> None:
        self.max_steps = max_steps
        self.plateau_window = plateau_window
        self.plateau_threshold = plateau_threshold
        self.silhouette_terminal = silhouette_terminal
        self.n_medoid_candidates = n_medoid_candidates
        self._book: Optional[_EpisodeBookkeeping] = None
        self._state: Optional[ClusteringState] = None
        self._rng: Optional[np.random.Generator] = None

    # ------------------------------------------------------------------
    def reset(
        self,
        X: np.ndarray,
        k_target: int,
        seed: int = 0,
    ) -> ClusteringState:
        """Start a fresh episode.

        Initial labels are random in ``[0, k_target)`` so the initial cost
        is well-defined and reward normalisation is stable.
        """
        rng = np.random.default_rng(seed)
        self._rng = rng
        X = np.asarray(X, dtype=np.float32)
        n = X.shape[0]
        labels = rng.integers(0, max(1, k_target), size=n).astype(np.int64)
        cost0 = _inertia_from_labels_centers(X, labels, None)
        sil = _safe_silhouette(X, labels)
        state = ClusteringState(
            X=X,
            embedding=X,
            labels=labels,
            centers=None,
            k_target=int(k_target),
            step_idx=0,
            cost=cost0,
            silhouette=sil,
            cost_initial=max(cost0, 1e-9),
        )
        self._state = state
        self._book = _EpisodeBookkeeping(
            max_steps=self.max_steps,
            plateau_window=self.plateau_window,
            plateau_threshold=self.plateau_threshold,
            silhouette_terminal=self.silhouette_terminal,
        )
        return state

    # ------------------------------------------------------------------
    def step(
        self, action: Action, params: Optional[Dict[str, Any]] = None
    ) -> Tuple[ClusteringState, float, bool, Dict[str, Any]]:
        """Apply ``action`` with ``params``; return (next_state, reward, done, info)."""
        assert self._state is not None and self._book is not None, "Call reset() first."
        params = dict(params or {})
        prev = self._state
        fn = PRIMITIVE_FUNCS[action]
        try:
            after = fn(prev, params)
        except Exception as e:
            # Catastrophic primitive failure: keep the state, mild penalty.
            info = {"error": str(e), "action": action.value, "params": params}
            new_state = prev.replace(step_idx=prev.step_idx + 1)
            self._state = new_state
            return new_state, -0.1, self._is_done(new_state), info

        # Recompute cost & silhouette on the new state.
        new_cost = _inertia_from_labels_centers(
            after.embedding, after.labels, after.centers
        )
        new_sil = _safe_silhouette(after.embedding, after.labels)
        new_state = after.replace(
            step_idx=prev.step_idx + 1,
            cost=new_cost,
            silhouette=new_sil,
        )

        delta_cost = new_cost - prev.cost
        c0 = max(prev.cost_initial, 1e-9)
        reward = float(-delta_cost / c0)
        reward = float(np.clip(reward, -1.0, 1.0))

        # Plateau bookkeeping.
        self._book.recent_deltas.append(abs(delta_cost) / c0)
        if len(self._book.recent_deltas) > self._book.plateau_window:
            self._book.recent_deltas.pop(0)

        done = self._is_done(new_state)
        info = {
            "delta_cost": float(delta_cost),
            "cost": float(new_cost),
            "silhouette": float(new_sil) if np.isfinite(new_sil) else None,
            "action": action.value,
            "params": params,
            "terminal_reason": self._terminal_reason(new_state) if done else None,
        }
        self._state = new_state
        return new_state, reward, done, info

    # ------------------------------------------------------------------
    def _is_done(self, state: ClusteringState) -> bool:
        if state.step_idx >= self._book.max_steps:
            return True
        if (
            np.isfinite(state.silhouette)
            and state.silhouette > self._book.silhouette_terminal
        ):
            return True
        if len(self._book.recent_deltas) >= self._book.plateau_window:
            if sum(self._book.recent_deltas) < self._book.plateau_threshold:
                return True
        return False

    def _terminal_reason(self, state: ClusteringState) -> str:
        if state.step_idx >= self._book.max_steps:
            return "max_steps"
        if (
            np.isfinite(state.silhouette)
            and state.silhouette > self._book.silhouette_terminal
        ):
            return "silhouette"
        return "plateau"

    # ------------------------------------------------------------------
    def action_space(
        self, state: ClusteringState
    ) -> List[Tuple[Action, Dict[str, Any]]]:
        """Enumerate currently-legal (action, params) tuples.

        For continuous-parameter actions like medoid_swap we sample
        ``n_medoid_candidates`` discrete options so the policy chooses
        from a fixed discrete head.
        """
        rng = self._rng if self._rng is not None else np.random.default_rng(0)
        out: List[Tuple[Action, Dict[str, Any]]] = []
        n = state.embedding.shape[0]
        n_clusters = int(np.unique(state.labels[state.labels != _NOISE_LABEL]).size)
        has_centers = state.centers is not None and state.centers.shape[0] > 0

        # KPP_INIT — always legal.
        out.append((Action.KPP_INIT, {"k": state.k_target, "seed": int(rng.integers(1 << 30))}))

        # ASSIGN_TO_CENTERS — legal iff centers exist.
        if has_centers:
            out.append((Action.ASSIGN_TO_CENTERS, {}))

        # UPDATE_CENTERS — legal iff we have at least one non-noise point.
        if (state.labels != _NOISE_LABEL).any():
            out.append((Action.UPDATE_CENTERS, {}))

        # MEDOID_SWAP — legal iff centers exist and n > 1. Sample candidates.
        if has_centers and n > 1:
            k = state.centers.shape[0]
            for _ in range(self.n_medoid_candidates):
                out.append(
                    (
                        Action.MEDOID_SWAP,
                        {
                            "out_idx": int(rng.integers(k)),
                            "in_idx": int(rng.integers(n)),
                        },
                    )
                )

        # WARD_MERGE — legal iff >= 2 clusters.
        if n_clusters >= 2:
            out.append((Action.WARD_MERGE, {}))

        # DENSITY_PARTITION — always legal; params discretised over a small grid.
        for eps in (0.3, 0.6, 1.0):
            out.append(
                (Action.DENSITY_PARTITION, {"eps": eps, "min_samples": 5})
            )

        # EIGEN_EMBED — legal if n >= 5. Two component options.
        if n >= 5:
            for nc in (2, max(2, state.k_target)):
                out.append(
                    (
                        Action.EIGEN_EMBED,
                        {"n_components": nc, "n_neighbors": min(10, n - 1)},
                    )
                )

        # OUTLIER_TRIM — legal if n >= 5.
        if n >= 5:
            for q in (0.02, 0.05, 0.10):
                out.append((Action.OUTLIER_TRIM, {"q": q, "n_neighbors": min(20, n - 1)}))

        return out


# ---------------------------------------------------------------------------
# 5. Behavior-cloning trace collection
# ---------------------------------------------------------------------------


def _trace_kmeans(env: ClusteringEnv, X: np.ndarray, k: int, seed: int) -> List[Dict[str, Any]]:
    """Translate a kmeans run into env actions and record (state, action, reward)."""
    env.reset(X, k_target=k, seed=seed)
    rows: List[Dict[str, Any]] = []
    # KPP init then 5 EM rounds.
    plan: List[Tuple[Action, Dict[str, Any]]] = [
        (Action.KPP_INIT, {"k": k, "seed": seed}),
    ]
    for _ in range(5):
        plan.append((Action.ASSIGN_TO_CENTERS, {}))
        plan.append((Action.UPDATE_CENTERS, {}))
    for action, params in plan:
        state_before = env._state
        feats = state_before.to_features()
        _, reward, done, info = env.step(action, params)
        rows.append(
            {
                "action": action.value,
                "params": params,
                "state_features": feats.tolist(),
                "reward": float(reward),
                "step_idx": state_before.step_idx,
            }
        )
        if done:
            break
    return rows


def _trace_clarans(env: ClusteringEnv, X: np.ndarray, k: int, seed: int) -> List[Dict[str, Any]]:
    env.reset(X, k_target=k, seed=seed)
    rng = np.random.default_rng(seed)
    rows: List[Dict[str, Any]] = []
    plan: List[Tuple[Action, Dict[str, Any]]] = [
        (Action.KPP_INIT, {"k": k, "seed": seed}),
        (Action.ASSIGN_TO_CENTERS, {}),
    ]
    for _ in range(8):
        plan.append(
            (
                Action.MEDOID_SWAP,
                {"out_idx": int(rng.integers(k)), "in_idx": int(rng.integers(X.shape[0]))},
            )
        )
        plan.append((Action.ASSIGN_TO_CENTERS, {}))
    for action, params in plan:
        state_before = env._state
        feats = state_before.to_features()
        _, reward, done, info = env.step(action, params)
        rows.append(
            {
                "action": action.value,
                "params": params,
                "state_features": feats.tolist(),
                "reward": float(reward),
                "step_idx": state_before.step_idx,
            }
        )
        if done:
            break
    return rows


def _trace_agglomerative(env: ClusteringEnv, X: np.ndarray, k: int, seed: int) -> List[Dict[str, Any]]:
    """Use sklearn ward to seed an over-clustered partition, then issue
    ``(n_components - k)`` WARD_MERGE actions inside the env."""
    env.reset(X, k_target=k, seed=seed)
    rows: List[Dict[str, Any]] = []
    n = X.shape[0]
    # Seed: agglomerative to (k + a few) clusters via direct sklearn call,
    # then merge down via env actions.
    seed_k = min(n, k + 4)
    try:
        agg = AgglomerativeClustering(n_clusters=seed_k, linkage="ward")
        seed_labels = agg.fit_predict(X).astype(np.int64)
    except Exception:
        seed_labels = np.zeros(n, dtype=np.int64)
    # Inject the seed labels directly — this is the env-state precondition
    # that lets WARD_MERGE be a meaningful sequence.
    env._state = env._state.replace(labels=seed_labels)
    n_merges = max(0, seed_k - k)
    for _ in range(n_merges):
        state_before = env._state
        feats = state_before.to_features()
        _, reward, done, info = env.step(Action.WARD_MERGE, {})
        rows.append(
            {
                "action": Action.WARD_MERGE.value,
                "params": {},
                "state_features": feats.tolist(),
                "reward": float(reward),
                "step_idx": state_before.step_idx,
            }
        )
        if done:
            break
    return rows


def _trace_spectral(env: ClusteringEnv, X: np.ndarray, k: int, seed: int) -> List[Dict[str, Any]]:
    env.reset(X, k_target=k, seed=seed)
    rows: List[Dict[str, Any]] = []
    plan: List[Tuple[Action, Dict[str, Any]]] = [
        (Action.EIGEN_EMBED, {"n_components": max(2, k), "n_neighbors": min(10, X.shape[0] - 1)}),
        (Action.KPP_INIT, {"k": k, "seed": seed}),
    ]
    for _ in range(3):
        plan.append((Action.ASSIGN_TO_CENTERS, {}))
        plan.append((Action.UPDATE_CENTERS, {}))
    for action, params in plan:
        state_before = env._state
        feats = state_before.to_features()
        _, reward, done, info = env.step(action, params)
        rows.append(
            {
                "action": action.value,
                "params": params,
                "state_features": feats.tolist(),
                "reward": float(reward),
                "step_idx": state_before.step_idx,
            }
        )
        if done:
            break
    return rows


def _trace_dbscan(env: ClusteringEnv, X: np.ndarray, k: int, seed: int) -> List[Dict[str, Any]]:
    env.reset(X, k_target=k, seed=seed)
    rows: List[Dict[str, Any]] = []
    state_before = env._state
    feats = state_before.to_features()
    _, reward, done, info = env.step(Action.DENSITY_PARTITION, {"min_samples": 5})
    rows.append(
        {
            "action": Action.DENSITY_PARTITION.value,
            "params": {"min_samples": 5},
            "state_features": feats.tolist(),
            "reward": float(reward),
            "step_idx": state_before.step_idx,
        }
    )
    return rows


# Algorithm-name -> trace builder.
_TRACE_BUILDERS = {
    "kmeans": _trace_kmeans,
    "clarans": _trace_clarans,
    "agglomerative": _trace_agglomerative,
    "spectral": _trace_spectral,
    "dbscan": _trace_dbscan,
}


def collect_traces_from_existing_algorithms(
    datasets: Sequence[Tuple[str, np.ndarray, int]],
    algo_names: Sequence[str],
    max_episodes: int = 50,
    out_path: str | Path = "runs/rl_traces.parquet",
    seed: int = 0,
) -> Path:
    """Generate behavior-cloning rows by replaying existing algorithms in the env.

    Parameters
    ----------
    datasets
        Sequence of ``(name, X, k)`` triples.
    algo_names
        Algorithms to replay. Supported: ``kmeans``, ``clarans``,
        ``agglomerative``, ``spectral``, ``dbscan``. Unknown names are
        ignored with a warning.
    max_episodes
        Hard cap on the total (dataset x algo) pairs we record.
    out_path
        Parquet destination. Parent directory is created if missing.

    Returns the path of the written parquet file.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    env = ClusteringEnv()
    rows: List[Dict[str, Any]] = []
    episode_id = 0
    for ds_name, X, k in datasets:
        for algo_name in algo_names:
            if episode_id >= max_episodes:
                break
            builder = _TRACE_BUILDERS.get(algo_name)
            if builder is None:
                warnings.warn(f"No trace builder for algo '{algo_name}'; skipping.")
                continue
            try:
                ep_rows = builder(env, X, int(k), seed + episode_id)
            except Exception as e:
                warnings.warn(f"Trace builder failed for {algo_name} on {ds_name}: {e}")
                continue
            for r in ep_rows:
                rows.append(
                    {
                        "episode_id": int(episode_id),
                        "dataset": str(ds_name),
                        "algo": str(algo_name),
                        "step_idx": int(r["step_idx"]),
                        "action": str(r["action"]),
                        "params": str(r["params"]),  # stringified for stable parquet schema
                        "state_features": np.asarray(r["state_features"], dtype=np.float32).tolist(),
                        "reward": float(r["reward"]),
                    }
                )
            episode_id += 1
    df = pd.DataFrame(rows)
    if df.empty:
        # Still write an empty parquet with the correct schema so downstream
        # readers don't crash.
        df = pd.DataFrame(
            {
                "episode_id": pd.Series([], dtype="int64"),
                "dataset": pd.Series([], dtype="object"),
                "algo": pd.Series([], dtype="object"),
                "step_idx": pd.Series([], dtype="int64"),
                "action": pd.Series([], dtype="object"),
                "params": pd.Series([], dtype="object"),
                "state_features": pd.Series([], dtype="object"),
                "reward": pd.Series([], dtype="float64"),
            }
        )
    df.to_parquet(out_path, index=False)
    return out_path


# ---------------------------------------------------------------------------
# 6. Smoke tests (inline, runnable via ``python -m clustbench.rl_env``)
# ---------------------------------------------------------------------------


def _smoke() -> Dict[str, Any]:
    """Run the spec-required smoke tests. Returns a dict of results."""
    from .datasets import DataSpec, gen_circles, gen_mdcgen

    out: Dict[str, Any] = {}

    # 1+3. kmeans-equivalent on mdcgen blobs.
    X, y = gen_mdcgen(DataSpec(n_samples=300, n_features=8, centers=3, compactness=1.0, seed=1))
    env = ClusteringEnv(max_steps=40)
    env.reset(X, k_target=3, seed=0)
    # n_restarts=10 mirrors sklearn's default; ensures the smoke is robust
    # to seed-dependent unlucky inits even on the size-imbalanced cluster.
    plan: List[Tuple[Action, Dict[str, Any]]] = [
        (Action.KPP_INIT, {"k": 3, "seed": 0, "n_restarts": 10})
    ]
    for _ in range(5):
        plan.append((Action.ASSIGN_TO_CENTERS, {}))
        plan.append((Action.UPDATE_CENTERS, {}))
    for a, p in plan:
        env.step(a, p)
    ari_kmeans = float(adjusted_rand_score(y, env._state.labels))
    out["kmeans_ari"] = ari_kmeans

    # 4. spectral-equivalent on circles.
    Xc, yc = gen_circles(DataSpec(n_samples=300, n_features=2, centers=2, compactness=0.5, seed=2))
    env2 = ClusteringEnv(max_steps=40)
    env2.reset(Xc, k_target=2, seed=3)
    plan2: List[Tuple[Action, Dict[str, Any]]] = []
    for _ in range(3):
        plan2.append((Action.EIGEN_EMBED, {"n_components": 2, "n_neighbors": 10}))
        plan2.append((Action.KPP_INIT, {"k": 2, "seed": 3}))
        plan2.append((Action.ASSIGN_TO_CENTERS, {}))
        plan2.append((Action.UPDATE_CENTERS, {}))
    for a, p in plan2:
        env2.step(a, p)
    ari_spectral = float(adjusted_rand_score(yc, env2._state.labels))
    out["spectral_ari"] = ari_spectral

    # 5. trace collection on 5 algos × 10 tasks.
    datasets = []
    for i in range(10):
        Xi, _ = gen_mdcgen(
            DataSpec(n_samples=200, n_features=4, centers=3, compactness=1.0, seed=10 + i)
        )
        datasets.append((f"mdcgen_{i}", Xi, 3))
    out_path = collect_traces_from_existing_algorithms(
        datasets,
        algo_names=["kmeans", "clarans", "agglomerative", "spectral", "dbscan"],
        max_episodes=50,
        out_path="runs/rl_traces.parquet",
    )
    df = pd.read_parquet(out_path)
    out["parquet_path"] = str(out_path)
    out["parquet_rows"] = int(len(df))
    out["parquet_columns"] = list(df.columns)
    out["parquet_dtypes"] = {c: str(df[c].dtype) for c in df.columns}
    out["parquet_episode_ids"] = int(df["episode_id"].nunique()) if len(df) else 0
    return out


if __name__ == "__main__":
    res = _smoke()
    print(res)
