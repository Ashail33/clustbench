"""Mutant: kmeans × meta_clusterer_v2 crossover (mutator-proposed algorithm).

Produced by the algorithm-space mutator in
:mod:`clustbench.algorithm_mutator` as the top-ranked random mutation
in the 200-iteration smoke search (seed=0). The mutator's prediction:

  predicted mean ARI 0.926, +0.138 over the best existing card
  (kmeans_trimmed at 0.788)

  Mechanism tags inherited from the crossover parents:
    fingerprint_dispatch  (from meta_clusterer_v2)
    kpp_init              (from kmeans)
    mean_centroid         (from kmeans)
    silhouette_probe      (from meta_clusterer_v2)

  Inductive biases (union of parents):
    convex_clusters + isotropic_clusters + needs_k + scales_linear
    (kmeans side)
    + non_convex_capable + meta + ensemble
    (meta_clusterer_v2 side)

Implementation: kmeans always runs as a cheap baseline; a fingerprint
probe (convexity ratio of the kmeans partition) + silhouette check
decides whether to keep the kmeans labels or hand off to
meta_clusterer_v2's full fingerprint-then-route pipeline.

Decision rule:
  - If conv_cv < 0.45 AND silhouette >= threshold (default 0.30):
      data is convex-shaped and kmeans found a coherent partition →
      return kmeans (fast path).
  - Otherwise: route through meta_clusterer_v2, which picks
      spectral / gmm / lmm / pwcc_diverse / etc. based on the
      fingerprint. kmeans labels are kept as a safe fallback if meta
      routing itself errors.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np

from .base import Algorithm, AlgoResult, Step, register


def _kpp_init(X: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    n = X.shape[0]
    first = int(rng.integers(n))
    centers = [X[first]]
    for _ in range(1, k):
        d2 = np.min(
            np.sum((X[:, None, :] - np.stack(centers, axis=0)[None, :, :]) ** 2, axis=2),
            axis=1,
        )
        total = d2.sum()
        idx = int(rng.integers(n)) if total <= 0 else int(rng.choice(n, p=d2 / total))
        centers.append(X[idx])
    return np.stack(centers, axis=0)


def _kmeans_em(X: np.ndarray, k: int, max_iter: int, tol: float, rng: np.random.Generator):
    centroids = _kpp_init(X, k, rng).astype(np.float32)
    prev_inertia: Optional[float] = None
    for _ in range(max_iter):
        D = np.sum((X[:, None, :] - centroids[None, :, :]) ** 2, axis=2)
        labels = D.argmin(axis=1)
        inertia = float(D[np.arange(X.shape[0]), labels].sum())
        new_centroids = centroids.copy()
        for j in range(k):
            members = X[labels == j]
            if len(members) > 0:
                new_centroids[j] = members.mean(axis=0)
        shift = float(np.linalg.norm(new_centroids - centroids))
        centroids = new_centroids
        if prev_inertia is not None and abs(inertia - prev_inertia) < tol:
            break
        if shift < tol:
            break
        prev_inertia = inertia
    D = np.sum((X[:, None, :] - centroids[None, :, :]) ** 2, axis=2)
    return D.argmin(axis=1), centroids, inertia


@register
class Mutant_kmeans_meta(Algorithm):
    """Hybrid kmeans + meta_clusterer_v2 (mutator-proposed crossover).

    Parameters
    ----------
    silhouette_threshold : float
        Default 0.30. If the kmeans probe's silhouette clears this AND
        the convexity ratio is below ``conv_cv_threshold``, return
        kmeans (fast path).
    conv_cv_threshold : float
        Default 0.45. Above this, the data is judged non-convex and
        meta routing fires regardless of silhouette. Matches
        meta_clusterer_v2's non-convex routing rule.
    max_iter / n_init : kmeans EM parameters.
    """

    def __init__(
        self,
        silhouette_threshold: float = 0.30,
        conv_cv_threshold: float = 0.35,
        max_iter: int = 100,
        n_init: int = 3,
        random_state: int = 42,
        **kwargs: Any,
    ) -> None:
        self.name = "mutant_kmeans_meta"
        self.silhouette_threshold = silhouette_threshold
        self.conv_cv_threshold = conv_cv_threshold
        self.max_iter = max_iter
        self.n_init = n_init
        self.random_state = random_state

    def fit_predict(self, X: np.ndarray, k: Optional[int] = None) -> AlgoResult:
        from sklearn.metrics import silhouette_score

        assert k is not None, "k required"
        rng = np.random.default_rng(self.random_state)
        n = X.shape[0]
        trajectory: list[Step] = []

        # Step 1: always run kmeans as the cheap baseline.
        best_labels = None
        best_inertia = float("inf")
        for _ in range(self.n_init):
            labels, _, inertia = _kmeans_em(X, k, self.max_iter, 1e-4, rng)
            if inertia < best_inertia:
                best_labels, best_inertia = labels, inertia

        # Step 2: fingerprint + silhouette probe on the kmeans partition.
        try:
            dist = np.linalg.norm(
                X - np.array([X[best_labels == j].mean(axis=0) if (best_labels == j).any()
                              else np.zeros(X.shape[1])
                              for j in range(k)])[best_labels],
                axis=1,
            )
            conv_cv = float(np.std(dist) / (np.mean(dist) + 1e-9))
        except Exception:
            conv_cv = 0.5
        try:
            uniq = len(set(int(v) for v in best_labels))
            if 1 < uniq < n:
                sil = float(silhouette_score(X, best_labels))
            else:
                sil = -1.0
        except Exception:
            sil = -1.0

        trajectory.append(
            Step(
                step_idx=0,
                cost=float(best_inertia),
                delta_cost=None,
                accepted=True,
                action={"type": "primary_kmeans", "n_init": self.n_init,
                        "conv_cv": conv_cv, "silhouette": sil},
                state={"k": int(k), "conv_cv": conv_cv, "sil": sil},
            )
        )

        # Step 3: decide path.
        take_fast_path = (
            conv_cv < self.conv_cv_threshold
            and sil >= self.silhouette_threshold
        )
        trajectory.append(
            Step(
                step_idx=1,
                cost=conv_cv,
                delta_cost=None,
                accepted=take_fast_path,
                action={
                    "type": "dispatch_decision",
                    "fast_path": take_fast_path,
                    "conv_cv": conv_cv,
                    "silhouette": sil,
                },
                state={"chose": "kmeans_primary" if take_fast_path else "meta_route"},
            )
        )

        if take_fast_path:
            return AlgoResult(
                labels=best_labels.astype(np.int64),
                extra={
                    "router": "mutant_kmeans_meta",
                    "chose": "kmeans_primary",
                    "silhouette": sil,
                    "conv_cv": conv_cv,
                    "fast_path": True,
                },
                trajectory=trajectory,
            )

        # Step 4: meta routing via meta_clusterer_v2.
        try:
            from .meta_clusterer_v2 import Meta_clusterer_v2

            mc = Meta_clusterer_v2()
            res = mc.fit_predict(X, k=k)
        except Exception as e:
            return AlgoResult(
                labels=best_labels.astype(np.int64),
                extra={
                    "router": "mutant_kmeans_meta",
                    "chose": "kmeans_primary",
                    "silhouette": sil,
                    "conv_cv": conv_cv,
                    "fast_path": False,
                    "meta_route_error": f"{type(e).__name__}: {str(e)[:100]}",
                },
                trajectory=trajectory,
            )

        if res.trajectory:
            for s in res.trajectory:
                trajectory.append(
                    Step(
                        step_idx=len(trajectory),
                        cost=s.cost,
                        delta_cost=s.delta_cost,
                        accepted=s.accepted,
                        action=s.action,
                        state=s.state,
                    )
                )

        return AlgoResult(
            labels=res.labels,
            extra={
                "router": "mutant_kmeans_meta",
                "chose": (res.extra or {}).get("chose", "meta_clusterer_v2"),
                "silhouette": sil,
                "conv_cv": conv_cv,
                "fast_path": False,
                **(res.extra or {}),
            },
            trajectory=trajectory,
        )
