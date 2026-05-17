"""Structured representations of every registered clustering algorithm.

The benchmark gives us empirical (task, algorithm) -> metric tables. The
analysis doc walks through each algorithm's mechanism in prose. This
module turns that prose into a *machine-readable* card per algorithm,
so downstream code can:

  1. Compute theoretical wall-time / memory predictions from the
     algorithm's complexity expression evaluated at the task's
     (n, d, k) — see :func:`predict_performance`.
  2. Score the algorithm against a data fingerprint by mechanism
     compatibility — see :func:`predict_ari_upper_bound`. The "upper
     bound" is the ARI we'd expect if the algorithm's inductive
     biases match the data; the actual ARI may be lower due to
     hyperparameter brittleness etc.
  3. Search algorithm-space by composing or mutating cards
     (future :mod:`algorithm_search`).

A card captures:

- ``name``: registry key
- ``family``: taxonomy bucket (centroid, hierarchical, density, graph,
  EM, probabilistic-mixture, ensemble, meta)
- ``mechanism_tags``: small set of textual tags identifying the
  algorithm's structural primitives
- ``inductive_biases``: another small set; what the algorithm
  *assumes* about the data
- ``time_complexity`` / ``memory_complexity``: callable mapping
  (n, d, k) to a float "rough op count" or "rough bytes"
- ``hyperparameters``: dict of (name, range / set) describing the
  searchable space
- ``handles_k_directly``: bool — True if k is a known input, False for
  density-based algos that infer k

Cards are *handwritten* — they encode our understanding of each
algorithm; theoretical predictions calibrated from the benchmark
multiply that understanding by an empirical constant per card.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set


# Match the inductive-bias vocabulary to the ALGORITHM_ANALYSIS axes.
# These are the labels a fingerprint of (data, algorithm) can match against.
ALL_BIASES = {
    "convex_clusters",         # Voronoi-like; fails on non-convex
    "isotropic_clusters",      # spherical / equal-variance assumption
    "non_convex_capable",      # graph Laplacian / density / mode-seeking
    "outlier_robust",          # posterior-weighted / trimmed / medoid
    "subspace_structured",     # low-rank-in-high-d / sparse SC
    "scales_linear",           # O(n)-ish per iteration
    "scales_sublinear",        # n-independent inner work (mini-batch)
    "scales_quadratic",        # O(n^2) eigen / pairwise / agglomerative
    "needs_k",                 # k is a required parameter
    "discovers_k",             # algorithm chooses k internally
    "hyperparameter_brittle",  # known sensitive to a single parameter
    "ensemble",                # combines multiple base algos
    "meta",                    # dispatches to other algos
    "learned",                 # trained from past benchmark data
}


@dataclass
class AlgorithmCard:
    name: str
    family: str
    mechanism_tags: Set[str] = field(default_factory=set)
    inductive_biases: Set[str] = field(default_factory=set)
    time_complexity: Callable[[int, int, int], float] = field(
        default=lambda n, d, k: n * d * (k or 1)
    )
    memory_complexity: Callable[[int, int, int], float] = field(
        default=lambda n, d, k: n * d
    )
    handles_k_directly: bool = True
    hyperparameters: Dict[str, Any] = field(default_factory=dict)
    notes: str = ""

    def predict_wall_time(self, n: int, d: int, k: Optional[int],
                          calibration: float = 1e-8) -> float:
        """Predict wall time in seconds.

        ``calibration`` is the per-op overhead — multiplied by the
        algorithm's op count from ``time_complexity``. The default
        1e-8 s/op is roughly tuned to a numpy / sklearn CPU loop;
        per-algorithm calibration constants come from
        :func:`calibrate_from_benchmark`.
        """
        ops = self.time_complexity(n, d, max(1, k or 1))
        return max(1e-6, float(calibration) * float(ops))

    def predict_memory_mb(self, n: int, d: int, k: Optional[int],
                          calibration: float = 8e-6) -> float:
        """Predict memory delta in MB. ``calibration`` units: MB/elem."""
        bytes_est = self.memory_complexity(n, d, max(1, k or 1))
        return max(0.001, float(calibration) * float(bytes_est))


# ---------------------------------------------------------------------------
# Card registry — one entry per registered algorithm.
# ---------------------------------------------------------------------------

ALGORITHM_CARDS: Dict[str, AlgorithmCard] = {}


def _card(name, **kwargs) -> AlgorithmCard:
    card = AlgorithmCard(name=name, **kwargs)
    ALGORITHM_CARDS[name] = card
    return card


# --- Centroid / k-means family ----------------------------------------------

_card("kmeans",
      family="centroid",
      mechanism_tags={"em_update", "kpp_init", "mean_centroid"},
      inductive_biases={"convex_clusters", "isotropic_clusters", "needs_k", "scales_linear"},
      time_complexity=lambda n, d, k: 10 * n * d * k,  # ~10 iters
      memory_complexity=lambda n, d, k: n * d + k * d,
      hyperparameters={"max_iter": (10, 500), "n_init": (1, 20)})

_card("minibatch_kmeans",
      family="centroid",
      mechanism_tags={"online_update", "minibatch", "mean_centroid"},
      inductive_biases={"convex_clusters", "isotropic_clusters", "needs_k", "scales_sublinear"},
      time_complexity=lambda n, d, k: 5 * n * d,
      memory_complexity=lambda n, d, k: 200 * d + k * d,
      hyperparameters={"batch_size": (50, 2000), "max_iter": (10, 500)})

_card("parallel_kmeans",
      family="centroid",
      mechanism_tags={"em_update", "mapreduce", "mean_centroid"},
      inductive_biases={"convex_clusters", "isotropic_clusters", "needs_k", "scales_linear"},
      time_complexity=lambda n, d, k: 10 * n * d * k,
      memory_complexity=lambda n, d, k: n * d + k * d,
      hyperparameters={"max_iter": (10, 500), "n_workers": (1, 16)})

_card("kmeans_trimmed",
      family="centroid",
      mechanism_tags={"em_update", "trimmed_mean", "kpp_init"},
      inductive_biases={"convex_clusters", "outlier_robust", "needs_k", "scales_linear"},
      time_complexity=lambda n, d, k: 10 * n * d * k * 1.2,  # trim sort overhead
      memory_complexity=lambda n, d, k: n * d + k * d,
      hyperparameters={"trim": (0.0, 0.3)})

_card("clarans",
      family="centroid",
      mechanism_tags={"medoid", "random_swap_search"},
      inductive_biases={"convex_clusters", "needs_k"},
      time_complexity=lambda n, d, k: n * k * 50,  # ~50 swaps
      memory_complexity=lambda n, d, k: n * d + k,
      hyperparameters={"numlocal": (1, 10), "maxneigh": (20, 500)})

_card("clarans_pp",
      family="centroid",
      mechanism_tags={"medoid", "kpp_init", "random_swap_search"},
      inductive_biases={"convex_clusters", "needs_k"},
      time_complexity=lambda n, d, k: n * k * 80,
      memory_complexity=lambda n, d, k: n * d + k,
      hyperparameters={"maxneigh": (20, 500)})

# --- Hierarchical / density --------------------------------------------------

_card("birch_algo",
      family="hierarchical",
      mechanism_tags={"cf_tree", "single_pass", "threshold"},
      inductive_biases={"convex_clusters", "needs_k", "scales_linear", "hyperparameter_brittle"},
      time_complexity=lambda n, d, k: n * d * 5,
      memory_complexity=lambda n, d, k: n * d + 1000 * d,
      hyperparameters={"threshold": (0.1, 5.0), "branching_factor": (10, 200)})

_card("agglomerative",
      family="hierarchical",
      mechanism_tags={"ward_linkage", "bottom_up_merge"},
      inductive_biases={"convex_clusters", "needs_k", "scales_quadratic"},
      time_complexity=lambda n, d, k: n * n * d,
      memory_complexity=lambda n, d, k: n * n,
      hyperparameters={"linkage": ("ward", "single", "complete", "average")})

_card("dbscan",
      family="density",
      mechanism_tags={"epsilon_ball", "core_points"},
      inductive_biases={"discovers_k", "non_convex_capable", "hyperparameter_brittle", "outlier_robust"},
      time_complexity=lambda n, d, k: n * math.log(max(2, n)) * d,
      memory_complexity=lambda n, d, k: n * d,
      handles_k_directly=False,
      hyperparameters={"eps": (0.05, 3.0), "min_samples": (3, 50)})

_card("dbscan_auto",
      family="density",
      mechanism_tags={"epsilon_ball", "k_distance_knee"},
      inductive_biases={"discovers_k", "non_convex_capable", "outlier_robust"},
      time_complexity=lambda n, d, k: n * math.log(max(2, n)) * d * 1.5,
      memory_complexity=lambda n, d, k: n * d,
      handles_k_directly=False)

_card("optics",
      family="density",
      mechanism_tags={"reachability", "ordering"},
      inductive_biases={"discovers_k", "non_convex_capable", "scales_quadratic", "outlier_robust"},
      time_complexity=lambda n, d, k: n * n * d,
      memory_complexity=lambda n, d, k: n * d + n,
      handles_k_directly=False,
      hyperparameters={"min_samples": (3, 50), "xi": (0.01, 0.3)})

_card("meanshift",
      family="density",
      mechanism_tags={"kde", "mode_seeking", "bandwidth"},
      inductive_biases={"discovers_k", "non_convex_capable", "scales_quadratic", "hyperparameter_brittle"},
      time_complexity=lambda n, d, k: n * n * d,
      memory_complexity=lambda n, d, k: n * d,
      handles_k_directly=False,
      hyperparameters={"bandwidth": (0.05, 5.0)})

_card("meanshift_robust",
      family="density",
      mechanism_tags={"kde", "mode_seeking", "trimmed_bandwidth"},
      inductive_biases={"discovers_k", "non_convex_capable", "outlier_robust"},
      time_complexity=lambda n, d, k: n * n * d * 1.1,
      memory_complexity=lambda n, d, k: n * d,
      handles_k_directly=False,
      hyperparameters={"trim": (0.0, 0.3), "quantile": (0.1, 0.5)})

# --- EM / mixture ----------------------------------------------------------

_card("gmm",
      family="probabilistic_mixture",
      mechanism_tags={"em_update", "posterior_weighted", "full_covariance"},
      inductive_biases={"outlier_robust", "needs_k", "scales_linear"},
      time_complexity=lambda n, d, k: 30 * n * k * d * d,
      memory_complexity=lambda n, d, k: n * k + k * d * d,
      hyperparameters={"covariance_type": ("full", "tied", "diag", "spherical"),
                       "reg_covar": (1e-6, 1e-2)})

# --- Graph / spectral ------------------------------------------------------

_card("spectral",
      family="graph",
      mechanism_tags={"knn_graph", "laplacian_eigen", "kmeans_final"},
      inductive_biases={"non_convex_capable", "needs_k", "scales_quadratic", "hyperparameter_brittle"},
      time_complexity=lambda n, d, k: n * n + n * d + 30 * n * k,
      memory_complexity=lambda n, d, k: n * n,
      hyperparameters={"n_neighbors": (5, 50)})

_card("chameleon",
      family="graph",
      mechanism_tags={"knn_graph", "agglomerative_merge", "relative_closeness"},
      inductive_biases={"non_convex_capable", "needs_k", "scales_quadratic"},
      time_complexity=lambda n, d, k: n * n + n * d,
      memory_complexity=lambda n, d, k: n * n)

_card("lmm",
      family="graph",
      mechanism_tags={"knn_graph", "laplacian_eigen", "em_basis"},
      inductive_biases={"non_convex_capable", "needs_k", "scales_quadratic"},
      time_complexity=lambda n, d, k: n * n + 30 * n * k * d,
      memory_complexity=lambda n, d, k: n * n)

_card("fmm",
      family="probabilistic_mixture",
      mechanism_tags={"random_fourier_basis", "em_basis", "self_normalize"},
      inductive_biases={"needs_k", "scales_linear"},
      time_complexity=lambda n, d, k: 40 * n * 64 * k,
      memory_complexity=lambda n, d, k: n * 64 + k * 64)

_card("amm",
      family="probabilistic_mixture",
      mechanism_tags={"autoencoder_basis", "em_basis"},
      inductive_biases={"needs_k", "scales_linear"},
      time_complexity=lambda n, d, k: 50 * n * d + 30 * n * k,
      memory_complexity=lambda n, d, k: n * d + k * d)

_card("mri",
      family="probabilistic_mixture",
      mechanism_tags={"relaxation_signature", "knn_local", "kmeans_final"},
      inductive_biases={"needs_k", "scales_quadratic"},
      time_complexity=lambda n, d, k: n * n * d * 0.3,
      memory_complexity=lambda n, d, k: n * d)

_card("s5c",
      family="graph",
      mechanism_tags={"sparse_subspace", "omp", "spectral_final"},
      inductive_biases={"subspace_structured", "needs_k", "scales_sublinear"},
      time_complexity=lambda n, d, k: 200 * 200 * d + n * 200,  # m=sample_size
      memory_complexity=lambda n, d, k: 200 * 200,
      hyperparameters={"sample_size": (50, 1000), "n_nonzero_coefs": (3, 20)})

# --- Ensembles -------------------------------------------------------------

_card("consensus",
      family="ensemble",
      mechanism_tags={"majority_vote", "hungarian_align"},
      inductive_biases={"ensemble", "needs_k"},
      time_complexity=lambda n, d, k: 3 * 10 * n * d * k,  # 3 base algos
      memory_complexity=lambda n, d, k: 3 * n + k * d)

_card("pwcc",
      family="ensemble",
      mechanism_tags={"weighted_vote", "purity_weights", "hungarian_align"},
      inductive_biases={"ensemble", "needs_k"},
      time_complexity=lambda n, d, k: 3 * 10 * n * d * k,
      memory_complexity=lambda n, d, k: 3 * n + k * d)

_card("pwcc_diverse",
      family="ensemble",
      mechanism_tags={"weighted_vote", "purity_weights", "diverse_base"},
      inductive_biases={"ensemble", "non_convex_capable", "outlier_robust", "needs_k"},
      time_complexity=lambda n, d, k: 3 * 50 * n * k,  # spectral dominates
      memory_complexity=lambda n, d, k: 3 * n + k * d)

# --- Syntheses (v1 / v2 / v3) ----------------------------------------------

_card("aura",
      family="meta",
      mechanism_tags={"nystrom_laplacian", "gmm_in_embed"},
      inductive_biases={"meta", "non_convex_capable", "outlier_robust", "needs_k"},
      time_complexity=lambda n, d, k: n * 100 + 30 * n * k * k,
      memory_complexity=lambda n, d, k: n * 100 + n * d)

_card("aura_v2",
      family="meta",
      mechanism_tags={"nystrom_laplacian", "zscored_kmeans"},
      inductive_biases={"meta", "non_convex_capable", "needs_k"},
      time_complexity=lambda n, d, k: n * 100 + 30 * n * k,
      memory_complexity=lambda n, d, k: n * 100 + n * d)

_card("aura_v3",
      family="meta",
      mechanism_tags={"nystrom_laplacian", "rank_dispatch", "zscored_kmeans_or_gmm"},
      inductive_biases={"meta", "non_convex_capable", "outlier_robust", "needs_k"},
      time_complexity=lambda n, d, k: n * 100 + 30 * n * k * 1.5,
      memory_complexity=lambda n, d, k: n * 100 + n * d)

_card("meta_clusterer",
      family="meta",
      mechanism_tags={"fingerprint_dispatch", "rule_based"},
      inductive_biases={"meta", "needs_k", "ensemble"},
      time_complexity=lambda n, d, k: n * d * 3 + 10 * n * d * k,  # probe + dispatch
      memory_complexity=lambda n, d, k: n * d)

_card("meta_clusterer_v2",
      family="meta",
      mechanism_tags={"fingerprint_dispatch", "convexity_ratio", "silhouette_probe"},
      inductive_biases={"meta", "needs_k", "non_convex_capable", "ensemble"},
      time_complexity=lambda n, d, k: n * d * 5 + 30 * n * k,
      memory_complexity=lambda n, d, k: n * d)

_card("meta_clusterer_v3",
      family="meta",
      mechanism_tags={"fingerprint_dispatch", "v1_v2_probe", "silhouette"},
      inductive_biases={"meta", "needs_k", "non_convex_capable", "ensemble"},
      time_complexity=lambda n, d, k: n * d * 7 + 30 * n * k,
      memory_complexity=lambda n, d, k: n * d)

_card("rapid",
      family="meta",
      mechanism_tags={"dbscan_partition", "per_region_route"},
      inductive_biases={"meta", "non_convex_capable", "discovers_k_partial"},
      time_complexity=lambda n, d, k: n * math.log(max(2, n)) * d + n * d * k,
      memory_complexity=lambda n, d, k: n * d)

_card("rapid_v2",
      family="meta",
      mechanism_tags={"lof_prefilter", "dbscan_partition", "per_region_route"},
      inductive_biases={"meta", "non_convex_capable", "outlier_robust"},
      time_complexity=lambda n, d, k: n * d * 25 + n * math.log(max(2, n)) * d,
      memory_complexity=lambda n, d, k: n * d)

_card("rapid_v3",
      family="meta",
      mechanism_tags={"lof_estimate", "conditional_prefilter", "dbscan_partition"},
      inductive_biases={"meta", "non_convex_capable", "outlier_robust"},
      time_complexity=lambda n, d, k: n * d * 20 + n * math.log(max(2, n)) * d,
      memory_complexity=lambda n, d, k: n * d)

_card("learned_router",
      family="meta",
      mechanism_tags={"knn_fingerprint", "dispatch"},
      inductive_biases={"meta", "learned", "needs_k"},
      time_complexity=lambda n, d, k: n * d * 7 + 30 * n * k,
      memory_complexity=lambda n, d, k: n * d)

_card("learned_router_v2",
      family="meta",
      mechanism_tags={"knn_fingerprint", "probe_features", "intrinsic_dim", "dispatch"},
      inductive_biases={"meta", "learned", "needs_k"},
      time_complexity=lambda n, d, k: n * d * 15 + 30 * n * k,
      memory_complexity=lambda n, d, k: n * d)


# ---------------------------------------------------------------------------
# Inductive-bias compatibility scoring against a data fingerprint.
# ---------------------------------------------------------------------------

# Pairs of (bias the algorithm has, fingerprint feature that the bias is
# *good* on or *bad* on). Used by predict_ari_upper_bound to penalise
# obvious mismatches.
BIAS_DATA_MATCH = {
    # algo bias                fingerprint signal       sign     weight
    "convex_clusters":         ("conv_cv",              "low",   0.30),
    "non_convex_capable":      ("conv_cv",              "high",  0.25),
    "outlier_robust":          ("outlier_frac",         "high",  0.30),
    "subspace_structured":     ("eff_dim",              "low_relative_to_d", 0.25),
    "scales_quadratic":        ("log_n",                "high",  -0.10),  # penalty
    "ensemble":                ("conv_cv",              "any",   0.05),
    "meta":                    ("conv_cv",              "any",   0.10),
}


def predict_ari_upper_bound(card: AlgorithmCard, fingerprint: Dict[str, float]) -> float:
    """Predict an *upper bound* on ARI given a card and a data fingerprint.

    Starts from 0.5 (the "we don't know" prior) and adds / subtracts
    based on how well the card's inductive biases match the
    fingerprint signals in :data:`BIAS_DATA_MATCH`. Clamped to [0, 1].
    """
    score = 0.5
    for bias in card.inductive_biases:
        match = BIAS_DATA_MATCH.get(bias)
        if match is None:
            continue
        feature, polarity, weight = match
        value = fingerprint.get(feature)
        if value is None:
            continue
        if polarity == "high" and value > 0.5:
            score += weight
        elif polarity == "low" and value < 0.3:
            score += weight
        elif polarity == "low_relative_to_d":
            # eff_dim much smaller than d_ambient
            d = fingerprint.get("d", value)
            if d > 0 and value / d < 0.25:
                score += weight
        elif polarity == "any":
            score += weight
    return max(0.0, min(1.0, score))


def predict_performance(card: AlgorithmCard, n: int, d: int, k: Optional[int],
                        fingerprint: Optional[Dict[str, float]] = None,
                        time_calibration: float = 1e-8,
                        mem_calibration: float = 8e-6) -> Dict[str, float]:
    """Bundle predictions for one (card, task) pair."""
    out = {
        "theoretical_wall_time_s": card.predict_wall_time(n, d, k, time_calibration),
        "theoretical_rss_mb":      card.predict_memory_mb(n, d, k, mem_calibration),
    }
    if fingerprint is not None:
        out["theoretical_ari_upper_bound"] = predict_ari_upper_bound(card, fingerprint)
    return out


def calibrate_from_benchmark(
    results: List[Dict[str, Any]],
    cards: Dict[str, AlgorithmCard] = ALGORITHM_CARDS,
) -> Dict[str, Dict[str, float]]:
    """Fit a multiplicative constant per algorithm from past benchmark
    rows so theoretical wall-time / memory predictions are calibrated
    to this hardware's overhead.

    For each algorithm with at least two empirical (n, d, k, wall_time)
    triples in ``results``, fits ``calibration = median(actual / theoretical)``.
    Returns a per-algorithm dict of calibration constants — pass
    ``time_calibration=calibrations[algo]["time"]`` to
    :func:`predict_performance` for the per-algo-calibrated prediction.
    """
    by_algo: Dict[str, Dict[str, List[float]]] = {}
    for row in results:
        algo = row.get("algo")
        card = cards.get(algo)
        if card is None:
            continue
        n = row.get("n_samples")
        d = row.get("n_features")
        k = row.get("k_target")
        wt_actual = row.get("wall_time_s")
        if not all(isinstance(v, (int, float)) for v in (n, d, k, wt_actual) if v is not None):
            continue
        if wt_actual is None or wt_actual <= 0:
            continue
        theoretical_ops = card.time_complexity(int(n), int(d), max(1, int(k or 1)))
        if theoretical_ops <= 0:
            continue
        by_algo.setdefault(algo, {"time_ratios": [], "mem_ratios": []})
        by_algo[algo]["time_ratios"].append(float(wt_actual) / float(theoretical_ops))
        rss_actual = row.get("rss_delta_mb")
        if isinstance(rss_actual, (int, float)) and rss_actual > 0:
            mem_ops = card.memory_complexity(int(n), int(d), max(1, int(k or 1)))
            if mem_ops > 0:
                by_algo[algo]["mem_ratios"].append(float(rss_actual) / float(mem_ops))

    out: Dict[str, Dict[str, float]] = {}
    for algo, d in by_algo.items():
        time_const = (
            float(sorted(d["time_ratios"])[len(d["time_ratios"]) // 2])
            if d["time_ratios"] else 1e-8
        )
        mem_const = (
            float(sorted(d["mem_ratios"])[len(d["mem_ratios"]) // 2])
            if d["mem_ratios"] else 8e-6
        )
        out[algo] = {"time": time_const, "mem": mem_const}
    return out
