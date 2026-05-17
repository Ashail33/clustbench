"""Find *gap fingerprints* — regimes the current algorithm registry misses.

A "gap" is a point in fingerprint space where the best card in
:data:`clustbench.algorithm_cards.ALGORITHM_CARDS` scores below a
threshold under :func:`predict_ari_upper_bound`. Those fingerprints are
the hardest tasks in algorithm-space; running new algorithms on them
is maximally discriminative.

This module is a pure library: it does no I/O. The companion script
``scripts/find_gaps.py`` wires it up to ``docs/data/results.json`` and
writes ``runs/gaps_report.json``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import product
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from .algorithm_cards import (
    ALGORITHM_CARDS,
    AlgorithmCard,
    predict_ari_upper_bound,
)


# Default sweep axes — Cartesian product is 3*4*3*3*3*3*3 = 2916, so the
# caller's cap (``max_grid``) will downsample. The axes intentionally
# straddle the regimes the benchmark exercises (small / medium / large n,
# low / high d, balanced / imbalanced clusters, etc).
DEFAULT_AXES: Dict[str, Sequence[float]] = {
    "log_n":           (2.0, 3.0, 4.0),
    "d":               (2.0, 10.0, 50.0, 200.0),
    "k":               (2.0, 5.0, 20.0),
    "eff_dim_ratio":   (0.05, 0.5, 1.0),
    "conv_cv":         (0.1, 0.5, 0.9),
    "outlier_frac":    (0.0, 0.05, 0.2),
    "density_skew":    (0.0, 0.3, 0.8),
}


# Fingerprint axes used for distance computations. Each entry is
# (key, scale) — the scale is the typical span of the axis, used to
# normalise differences so eg log_n=1.0 isn't dwarfed by d=200.
_DISTANCE_AXES: Tuple[Tuple[str, float], ...] = (
    ("log_n", 2.0),
    ("d", 200.0),
    ("k", 20.0),
    ("eff_dim_ratio", 1.0),
    ("conv_cv", 1.0),
    ("outlier_frac", 0.2),
    ("density_skew", 1.0),
)


@dataclass
class Gap:
    """One gap candidate: target fingerprint + nearest existing match."""

    target_fingerprint: Dict[str, float]
    max_existing_score: float
    top_algorithms: List[Tuple[str, float]]
    nearest_dataset_id: str
    nearest_distance: float
    nearest_fingerprint: Dict[str, float]
    suggested_overrides: Dict[str, Any]
    interest: float


# ---------------------------------------------------------------------------
# Grid construction
# ---------------------------------------------------------------------------

def build_fingerprint_grid(
    axes: Optional[Dict[str, Sequence[float]]] = None,
    max_grid: int = 500,
) -> List[Dict[str, float]]:
    """Cartesian product of ``axes``, deterministically subsampled to ``max_grid``.

    Each returned fingerprint has both ``eff_dim_ratio`` (the axis value)
    and ``eff_dim`` (= ``d * eff_dim_ratio``) so it plugs straight into
    :func:`predict_ari_upper_bound`, which keys on ``eff_dim``.
    """
    ax = dict(DEFAULT_AXES if axes is None else axes)
    keys = list(ax.keys())
    vals = [list(ax[k]) for k in keys]

    full = list(product(*vals))
    # Deterministic stride-subsample (no RNG → reproducible).
    if len(full) > max_grid:
        stride = max(1, len(full) // max_grid)
        full = [full[i] for i in range(0, len(full), stride)][:max_grid]

    grid: List[Dict[str, float]] = []
    for combo in full:
        fp = {k: float(v) for k, v in zip(keys, combo)}
        # Derived: absolute eff_dim, since predict_ari_upper_bound
        # checks ``value / d`` for the subspace_structured bias.
        d = fp.get("d", 1.0) or 1.0
        fp["eff_dim"] = d * fp.get("eff_dim_ratio", 1.0)
        grid.append(fp)
    return grid


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def score_fingerprint(
    fingerprint: Dict[str, float],
    cards: Dict[str, AlgorithmCard] = ALGORITHM_CARDS,
) -> List[Tuple[str, float]]:
    """Return ``[(algo_name, predicted_ari_upper_bound), ...]`` desc-sorted."""
    scores = [
        (name, predict_ari_upper_bound(card, fingerprint))
        for name, card in cards.items()
    ]
    scores.sort(key=lambda kv: kv[1], reverse=True)
    return scores


def fingerprint_distance(a: Dict[str, float], b: Dict[str, float]) -> float:
    """Scale-normalised Euclidean distance over ``_DISTANCE_AXES``."""
    s = 0.0
    for key, scale in _DISTANCE_AXES:
        va, vb = a.get(key, 0.0), b.get(key, 0.0)
        s += ((float(va) - float(vb)) / (scale or 1.0)) ** 2
    return math.sqrt(s)


# ---------------------------------------------------------------------------
# Inverse mapping: fingerprint -> DataSpec override hints.
# ---------------------------------------------------------------------------

def nearest_existing(
    fingerprint: Dict[str, float],
    existing: Sequence[Dict[str, float]],
) -> Tuple[int, float]:
    """Index + distance of the closest fingerprint in ``existing``."""
    best_i, best_d = -1, float("inf")
    for i, ex in enumerate(existing):
        d = fingerprint_distance(fingerprint, ex)
        if d < best_d:
            best_d, best_i = d, i
    return best_i, best_d


def suggest_overrides(
    target: Dict[str, float],
    nearest_row: Dict[str, Any],
    dataset_priors: Dict[str, Tuple[float, float]],
) -> Dict[str, Any]:
    """Translate a target fingerprint + nearest benchmark row into DataSpec tweaks.

    Returns a dict of suggested ``DataSpec`` field overrides. We only
    touch fields the existing generators expose (``n_samples``,
    ``n_features``, ``centers``, ``outliers``, ``noise``, ``density``,
    ``outlier_extremity``) and a hint ``_intrinsic_dim_ratio`` for
    callers (e.g. ``gen_inverse_pca``) that can use it.
    """
    n_target = int(round(10 ** float(target["log_n"])))
    d_target = int(round(float(target["d"])))
    k_target = int(round(float(target["k"])))

    # Outliers: outlier_frac is fraction of total points -> count.
    of = float(target.get("outlier_frac", 0.0))
    n_outliers = int(round(of * n_target / max(1e-9, 1.0 - of)))

    # density_skew in [0, 1] -> density in (0, 1]. Add some noise points
    # when skew is high (the existing fingerprint extractor in
    # annotate_predictions.py treats density<1 and noise>0 symmetrically).
    skew = float(target.get("density_skew", 0.0))
    density = max(0.05, 1.0 - skew)
    n_noise = int(round(0.05 * n_target)) if skew > 0.4 else 0

    # compactness from conv_cv: higher conv_cv -> stretchier / less convex
    # cluster shape. In MDCGen-style generators, larger compactness widens
    # the per-cluster sigma; in moons / circles / spiral it controls noise.
    cv = float(target.get("conv_cv", 0.5))
    compactness = round(0.5 + 1.5 * cv, 3)

    # outlier_extremity: only meaningful when outliers > 0. Push it up if
    # outlier_frac is high (worst-case stress for non-robust algos).
    extremity = 1.0 + 4.0 * of if n_outliers > 0 else 1.0

    overrides: Dict[str, Any] = {
        "n_samples": n_target,
        "n_features": d_target,
        "centers": k_target,
        "compactness": compactness,
        "outliers": n_outliers,
        "noise": n_noise,
        "density": round(density, 3),
        "outlier_extremity": round(extremity, 2),
    }

    # eff_dim_ratio is structural and only honoured by gen_inverse_pca
    # via its n_components knob; surface as a hint, not a real DataSpec
    # field, so unrelated generators don't blow up.
    edr = float(target.get("eff_dim_ratio", 1.0))
    overrides["_intrinsic_dim_ratio"] = round(edr, 3)

    # Annotate what we're *changing* relative to the nearest benchmark row
    # — humans skimming the report want to see the delta, not the absolute
    # values.
    deltas: Dict[str, Any] = {}
    for field in ("n_samples", "n_features", "centers", "outliers", "noise", "density"):
        old = nearest_row.get(field)
        new = overrides.get(field)
        if old is None or new is None:
            continue
        try:
            if float(old) != float(new):
                deltas[field] = {"from": old, "to": new}
        except (TypeError, ValueError):
            continue
    if deltas:
        overrides["_delta_from_nearest"] = deltas

    # Sanity: which dataset_id are we expected to start from, plus the
    # implied prior so callers can verify the choice.
    ds = nearest_row.get("dataset_id")
    if ds:
        overrides["_start_from"] = ds
        if ds in dataset_priors:
            overrides["_prior_conv_cv"], overrides["_prior_eff_dim_ratio"] = (
                dataset_priors[ds]
            )

    return overrides


# ---------------------------------------------------------------------------
# Existing-benchmark fingerprint extraction.
# ---------------------------------------------------------------------------

def existing_fingerprint(
    row: Dict[str, Any],
    dataset_priors: Dict[str, Tuple[float, float]],
) -> Dict[str, float]:
    """Mirror of ``scripts/annotate_predictions._fingerprint_from_row``.

    Re-implemented locally so this module has no dependency on the
    script. Output keys: ``log_n``, ``d``, ``k``, ``eff_dim``,
    ``eff_dim_ratio``, ``conv_cv``, ``outlier_frac``, ``density_skew``.
    """
    ds = row.get("dataset_id")
    n = float(row.get("n_samples", 0) or 0)
    d = float(row.get("n_features", 0) or 1)
    k = float(row.get("k_target", 0) or 1)
    outliers = float(row.get("outliers", 0) or 0)
    noise = float(row.get("noise", 0) or 0)
    density = float(row.get("density", 1.0) or 1.0)

    conv_cv, eff_dim_ratio = dataset_priors.get(ds, (0.50, 1.0))
    return {
        "log_n": math.log10(max(2.0, n)),
        "d": d,
        "k": k,
        "eff_dim_ratio": eff_dim_ratio,
        "eff_dim": d * eff_dim_ratio,
        "conv_cv": conv_cv,
        "outlier_frac": (outliers / max(1.0, n + outliers)),
        "density_skew": min(1.0, 1.0 - density) + 0.1 * (noise > 0),
    }


def unique_existing_fingerprints(
    rows: Iterable[Dict[str, Any]],
    dataset_priors: Dict[str, Tuple[float, float]],
) -> List[Tuple[Dict[str, float], Dict[str, Any]]]:
    """Deduplicate ``rows`` by (dataset_id, n, d, k, outliers, noise, density).

    Returns ``[(fingerprint, representative_row), ...]`` — one fingerprint
    per unique benchmark task, so distance / nearest-neighbour queries
    aren't dominated by tasks that simply have many algorithms run on
    them.
    """
    seen: Dict[Tuple, Dict[str, Any]] = {}
    for row in rows:
        key = (
            row.get("dataset_id"),
            row.get("n_samples"),
            row.get("n_features"),
            row.get("k_target"),
            row.get("outliers"),
            row.get("noise"),
            row.get("density"),
        )
        seen.setdefault(key, row)
    return [
        (existing_fingerprint(row, dataset_priors), row) for row in seen.values()
    ]


# ---------------------------------------------------------------------------
# Ranking
# ---------------------------------------------------------------------------

def rank_gaps(
    grid: Sequence[Dict[str, float]],
    existing: Sequence[Tuple[Dict[str, float], Dict[str, Any]]],
    dataset_priors: Dict[str, Tuple[float, float]],
    threshold: float = 0.65,
    top_n: int = 20,
    cards: Dict[str, AlgorithmCard] = ALGORITHM_CARDS,
) -> Tuple[List[Gap], int]:
    """Rank gap fingerprints by *interest*.

    Interest = (1 - max_existing_score) * proximity_bonus, where
    proximity_bonus = exp(-distance_to_nearest_benchmark). The product
    favours fingerprints that are (a) genuinely uncovered by current
    cards and (b) close enough to the existing benchmark that we can
    actually materialise them with the current generators.

    Returns ``(top_gaps, n_gaps_below_threshold)``.
    """
    existing_fps = [fp for fp, _ in existing]
    existing_rows = [row for _, row in existing]

    gaps: List[Gap] = []
    n_below = 0
    for fp in grid:
        scores = score_fingerprint(fp, cards)
        max_score = scores[0][1] if scores else 0.0
        if max_score >= threshold:
            continue
        n_below += 1

        i, dist = nearest_existing(fp, existing_fps)
        nearest_row = existing_rows[i] if i >= 0 else {}
        nearest_fp = existing_fps[i] if i >= 0 else {}

        interest = (1.0 - max_score) * math.exp(-dist)

        overrides = suggest_overrides(fp, nearest_row, dataset_priors)

        gaps.append(Gap(
            target_fingerprint={k: round(float(v), 4) for k, v in fp.items()},
            max_existing_score=round(max_score, 4),
            top_algorithms=[(n, round(s, 4)) for n, s in scores[:3]],
            nearest_dataset_id=nearest_row.get("dataset_id", "unknown"),
            nearest_distance=round(dist, 4),
            nearest_fingerprint={k: round(float(v), 4) for k, v in nearest_fp.items()},
            suggested_overrides=overrides,
            interest=round(interest, 4),
        ))

    gaps.sort(key=lambda g: g.interest, reverse=True)
    return gaps[:top_n], n_below


def gap_to_dict(g: Gap) -> Dict[str, Any]:
    """JSON-serialisable view of a :class:`Gap`."""
    return {
        "target_fingerprint": g.target_fingerprint,
        "max_existing_score": g.max_existing_score,
        "top_algorithms": [{"algo": a, "predicted_ari_upper_bound": s} for a, s in g.top_algorithms],
        "nearest_dataset_id": g.nearest_dataset_id,
        "nearest_distance": g.nearest_distance,
        "nearest_fingerprint": g.nearest_fingerprint,
        "suggested_overrides": g.suggested_overrides,
        "interest": g.interest,
    }
