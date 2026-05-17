"""Algorithm mutator — propose novel clustering algorithms by mutating
existing :class:`AlgorithmCard` instances and ranking the proposals by
predicted ARI on a benchmark-derived fingerprint distribution.

The goal: surface mutations that, if implemented, would land at the top
of the leaderboard. We do this by:

1. Defining four mutation operators (``swap_mechanism``, ``add_bias``,
   ``crossover``, ``stack``) that each take one or two existing cards
   and return a new candidate card.
2. Sampling ~50 distinct fingerprints from ``docs/data/results.json``
   (the 4332-row benchmark) and predicting each candidate's mean ARI
   upper bound across them via
   :func:`clustbench.algorithm_cards.predict_ari_upper_bound`.
3. Only keeping candidates whose mean predicted ARI beats the best
   existing card's mean predicted ARI on those same fingerprints by at
   least 0.02.
4. Running a random search (default 200 iterations) and returning the
   top-N (default 10) candidates plus a "next concrete implementation
   step" hint per surviving candidate.

The vocabulary of mechanism tags and inductive biases is restricted to
what already appears in the registry — this keeps the upper-bound
prediction grounded.
"""

from __future__ import annotations

import json
import math
import os
import random
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Set, Tuple

from .algorithm_cards import (
    ALGORITHM_CARDS,
    ALL_BIASES,
    AlgorithmCard,
    predict_ari_upper_bound,
)


# ---------------------------------------------------------------------------
# Vocabulary — pulled from the existing registry so predictions stay grounded.
# ---------------------------------------------------------------------------


def _all_mechanism_tags(cards: Dict[str, AlgorithmCard] = ALGORITHM_CARDS) -> Set[str]:
    """Union of mechanism tags across every registered card."""
    tags: Set[str] = set()
    for card in cards.values():
        tags |= set(card.mechanism_tags)
    return tags


# A small map of mechanism-tag implications: adding one of these tags
# nudges the inductive-bias set as well. Drawn from the prose in the
# registry (e.g. medoid implies a small lift in outlier robustness).
_TAG_IMPLIES_BIAS: Dict[str, Set[str]] = {
    "medoid":               {"outlier_robust"},
    "trimmed_mean":         {"outlier_robust"},
    "trimmed_bandwidth":    {"outlier_robust"},
    "posterior_weighted":   {"outlier_robust"},
    "lof_prefilter":        {"outlier_robust"},
    "lof_estimate":         {"outlier_robust"},
    "knn_graph":            {"non_convex_capable"},
    "laplacian_eigen":      {"non_convex_capable"},
    "kde":                  {"non_convex_capable"},
    "mode_seeking":         {"non_convex_capable"},
    "epsilon_ball":         {"non_convex_capable", "outlier_robust"},
    "reachability":         {"non_convex_capable"},
    "minibatch":            {"scales_sublinear"},
    "online_update":        {"scales_sublinear"},
    "mapreduce":            {"scales_linear"},
    "random_fourier_basis": {"scales_linear"},
    "autoencoder_basis":    {"scales_linear"},
    "sparse_subspace":      {"subspace_structured"},
    "majority_vote":        {"ensemble"},
    "weighted_vote":        {"ensemble"},
    "fingerprint_dispatch": {"meta"},
    "knn_fingerprint":      {"meta", "learned"},
    "rule_based":           {"meta"},
    "nystrom_laplacian":    {"non_convex_capable", "scales_linear"},
}


def _mechanism_implied_biases(tags: Iterable[str]) -> Set[str]:
    biases: Set[str] = set()
    for t in tags:
        biases |= _TAG_IMPLIES_BIAS.get(t, set())
    return biases


# ---------------------------------------------------------------------------
# Mutation operators — pure functions returning new AlgorithmCard instances.
# ---------------------------------------------------------------------------


def _scale_complexity(
    base: Callable[[int, int, int], float], factor: float
) -> Callable[[int, int, int], float]:
    """Return a new complexity callable that's ``factor *`` the base."""
    return lambda n, d, k, _b=base, _f=factor: _b(n, d, k) * _f


def _max_complexity(
    a: Callable[[int, int, int], float], b: Callable[[int, int, int], float]
) -> Callable[[int, int, int], float]:
    return lambda n, d, k, _a=a, _b=b: max(_a(n, d, k), _b(n, d, k))


def _sum_complexity(
    a: Callable[[int, int, int], float], b: Callable[[int, int, int], float]
) -> Callable[[int, int, int], float]:
    return lambda n, d, k, _a=a, _b=b: _a(n, d, k) + _b(n, d, k)


def swap_mechanism(
    card: AlgorithmCard, old_tag: str, new_tag: str, name: Optional[str] = None
) -> AlgorithmCard:
    """Replace ``old_tag`` in ``card``'s mechanism set with ``new_tag``.

    If the new tag has an implied bias (e.g. ``medoid -> outlier_robust``)
    that bias is added. If the *removed* tag had an implied bias that no
    other remaining tag also implies, that bias is dropped.
    """
    if old_tag not in card.mechanism_tags:
        # Nothing to swap; just return a shallow rename so the search
        # loop can still treat this as a (no-op) candidate.
        new_tags = set(card.mechanism_tags) | {new_tag}
    else:
        new_tags = (set(card.mechanism_tags) - {old_tag}) | {new_tag}

    # Recompute biases: start from the parent set, then drop biases
    # that were ONLY implied by the removed tag, then union in the new
    # tag's implications.
    parent_biases = set(card.inductive_biases)
    removed_implied = _TAG_IMPLIES_BIAS.get(old_tag, set())
    still_supported = _mechanism_implied_biases(new_tags)
    biases = (parent_biases - (removed_implied - still_supported)) | _TAG_IMPLIES_BIAS.get(new_tag, set())

    return AlgorithmCard(
        name=name or f"swap[{card.name}:{old_tag}->{new_tag}]",
        family=card.family,
        mechanism_tags=new_tags,
        inductive_biases=biases,
        time_complexity=card.time_complexity,
        memory_complexity=card.memory_complexity,
        handles_k_directly=card.handles_k_directly,
        hyperparameters=dict(card.hyperparameters),
        notes=f"swap({old_tag}->{new_tag}) from {card.name}",
    )


def add_bias(
    card: AlgorithmCard, new_bias: str, name: Optional[str] = None
) -> AlgorithmCard:
    """Add ``new_bias`` to ``card``'s inductive biases.

    Time complexity is scaled by 1.2 to reflect the cost of an added
    mechanism (e.g. trimming). Memory complexity is left unchanged.
    """
    biases = set(card.inductive_biases) | {new_bias}
    return AlgorithmCard(
        name=name or f"add[{card.name}:+{new_bias}]",
        family=card.family,
        mechanism_tags=set(card.mechanism_tags),
        inductive_biases=biases,
        time_complexity=_scale_complexity(card.time_complexity, 1.2),
        memory_complexity=card.memory_complexity,
        handles_k_directly=card.handles_k_directly,
        hyperparameters=dict(card.hyperparameters),
        notes=f"add_bias({new_bias}) onto {card.name}",
    )


def crossover(
    card_a: AlgorithmCard,
    card_b: AlgorithmCard,
    name: Optional[str] = None,
    rng: Optional[random.Random] = None,
) -> AlgorithmCard:
    """Mix mechanism tags from two cards.

    Take half (rounded up) of each parent's mechanism tags; the
    inductive biases are the union; the complexity is the max of the
    two parents (worst-case).
    """
    rng = rng or random.Random()
    a_tags = list(card_a.mechanism_tags)
    b_tags = list(card_b.mechanism_tags)
    rng.shuffle(a_tags)
    rng.shuffle(b_tags)
    half_a = a_tags[: max(1, (len(a_tags) + 1) // 2)]
    half_b = b_tags[: max(1, (len(b_tags) + 1) // 2)]
    new_tags = set(half_a) | set(half_b)

    biases = (
        set(card_a.inductive_biases)
        | set(card_b.inductive_biases)
        | _mechanism_implied_biases(new_tags)
    )

    hp = dict(card_a.hyperparameters)
    hp.update(card_b.hyperparameters)

    return AlgorithmCard(
        name=name or f"cross[{card_a.name}x{card_b.name}]",
        family=(card_a.family if card_a.family == card_b.family else "hybrid"),
        mechanism_tags=new_tags,
        inductive_biases=biases,
        time_complexity=_max_complexity(card_a.time_complexity, card_b.time_complexity),
        memory_complexity=_max_complexity(
            card_a.memory_complexity, card_b.memory_complexity
        ),
        handles_k_directly=card_a.handles_k_directly or card_b.handles_k_directly,
        hyperparameters=hp,
        notes=f"crossover({card_a.name}, {card_b.name})",
    )


def stack(
    base_card: AlgorithmCard,
    post_card: AlgorithmCard,
    name: Optional[str] = None,
) -> AlgorithmCard:
    """Pipeline two cards: run ``base_card``, then ``post_card`` on its
    output (e.g. spectral embedding -> kmeans final).

    Complexity sums; inductive biases are the union.
    """
    new_tags = set(base_card.mechanism_tags) | set(post_card.mechanism_tags)
    biases = (
        set(base_card.inductive_biases)
        | set(post_card.inductive_biases)
        | _mechanism_implied_biases(new_tags)
    )

    hp = dict(base_card.hyperparameters)
    hp.update(post_card.hyperparameters)

    return AlgorithmCard(
        name=name or f"stack[{base_card.name}->{post_card.name}]",
        family=(
            "pipeline"
            if base_card.family != post_card.family
            else base_card.family
        ),
        mechanism_tags=new_tags,
        inductive_biases=biases,
        time_complexity=_sum_complexity(
            base_card.time_complexity, post_card.time_complexity
        ),
        memory_complexity=_sum_complexity(
            base_card.memory_complexity, post_card.memory_complexity
        ),
        handles_k_directly=post_card.handles_k_directly
        or base_card.handles_k_directly,
        hyperparameters=hp,
        notes=f"stack({base_card.name} -> {post_card.name})",
    )


# ---------------------------------------------------------------------------
# Fingerprint sampling from the benchmark.
# ---------------------------------------------------------------------------


def _row_fingerprint(row: Dict[str, Any]) -> Dict[str, float]:
    """Synthesise the fingerprint vector used by
    :func:`predict_ari_upper_bound` from a single benchmark row.

    ``results.json`` stores ``outliers``, ``outlier_extremity``,
    ``density``, ``noise``, ``compactness``, ``n_samples``,
    ``n_features``, ``k_target``, ``dataset_id`` but not the abstract
    ``conv_cv`` / ``outlier_frac`` / ``eff_dim`` / ``log_n`` features
    that :data:`BIAS_DATA_MATCH` references. We synthesise them
    deterministically from what's available:

    * ``conv_cv``  — proxy for convexity-violation coefficient of
      variation. Datasets with a ``shapes``/``spirals`` style id, or
      with low ``compactness``, are non-convex.
    * ``outlier_frac`` — proportional to ``outliers``.
    * ``eff_dim``  — clipped at ``min(n_features, k_target * 2 + 4)``.
      This roughly tracks PCA variance dropping off near ``k``.
    * ``log_n``    — normalised ``log10(n_samples) / 5``.
    * ``d``        — copy of ``n_features``.
    """
    n = float(row.get("n_samples") or 1)
    d_raw = float(row.get("n_features") or 1)
    k = float(row.get("k_target") or 2)
    outliers = float(row.get("outliers") or 0)
    compactness = float(row.get("compactness") or 1.0)
    noise = float(row.get("noise") or 0)
    dataset_id = (row.get("dataset_id") or "").lower()

    # conv_cv: 0=convex blobs, 1=fully non-convex (rings, spirals).
    non_convex_hint = 0.0
    for tag in ("shape", "ring", "spiral", "moon", "graves", "fcps", "wut", "sipu"):
        if tag in dataset_id:
            non_convex_hint += 0.25
    # Soft non-convexity from low compactness or high noise.
    non_convex_hint += max(0.0, 1.0 - compactness) * 0.5
    non_convex_hint += min(0.4, noise * 2.0)
    conv_cv = max(0.0, min(1.0, non_convex_hint))

    outlier_frac = max(0.0, min(1.0, outliers + 0.5 * (1.0 - compactness)))

    eff_dim = min(d_raw, k * 2.0 + 4.0)
    log_n = math.log10(max(n, 10.0)) / 5.0

    return {
        "conv_cv": conv_cv,
        "outlier_frac": outlier_frac,
        "eff_dim": eff_dim,
        "log_n": log_n,
        "d": d_raw,
        "n": n,
        "k": k,
    }


def _row_key(row: Dict[str, Any]) -> Tuple:
    """Identity of a *task* (ignoring algorithm)."""
    return (
        row.get("dataset_id"),
        row.get("seed"),
        row.get("n_samples"),
        row.get("n_features"),
        row.get("k_target"),
        row.get("outliers"),
        row.get("noise"),
        row.get("density"),
        row.get("compactness"),
    )


def sample_fingerprints(
    results_path: os.PathLike,
    n_samples: int = 50,
    seed: int = 0,
) -> List[Dict[str, float]]:
    """Load ``results.json`` and return ``n_samples`` distinct fingerprints."""
    with open(results_path) as f:
        rows = json.load(f)
    seen: Dict[Tuple, Dict[str, float]] = {}
    for row in rows:
        key = _row_key(row)
        if key not in seen:
            seen[key] = _row_fingerprint(row)
    fps = list(seen.values())
    rng = random.Random(seed)
    rng.shuffle(fps)
    return fps[: min(n_samples, len(fps))]


# ---------------------------------------------------------------------------
# Scoring + random search.
# ---------------------------------------------------------------------------


def score_card(card: AlgorithmCard, fingerprints: List[Dict[str, float]]) -> float:
    """Mean ``predict_ari_upper_bound`` across the fingerprint sample."""
    if not fingerprints:
        return 0.0
    total = 0.0
    for fp in fingerprints:
        total += predict_ari_upper_bound(card, fp)
    return total / len(fingerprints)


def best_existing_score(
    fingerprints: List[Dict[str, float]],
    cards: Dict[str, AlgorithmCard] = ALGORITHM_CARDS,
) -> Tuple[str, float]:
    """Return ``(best_card_name, mean_predicted_ari)`` over the sample."""
    best_name = ""
    best_score = -1.0
    for name, card in cards.items():
        s = score_card(card, fingerprints)
        if s > best_score:
            best_score = s
            best_name = name
    return best_name, best_score


def suggest_implementation(parents: List[str], op: str, child: AlgorithmCard) -> str:
    """One-sentence "next step" mapping mutation -> concrete code idea."""
    parent_str = " + ".join(parents)
    if op == "swap_mechanism":
        # Find the swap from the notes.
        return (
            f"Fork {parents[0]} into a new class with the swapped primitive — "
            f"e.g. replace the mean/centroid update with the mechanism implied "
            f"by tags {sorted(child.mechanism_tags)} and keep the existing "
            f"initialisation + hyperparameter grid."
        )
    if op == "add_bias":
        added = sorted(set(child.inductive_biases))
        return (
            f"Wrap {parents[0]} in a pre/post step that enforces the added "
            f"bias (e.g. trim the {int(0.1 * 100)}% highest-residual points, "
            f"or whiten the input) — target biases: {added}."
        )
    if op == "crossover":
        return (
            f"Implement a hybrid {parent_str} clusterer that uses "
            f"{parents[0]}'s primary mechanism to produce candidate "
            f"assignments and {parents[1]}'s mechanism to refine them; "
            f"start with the union of their hyperparameter grids."
        )
    if op == "stack":
        a, b = parents[0], parents[1]
        return (
            f"Implement a {a} -> {b} pipeline: run {a} to produce an "
            f"embedding / pre-partition, then feed that representation "
            f"into {b} for the final assignment (reg_covar=1e-3 if {b} "
            f"is a mixture model to absorb rank-deficient blocks)."
        )
    return "Wire up the mutated card mechanism in src/clustbench/algorithms/."


def _card_to_spec(card: AlgorithmCard, n: int = 1000, d: int = 10, k: int = 5) -> Dict[str, Any]:
    """JSON-friendly summary of a card (mechanism tags, biases, sample
    complexity numbers evaluated at a reference shape)."""
    return {
        "name": card.name,
        "family": card.family,
        "mechanism_tags": sorted(card.mechanism_tags),
        "inductive_biases": sorted(card.inductive_biases),
        "handles_k_directly": card.handles_k_directly,
        "hyperparameters": {
            k_: list(v) if isinstance(v, tuple) else v
            for k_, v in card.hyperparameters.items()
        },
        "time_complexity_at_1000x10x5": float(card.time_complexity(n, d, k)),
        "memory_complexity_at_1000x10x5": float(card.memory_complexity(n, d, k)),
        "notes": card.notes,
    }


def _is_near_duplicate(
    candidate: AlgorithmCard, registry: Dict[str, AlgorithmCard], threshold: float = 0.85
) -> bool:
    """Heuristic: candidate has Jaccard >= threshold with some existing
    card on (mechanism_tags U inductive_biases)."""
    cand_set = set(candidate.mechanism_tags) | set(candidate.inductive_biases)
    if not cand_set:
        return True
    for existing in registry.values():
        exist_set = set(existing.mechanism_tags) | set(existing.inductive_biases)
        if not exist_set:
            continue
        inter = len(cand_set & exist_set)
        union = len(cand_set | exist_set)
        if union > 0 and inter / union >= threshold:
            return True
    return False


def random_search(
    n_iter: int = 200,
    top_n: int = 10,
    seed: int = 0,
    margin: float = 0.02,
    results_path: Optional[os.PathLike] = None,
    n_fingerprints: int = 50,
    cards: Dict[str, AlgorithmCard] = ALGORITHM_CARDS,
) -> Dict[str, Any]:
    """Drive the random-mutation search loop.

    Returns a dict with:

    * ``n_evaluated``            — total candidates generated
    * ``n_beat_existing``        — candidates whose mean predicted ARI
      beat the best existing card by ``margin``
    * ``best_existing_card``     — (name, score) over the fingerprint sample
    * ``top``                    — list of top-N candidate records
    * ``fingerprint_sample_size``
    """
    if results_path is None:
        # Default: docs/data/results.json relative to repo root.
        here = Path(__file__).resolve()
        results_path = here.parents[2] / "docs" / "data" / "results.json"

    rng = random.Random(seed)
    fps = sample_fingerprints(results_path, n_samples=n_fingerprints, seed=seed)

    best_name, best_score = best_existing_score(fps, cards)

    card_list = list(cards.values())
    all_tags = sorted(_all_mechanism_tags(cards))
    all_biases = sorted(ALL_BIASES)

    operators = ["swap_mechanism", "add_bias", "crossover", "stack"]

    survivors: List[Dict[str, Any]] = []
    n_beat = 0

    for i in range(n_iter):
        op = rng.choice(operators)
        parent_a = rng.choice(card_list)
        parents = [parent_a.name]

        try:
            if op == "swap_mechanism":
                if not parent_a.mechanism_tags:
                    continue
                old_tag = rng.choice(sorted(parent_a.mechanism_tags))
                new_tag = rng.choice(all_tags)
                child = swap_mechanism(
                    parent_a,
                    old_tag,
                    new_tag,
                    name=f"mutant_{i}_{parent_a.name}_swap_{new_tag}",
                )
            elif op == "add_bias":
                # Pick a bias the parent doesn't have.
                missing = [b for b in all_biases if b not in parent_a.inductive_biases]
                if not missing:
                    continue
                new_bias = rng.choice(missing)
                child = add_bias(
                    parent_a,
                    new_bias,
                    name=f"mutant_{i}_{parent_a.name}_addbias_{new_bias}",
                )
            elif op == "crossover":
                parent_b = rng.choice(card_list)
                if parent_b.name == parent_a.name:
                    continue
                parents.append(parent_b.name)
                child = crossover(
                    parent_a,
                    parent_b,
                    name=f"mutant_{i}_{parent_a.name}x{parent_b.name}_cross",
                    rng=rng,
                )
            elif op == "stack":
                parent_b = rng.choice(card_list)
                if parent_b.name == parent_a.name:
                    continue
                parents.append(parent_b.name)
                child = stack(
                    parent_a,
                    parent_b,
                    name=f"mutant_{i}_{parent_a.name}_then_{parent_b.name}_stack",
                )
            else:
                continue
        except Exception:  # pragma: no cover -- defensive
            continue

        score = score_card(child, fps)
        delta = score - best_score
        if delta < margin:
            continue
        n_beat += 1

        record = {
            "name": child.name,
            "parents": parents,
            "op": op,
            "predicted_mean_ari": score,
            "delta_vs_best_existing": delta,
            "best_existing_compared_to": best_name,
            "near_duplicate_of_existing": _is_near_duplicate(child, cards),
            "card": _card_to_spec(child),
            "next_implementation_step": suggest_implementation(parents, op, child),
        }
        survivors.append(record)

    # Sort by score desc, dedup by (parents, sorted mechanism tags + biases)
    survivors.sort(key=lambda r: r["predicted_mean_ari"], reverse=True)
    seen_keys: Set[Tuple] = set()
    deduped: List[Dict[str, Any]] = []
    for rec in survivors:
        key = (
            tuple(rec["parents"]),
            tuple(rec["card"]["mechanism_tags"]),
            tuple(rec["card"]["inductive_biases"]),
        )
        if key in seen_keys:
            continue
        seen_keys.add(key)
        deduped.append(rec)
        if len(deduped) >= top_n:
            break

    return {
        "n_evaluated": n_iter,
        "n_beat_existing": n_beat,
        "best_existing_card": {"name": best_name, "predicted_mean_ari": best_score},
        "fingerprint_sample_size": len(fps),
        "margin": margin,
        "seed": seed,
        "top": deduped,
    }


def smoke_test(seed: int = 0, n_iter: int = 100) -> Dict[str, Any]:
    """Inline smoke test — runs the search and prints a brief summary.

    Returns the same report dict that ``random_search`` returns.
    """
    report = random_search(n_iter=n_iter, top_n=10, seed=seed)
    print(f"[mutator-smoke] candidates_evaluated={report['n_evaluated']}")
    print(
        f"[mutator-smoke] candidates_beating_best_existing="
        f"{report['n_beat_existing']} "
        f"(best existing: {report['best_existing_card']['name']} = "
        f"{report['best_existing_card']['predicted_mean_ari']:.4f})"
    )
    print(f"[mutator-smoke] fingerprint_sample_size={report['fingerprint_sample_size']}")
    print("[mutator-smoke] top 3 mutations:")
    for r in report["top"][:3]:
        parents = " + ".join(r["parents"])
        print(
            f"  - {r['name']} | parents={parents} | op={r['op']} "
            f"| predicted_mean_ari={r['predicted_mean_ari']:.4f} "
            f"(delta={r['delta_vs_best_existing']:+.4f})"
        )
    # Pick the most-exciting non-duplicate suggestion.
    exciting = next(
        (r for r in report["top"] if not r["near_duplicate_of_existing"]),
        report["top"][0] if report["top"] else None,
    )
    if exciting is not None:
        print("[mutator-smoke] most-exciting next step:")
        print(f"  -> {exciting['name']}")
        print(f"     {exciting['next_implementation_step']}")
    return report


__all__ = [
    "swap_mechanism",
    "add_bias",
    "crossover",
    "stack",
    "sample_fingerprints",
    "score_card",
    "best_existing_score",
    "random_search",
    "smoke_test",
    "suggest_implementation",
]
