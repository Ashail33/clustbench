"""Statistical comparison of algorithms across benchmark tasks.

The Friedman test answers "do *any* algorithms differ?". With dozens of
algorithms and tasks it is nearly always significant, so it says little about
whether two *particular* algorithms differ. This module adds what Demšar
(2006) and Benavoli et al. (2016) recommend on top of it:

* :func:`task_matrix` — the (task × algorithm) score matrix every other
  function works on. A task is the full generator identity *including*
  ``outlier_extremity``; seeds are separate tasks.
* :func:`friedman` — Friedman χ² over tasks complete for every algorithm.
* :func:`nemenyi_cd` — the Nemenyi critical difference for average ranks.
* :func:`pairwise_wilcoxon` — Wilcoxon signed-rank tests between pairs of
  algorithms with Holm step-down correction for multiple comparisons.
* :func:`bootstrap_ci` — percentile bootstrap confidence intervals of each
  algorithm's mean score, resampling *dataset configurations* (all seeds of a
  configuration move together, so seed correlation does not shrink the CI).
* :func:`seed_variability` — per-configuration spread across seeds.

Everything here is pure (DataFrame in, plain dicts out) so it can run on a
fresh run directory or on the committed dashboard JSON.
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, rankdata, wilcoxon

# Columns that identify a dataset configuration (everything but the seed).
CONFIG_KEYS: List[str] = [
    "dataset_id",
    "n_samples",
    "n_features",
    "k_target",
    "compactness",
    "outliers",
    "noise",
    "density",
    "outlier_extremity",
]
TASK_KEYS: List[str] = CONFIG_KEYS + ["seed"]

# Studentized range statistic q_alpha / sqrt(2) for the Nemenyi test at
# alpha = 0.05, for k = 2..10 algorithms (Demšar 2006, Table 5). For larger k
# it is computed from scipy's studentized range distribution.
_Q05 = {2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850, 7: 2.949, 8: 3.031, 9: 3.102, 10: 3.164}


def _keys(df: pd.DataFrame, keys: Sequence[str]) -> List[str]:
    return [c for c in keys if c in df.columns]


def _numeric(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Copy of ``df`` with ``metric`` as float and rows lacking it dropped.

    Metrics that are only defined on some tasks (``ari_inliers``) arrive as
    object columns mixing None and floats after concatenation.
    """
    d = df.copy()
    d[metric] = pd.to_numeric(d[metric], errors="coerce")
    return d[d[metric].notna()]


def task_matrix(
    df: pd.DataFrame,
    metric: str,
    higher: bool = True,
    algos: Optional[Iterable[str]] = None,
    complete_only: bool = True,
) -> pd.DataFrame:
    """Return a (task × algo) matrix of ``metric`` with higher = better.

    Duplicate (task, algo) rows are averaged. When ``complete_only`` is set,
    tasks missing any algorithm are dropped so every column is comparable.
    """
    keys = _keys(df, TASK_KEYS)
    d = _numeric(df, metric)
    if algos is not None:
        d = d[d["algo"].isin(list(algos))]
    for k in keys:
        d[k] = d[k].fillna(-1) if d[k].dtype != object else d[k].fillna("")
    M = d.pivot_table(index=keys, columns="algo", values=metric, aggfunc="mean").astype(float)
    if not higher:
        M = -M
    if complete_only:
        M = M.dropna(axis=0, how="any")
    return M


def average_ranks(M: pd.DataFrame) -> pd.Series:
    """Average rank per algorithm (1 = best) over the rows of ``M``."""
    R = M.apply(lambda row: pd.Series(rankdata(-row.values, method="average"), index=row.index), axis=1)
    return R.mean(axis=0).sort_values()


def friedman(M: pd.DataFrame) -> Dict[str, float]:
    if M.shape[0] < 2 or M.shape[1] < 3:
        return {"error": "need >= 2 complete tasks and >= 3 algorithms"}
    stat, p = friedmanchisquare(*[M[c].values for c in M.columns])
    return {"statistic": float(stat), "p_value": float(p), "k": int(M.shape[1]), "n_tasks": int(M.shape[0])}


def _q_alpha(k: int, alpha: float = 0.05) -> float:
    if alpha == 0.05 and k in _Q05:
        return _Q05[k]
    from scipy.stats import studentized_range

    return float(studentized_range.ppf(1 - alpha, k, np.inf) / np.sqrt(2))


def nemenyi_cd(k: int, n_tasks: int, alpha: float = 0.05) -> float:
    """Critical difference in average rank for the Nemenyi post-hoc test."""
    return _q_alpha(k, alpha) * float(np.sqrt(k * (k + 1) / (6.0 * n_tasks)))


def holm(pvals: Sequence[float]) -> np.ndarray:
    """Holm step-down adjusted p-values (monotone, capped at 1)."""
    p = np.asarray(pvals, dtype=float)
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m)
    running = 0.0
    for i, idx in enumerate(order):
        running = max(running, (m - i) * p[idx])
        adj[idx] = min(1.0, running)
    return adj


def pairwise_wilcoxon(
    M: pd.DataFrame,
    pairs: Optional[Sequence[tuple]] = None,
    alpha: float = 0.05,
) -> List[Dict[str, object]]:
    """Wilcoxon signed-rank tests with Holm correction.

    ``pairs`` defaults to every pair of columns. Returns one dict per pair
    with the median paired difference, raw and Holm-adjusted p-values, and
    whether the difference survives correction.
    """
    cols = list(M.columns)
    if pairs is None:
        pairs = [(a, b) for i, a in enumerate(cols) for b in cols[i + 1:]]
    out = []
    raw = []
    for a, b in pairs:
        diff = (M[a] - M[b]).values
        nz = diff[np.abs(diff) > 1e-12]
        if len(nz) < 1:
            p = 1.0
        else:
            try:
                p = float(wilcoxon(M[a].values, M[b].values, zero_method="wilcox").pvalue)
            except ValueError:
                p = 1.0
        raw.append(p)
        out.append({"a": a, "b": b, "median_diff": float(np.median(diff)),
                    "mean_diff": float(np.mean(diff)), "n_nonzero": int(len(nz)), "p": p})
    adj = holm(raw) if raw else []
    for r, pa in zip(out, adj):
        r["p_holm"] = float(pa)
        r["significant"] = bool(pa < alpha)
    return out


def bootstrap_ci(
    df: pd.DataFrame,
    metric: str,
    n_boot: int = 2000,
    level: float = 0.95,
    seed: int = 0,
) -> Dict[str, Dict[str, float]]:
    """Cluster bootstrap CI of each algorithm's mean ``metric``.

    The resampling unit is the dataset *configuration*: each bootstrap
    replicate draws configurations with replacement and keeps every seed of
    each, so the interval reflects between-configuration variability rather
    than treating correlated seeds as independent evidence.
    """
    rng = np.random.default_rng(seed)
    ck = _keys(df, CONFIG_KEYS)
    d = _numeric(df, metric)
    for k in ck:
        d[k] = d[k].fillna(-1) if d[k].dtype != object else d[k].fillna("")
    # Per (config, algo): mean over seeds.
    per = d.groupby(ck + ["algo"], dropna=False)[metric].mean().unstack("algo")
    lo_q, hi_q = (1 - level) / 2, 1 - (1 - level) / 2
    out: Dict[str, Dict[str, float]] = {}
    n = per.shape[0]
    idx = rng.integers(0, n, size=(n_boot, n))
    for a in per.columns:
        v = per[a].values
        ok = ~np.isnan(v)
        if ok.sum() == 0:
            continue
        boots = np.array([np.nanmean(v[i]) for i in idx])
        out[a] = {
            "mean": float(np.nanmean(v)),
            "ci_low": float(np.nanquantile(boots, lo_q)),
            "ci_high": float(np.nanquantile(boots, hi_q)),
            "n_configs": int(ok.sum()),
        }
    return out


def seed_variability(df: pd.DataFrame, metric: str) -> Dict[str, Dict[str, float]]:
    """Mean and max across configurations of the across-seed std per algorithm."""
    ck = _keys(df, CONFIG_KEYS)
    d = _numeric(df, metric)
    for k in ck:
        d[k] = d[k].fillna(-1) if d[k].dtype != object else d[k].fillna("")
    g = d.groupby(ck + ["algo"], dropna=False)[metric]
    sd = g.std(ddof=1).dropna()
    cnt = g.count()
    out = {}
    for a, s in sd.groupby(level="algo"):
        out[a] = {"mean_seed_sd": float(s.mean()), "max_seed_sd": float(s.max()),
                  "seeds_per_config": float(cnt.xs(a, level="algo").median())}
    return out


def summarize(
    df: pd.DataFrame,
    metric: str,
    higher: bool = True,
    top_k: int = 10,
    n_boot: int = 2000,
) -> Dict[str, object]:
    """One-stop summary: ranks, Friedman, Nemenyi CD, Holm-Wilcoxon, CIs.

    Pairwise tests are run between the ``top_k`` algorithms by average rank
    (all pairs) plus each of the rest against the best, which keeps the
    number of comparisons, and so the Holm penalty, manageable.
    """
    M = task_matrix(df, metric, higher)
    if M.shape[0] < 2:
        return {"error": "no complete tasks"}
    ranks = average_ranks(M)
    fr = friedman(M)
    cd = nemenyi_cd(M.shape[1], M.shape[0])
    order = list(ranks.index)
    top = order[:top_k]
    pairs = [(a, b) for i, a in enumerate(top) for b in top[i + 1:]]
    best = order[0]
    pairs += [(best, b) for b in order[top_k:]]
    pw = pairwise_wilcoxon(M, pairs)
    out: Dict[str, object] = {
        "average_ranks": [{"algo": a, "rank": float(r)} for a, r in ranks.items()],
        "friedman": fr,
        "nemenyi_cd": float(cd),
        "pairwise_wilcoxon_holm": pw,
    }
    if higher:
        out["bootstrap_ci"] = bootstrap_ci(df, metric, n_boot=n_boot)
        out["seed_variability"] = seed_variability(df, metric)
    return out
