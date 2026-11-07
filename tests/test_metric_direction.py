"""Guards against the failure mode that invalidated Table 4.1 of Maharaj (2024):
ranking a higher-is-better metric in ascending order, which hands rank 1 to the
worst (or failed) algorithm. Every ranking consumer must agree with
``clustbench.metrics.HIGHER_IS_BETTER``, and a perfect clustering must rank
first on every quality metric."""
from __future__ import annotations

import json
import pathlib
import re

import numpy as np
import pandas as pd
import pytest

from clustbench import stats
from clustbench.benchmark import average_ranks, friedman
from clustbench.metrics import HIGHER_IS_BETTER, bundle_scores, higher_is_better

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _blobs(n=300, k=3, seed=0):
    rng = np.random.default_rng(seed)
    centres = np.array([[0, 0], [8, 0], [0, 8]], dtype=float)[:k]
    y = np.repeat(np.arange(k), n // k)
    X = centres[y] + rng.normal(scale=0.6, size=(len(y), 2))
    return X, y


def _rows_for(labelings, X, y, task):
    rows = []
    for algo, labels in labelings.items():
        s = bundle_scores(X, labels, y)
        s["metric_compactness"] = s.pop("compactness")  # as cli.run.flatten stores it
        rows.append({"algo": algo, "dataset_id": "blobs", "n_samples": len(y), "n_features": 2,
                     "k_target": 3, "compactness": 1.0, "outliers": 0, "noise": 0, "density": 1.0,
                     "outlier_extremity": 1.0, "seed": task, **s})
    return rows


@pytest.fixture(scope="module")
def perfect_vs_bad():
    rows = []
    for task in range(4):
        X, y = _blobs(seed=task)
        rng = np.random.default_rng(100 + task)
        labelings = {
            "perfect": y.copy(),
            "noisy": np.where(rng.random(len(y)) < 0.3, rng.integers(0, 3, len(y)), y),
            "random": rng.integers(0, 3, len(y)),
        }
        rows += _rows_for(labelings, X, y, task)
    return pd.DataFrame(rows)


QUALITY = ["ari", "nmi", "silhouette", "dunn", "davies_bouldin", "metric_compactness"]


@pytest.mark.parametrize("metric", QUALITY)
def test_perfect_clustering_ranks_first_in_stats(perfect_vs_bad, metric):
    ranks = stats.average_ranks(stats.task_matrix(perfect_vs_bad, metric))
    assert ranks.index[0] == "perfect", (metric, ranks.to_dict())
    assert ranks.index[-1] == "random", (metric, ranks.to_dict())


@pytest.mark.parametrize("metric", QUALITY)
def test_perfect_clustering_ranks_first_in_benchmark(perfect_vs_bad, metric):
    r = average_ranks(perfect_vs_bad, metric)
    assert r.iloc[0]["algo"] == "perfect", (metric, r.to_dict("records"))
    assert r.iloc[-1]["algo"] == "random", (metric, r.to_dict("records"))


def test_explicit_direction_matches_registry(perfect_vs_bad):
    for metric in QUALITY:
        a = stats.average_ranks(stats.task_matrix(perfect_vs_bad, metric))
        b = stats.average_ranks(stats.task_matrix(perfect_vs_bad, metric, higher_is_better(metric)))
        pd.testing.assert_series_equal(a, b)
        fa = friedman(perfect_vs_bad, metric)
        fb = friedman(perfect_vs_bad, metric, higher_is_better(metric))
        assert fa == fb


def test_unknown_metric_has_no_silent_default():
    with pytest.raises(KeyError):
        higher_is_better("not_a_metric")


def test_every_recorded_metric_has_a_direction():
    from clustbench.schemas import Record

    fields = set(Record.model_fields)
    from clustbench.schemas import MetricBundle

    for m in MetricBundle.model_fields:
        key = "metric_compactness" if m == "compactness" else m
        assert key in HIGHER_IS_BETTER, m
    for m in ("wall_time_s", "rss_delta_mb", "cpu_user_s", "cpu_system_s"):
        assert m in fields and m in HIGHER_IS_BETTER, m


def test_dashboard_directions_match_registry():
    html = (ROOT / "docs" / "index.html").read_text()
    block = re.search(r"const RANK_METRICS = \[(.*?)\];", html, re.S).group(1)
    pairs = re.findall(r'\["(\w+)",\s*(true|false)\]', block)
    assert pairs, "RANK_METRICS not found in dashboard"
    for metric, flag in pairs:
        assert HIGHER_IS_BETTER[metric] == (flag == "true"), metric


def test_cli_summary_uses_registry():
    src = (ROOT / "src" / "clustbench" / "cli" / "run.py").read_text()
    # No hard-coded (metric, bool) pairs may come back.
    assert "higher_is_better(" in src
    assert not re.search(r'\("(ari|nmi|silhouette|davies_bouldin|wall_time_s|rss_delta_mb)",\s*(True|False)\)', src)


def test_dunn_index_is_deterministic():
    from clustbench.metrics import dunn_index

    rng = np.random.default_rng(0)
    X = rng.normal(size=(6000, 3))
    labels = (X[:, 0] > 0).astype(int)  # clusters larger than the 2000 sub-sample
    assert dunn_index(X, labels) == dunn_index(X, labels)
