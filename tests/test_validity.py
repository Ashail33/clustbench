"""Tests for the validity controls: statistics, inlier metrics, provenance,
timing repeats, size caps and router training-data isolation."""
from __future__ import annotations

import json
import subprocess
import sys
import warnings

import numpy as np
import pandas as pd
import pytest

from clustbench import stats
from clustbench.benchmark import AlgoCfg, ProvenanceError, generate_with_provenance, run_task
from clustbench.datasets import DATASETS, DataSpec
from clustbench.metrics import bundle_scores, inlier_scores


def _toy_results(n_cfg=6, seeds=(1, 2, 3)):
    rng = np.random.default_rng(0)
    rows = []
    for c in range(n_cfg):
        for s in seeds:
            base = rng.random()
            for algo, shift in (("good", 0.3), ("mid", 0.1), ("bad", 0.0)):
                rows.append({"algo": algo, "dataset_id": f"d{c}", "n_samples": 100, "n_features": 2,
                             "k_target": 2, "compactness": 1.0, "outliers": 0, "noise": 0,
                             "density": 1.0, "outlier_extremity": 1.0, "seed": s,
                             "ari": min(1.0, base * 0.5 + shift + 0.01 * rng.standard_normal())})
    return pd.DataFrame(rows)


def test_holm_matches_hand_computation():
    adj = stats.holm([0.01, 0.04, 0.03])
    # sorted p: 0.01*3=0.03, 0.03*2=0.06, 0.04*1 -> max(0.06, 0.04)=0.06
    assert np.allclose(adj, [0.03, 0.06, 0.06])


def test_nemenyi_cd_matches_demsar_table():
    # Demšar (2006): k=5, N=10 -> CD = 2.728 * sqrt(5*6/60) ~= 1.929
    assert stats.nemenyi_cd(5, 10) == pytest.approx(2.728 * np.sqrt(0.5), rel=1e-3)
    # Large k falls back to the studentized range distribution.
    assert stats.nemenyi_cd(20, 50) > stats.nemenyi_cd(10, 50)


def test_summary_finds_the_real_ordering():
    df = _toy_results()
    out = stats.summarize(df, "ari")
    order = [r["algo"] for r in out["average_ranks"]]
    assert order == ["good", "mid", "bad"]
    pw = {(r["a"], r["b"]): r for r in out["pairwise_wilcoxon_holm"]}
    assert pw[("good", "bad")]["significant"]
    ci = out["bootstrap_ci"]["good"]
    assert ci["ci_low"] <= ci["mean"] <= ci["ci_high"]
    # 6 configurations, each seed of a config resampled together.
    assert ci["n_configs"] == 6
    assert out["seed_variability"]["good"]["seeds_per_config"] == 3


def test_task_matrix_distinguishes_outlier_extremity():
    df = _toy_results(n_cfg=1, seeds=(1,))
    df2 = df.copy()
    df2["outlier_extremity"] = 10.0
    M = stats.task_matrix(pd.concat([df, df2]), "ari")
    assert M.shape[0] == 2


def test_inlier_scores_ignore_injected_outliers():
    y = np.array([0, 0, 1, 1, -1, -1])
    perfect_inliers = np.array([0, 0, 1, 1, 0, 1])  # outliers absorbed by clusters
    s = inlier_scores(perfect_inliers, y)
    assert s["ari_inliers"] == pytest.approx(1.0)
    assert inlier_scores(np.array([0, 0, 1, 1]), np.array([0, 0, 1, 1]))["ari_inliers"] is None
    X = np.random.default_rng(0).normal(size=(6, 2))
    assert "ari_inliers" in bundle_scores(X, perfect_inliers, y_true=y)


def test_provenance_tags_and_strict_mode(monkeypatch):
    spec = DataSpec(n_samples=60, n_features=4, centers=3, compactness=1.0, seed=1)
    _, _, src = generate_with_provenance("mdcgen", spec)
    assert src == "synthetic"

    def fake_real(spec):
        warnings.warn("ECG200 UCR fetch failed; using synthetic sinusoid fallback")
        return DATASETS["blobs"](spec)

    monkeypatch.setitem(DATASETS, "ts_ecg200", fake_real)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _, _, src = generate_with_provenance("ts_ecg200", spec)
    assert src == "synthetic_fallback"
    with pytest.raises(ProvenanceError):
        generate_with_provenance("ts_ecg200", spec, strict=True)


def test_run_task_repeats_and_max_n(tmp_path):
    algos = [AlgoCfg(name="kmeans", kind="python", entry="kmeans", params={"n_init": 1}),
             AlgoCfg(name="capped", kind="python", entry="kmeans", params={}, max_n=10)]
    recs = run_task("blobs", 80, 3, 3, 1.0, 1, algos, tmp_path, timing_repeats=3)
    assert [r["algo"] for r in recs] == ["kmeans"]
    r = recs[0]
    assert len(r["wall_time_repeats"]) == 3
    assert r["wall_time_s"] == pytest.approx(float(np.median(r["wall_time_repeats"])))
    assert r["dataset_source"] == "synthetic"
    assert list(tmp_path.glob("artifacts/skipped_capped__*.json"))


def test_router_training_respects_exclusions(tmp_path):
    rows = [{"algo": "kmeans", "dataset_id": "moons", "ari": 0.3},
            {"algo": "kmeans", "dataset_id": "circles", "ari": 0.0},
            {"algo": "kmeans", "dataset_id": "mdcgen", "ari": 0.9}]
    p = tmp_path / "train.json"
    p.write_text(json.dumps(rows))
    code = ("from clustbench.training_data import load_training_rows; "
            "print(sorted(r['dataset_id'] for r in load_training_rows()))")
    env = {"CLUSTBENCH_ROUTER_TRAINING": str(p), "CLUSTBENCH_ROUTER_EXCLUDE_DATASETS": "moons, circles"}
    import os

    out = subprocess.run([sys.executable, "-c", code], env={**os.environ, **env},
                         capture_output=True, text=True, check=True).stdout
    assert out.strip() == "['mdcgen']"


def test_gmm_accepts_yaml_string_reg_covar_and_recovers_from_singularity():
    from clustbench.algorithms.base import ALGO_REGISTRY

    g = ALGO_REGISTRY["gmm"](reg_covar="1e-4")  # YAML 1.1 parses 1e-4 as a str
    assert g.reg_covar == pytest.approx(1e-4)
    rng = np.random.default_rng(0)
    # Two clusters, one of them collapsed onto a line -> singular covariance.
    X = np.vstack([rng.normal(size=(50, 3)), np.tile([5.0, 5.0, 5.0], (50, 1))])
    res = ALGO_REGISTRY["gmm"]().fit_predict(X, k=2)
    assert len(res.labels) == 100 and res.extra["reg_covar_used"] >= 1e-6


def test_consensus_has_default_base():
    from clustbench.algorithms.base import ALGO_REGISTRY

    X = np.random.default_rng(0).normal(size=(60, 2))
    assert len(ALGO_REGISTRY["consensus"]().fit_predict(X, k=2).labels) == 60


def test_tbmm_concatenates_blocks_and_solves_circles():
    from sklearn.metrics import adjusted_rand_score

    from clustbench.algorithms.base import ALGO_REGISTRY

    X, y = DATASETS["circles"](DataSpec(n_samples=300, n_features=2, centers=2, compactness=1.0, seed=1))
    res = ALGO_REGISTRY["tbmm"](ae_max_iter=50).fit_predict(X, k=2)
    assert set(res.extra["blocks"]) == {"fourier", "laplacian", "autoencoder"}
    assert all(v > 0 for v in res.extra["block_dims"].values())
    assert adjusted_rand_score(y, res.labels) > 0.9
    with pytest.raises(ValueError):
        ALGO_REGISTRY["tbmm"](blocks=("wavelet",))


def test_flhmm_transform_blocks_and_heads():
    from clustbench.algorithms.base import ALGO_REGISTRY
    from clustbench.algorithms.flhmm import hilbert_features, laplace_features

    X, y = DATASETS["blobs"](DataSpec(n_samples=120, n_features=3, centers=3, compactness=1.0, seed=1))
    rng = np.random.default_rng(0)
    L = laplace_features(X, rng, n_s=5)
    assert L.shape == (len(X), 5) and np.all((L > 0) & (L <= 1))
    H = hilbert_features(X, rng, n_dirs=2)
    assert H.shape == (len(X), 6) and np.allclose(H[:, 1] ** 2 + H[:, 2] ** 2, 1.0)
    for head in ("mixture", "kmeans"):
        res = ALGO_REGISTRY["flhmm"](head=head).fit_predict(X, k=3)
        assert len(res.labels) == len(X) and res.extra["head"] == head
        assert set(res.extra["blocks"]) == {"fourier", "laplace", "hilbert"}
    with pytest.raises(ValueError):
        ALGO_REGISTRY["flhmm"](head="dbscan")


def test_flh_cycle_records_one_step_per_view():
    from clustbench.algorithms.base import ALGO_REGISTRY

    X, y = DATASETS["blobs"](DataSpec(n_samples=150, n_features=3, centers=3, compactness=1.0, seed=2))
    for kw in ({}, {"estep": "product", "handoff": 0.5}):
        res = ALGO_REGISTRY["flh_cycle"](n_cycles=4, **kw).fit_predict(X, k=3)
        assert len(res.labels) == len(X)
        views = [s.action["view"] for s in res.trajectory]
        assert views[:3] == ["fourier", "laplace", "hilbert"]
        assert len(views) == 3 * res.extra["cycles"]
    with pytest.raises(ValueError):
        ALGO_REGISTRY["flh_cycle"](estep="median")


def test_timeout_records_did_not_finish(tmp_path):
    import json as _json

    from clustbench.algorithms.base import ALGO_REGISTRY, AlgoResult, Algorithm

    class _Slow(Algorithm):
        def fit_predict(self, X, k=None):
            import time
            t0 = time.time()
            while time.time() - t0 < 5:
                pass
            return AlgoResult(labels=np.zeros(len(X), dtype=int), extra={})

    ALGO_REGISTRY["_slow_test"] = _Slow
    try:
        algos = [AlgoCfg(name="slow", kind="python", entry="_slow_test", params={}, timeout_s=0.3)]
        assert run_task("blobs", 50, 2, 2, 1.0, 1, algos, tmp_path) == []
        err = _json.loads(next(tmp_path.glob("artifacts/error_slow__*.json")).read_text())
        assert err["error_type"] == "Timeout"
    finally:
        ALGO_REGISTRY.pop("_slow_test", None)


def test_isolated_runs_enforce_time_and_memory(tmp_path):
    import json as _json

    from clustbench.algorithms.base import ALGO_REGISTRY, AlgoResult, Algorithm

    class _Hog(Algorithm):
        def fit_predict(self, X, k=None):
            np.ones((40000, 40000))  # ~12.8 GB
            return AlgoResult(labels=np.zeros(len(X), dtype=int), extra={})

    class _Spin(Algorithm):
        def fit_predict(self, X, k=None):
            import time
            time.sleep(30)  # blocking call, not interruptible by SIGALRM handlers
            return AlgoResult(labels=np.zeros(len(X), dtype=int), extra={})

    ALGO_REGISTRY["_hog_test"], ALGO_REGISTRY["_spin_test"] = _Hog, _Spin
    try:
        algos = [AlgoCfg(name="km", kind="python", entry="kmeans", params={"n_init": 1}),
                 AlgoCfg(name="hog", kind="python", entry="_hog_test", params={}),
                 AlgoCfg(name="spin", kind="python", entry="_spin_test", params={}, timeout_s=1.0)]
        recs = run_task("blobs", 80, 2, 2, 1.0, 1, algos, tmp_path, isolate=True, mem_budget_gb=1.0)
        assert [r["algo"] for r in recs] == ["km"] and recs[0]["metrics"]["ari"] is not None
        errs = {p.name.split("__")[0]: _json.loads(p.read_text())["error_type"]
                for p in tmp_path.glob("artifacts/error_*.json")}
        assert errs == {"error_hog": "MemoryError", "error_spin": "Timeout"}
    finally:
        ALGO_REGISTRY.pop("_hog_test", None)
        ALGO_REGISTRY.pop("_spin_test", None)
