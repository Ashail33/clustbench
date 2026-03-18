"""Tests for the benchmark harness and programmatic run_benchmark API."""

from __future__ import annotations

import json
import pathlib

import numpy as np
import pandas as pd
import pytest

from clustbench.benchmark import (
    AlgoCfg,
    average_ranks,
    friedman,
    measure_resources,
    run_benchmark,
    run_task,
)


# ---------------------------------------------------------------------------
# Minimal config fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def minimal_config():
    return {
        "datasets": [
            {
                "id": "blobs",
                "n_samples": [200],
                "n_features": [4],
                "k_targets": [3],
                "compactness": [0.5],
            }
        ],
        "algorithms": [
            {"name": "kmeans", "kind": "python", "entry": "kmeans", "params": {"max_iter": 50}},
            {"name": "dbscan", "kind": "python", "entry": "dbscan", "params": {"eps": 0.5, "min_samples": 3}},
        ],
        "seeds": [0],
    }


@pytest.fixture
def multi_seed_config():
    """Config with multiple seeds, useful for testing ranking / Friedman."""
    return {
        "datasets": [
            {
                "id": "blobs",
                "n_samples": [200],
                "n_features": [4],
                "k_targets": [3],
                "compactness": [0.5],
            },
            {
                "id": "blobs",
                "n_samples": [200],
                "n_features": [4],
                "k_targets": [3],
                "compactness": [1.5],
            },
        ],
        "algorithms": [
            {"name": "kmeans", "kind": "python", "entry": "kmeans", "params": {"max_iter": 50}},
            {"name": "birch", "kind": "python", "entry": "birch_algo", "params": {}},
            {"name": "minibatch_kmeans", "kind": "python", "entry": "minibatch_kmeans", "params": {}},
        ],
        "seeds": [0, 1, 2],
    }


# ---------------------------------------------------------------------------
# measure_resources
# ---------------------------------------------------------------------------

def test_measure_resources_basic():
    def _fn():
        import time
        time.sleep(0.01)
        return {"result": 42}

    out = measure_resources(_fn)
    assert out["wall_time_s"] >= 0.005
    assert "rss_delta_mb" in out
    assert "cpu_user_s" in out
    assert out["result"] == 42


# ---------------------------------------------------------------------------
# run_task
# ---------------------------------------------------------------------------

def test_run_task(tmp_path):
    algos = [
        AlgoCfg(name="kmeans", kind="python", entry="kmeans", params={"max_iter": 50}),
    ]
    records = run_task("blobs", 100, 4, 3, 0.5, 42, algos, tmp_path)
    assert len(records) == 1
    rec = records[0]
    assert rec["algo"] == "kmeans"
    assert rec["dataset_id"] == "blobs"
    assert rec["n_clusters_found"] == 3
    assert "metrics" in rec
    assert "ari" in rec["metrics"]


def test_run_task_writes_artifacts(tmp_path):
    algos = [
        AlgoCfg(name="kmeans", kind="python", entry="kmeans", params={"max_iter": 50}),
    ]
    run_task("blobs", 100, 4, 3, 0.5, 42, algos, tmp_path)
    assert (tmp_path / "artifacts" / "labels_kmeans.npy").exists()
    assert (tmp_path / "artifacts" / "metrics_kmeans.json").exists()

    metric_data = json.loads((tmp_path / "artifacts" / "metrics_kmeans.json").read_text())
    assert "algo" in metric_data


# ---------------------------------------------------------------------------
# run_benchmark
# ---------------------------------------------------------------------------

def test_run_benchmark_returns_dataframe(tmp_path, minimal_config):
    df = run_benchmark(minimal_config, tmp_path / "run1")
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 2  # 2 algorithms × 1 task
    assert set(df["algo"]) == {"kmeans", "dbscan"}


def test_run_benchmark_writes_files(tmp_path, minimal_config):
    outdir = tmp_path / "run1"
    run_benchmark(minimal_config, outdir)
    assert (outdir / "manifest.json").exists()
    assert (outdir / "results.csv").exists()
    assert (outdir / "summary.json").exists()


def test_run_benchmark_manifest_has_env(tmp_path, minimal_config):
    outdir = tmp_path / "run1"
    run_benchmark(minimal_config, outdir)
    manifest = json.loads((outdir / "manifest.json").read_text())
    assert "env" in manifest
    assert "python" in manifest["env"]


def test_run_benchmark_csv_has_metrics(tmp_path, minimal_config):
    outdir = tmp_path / "run1"
    run_benchmark(minimal_config, outdir)
    df = pd.read_csv(outdir / "results.csv")
    for col in ["ari", "nmi", "wall_time_s", "rss_delta_mb"]:
        assert col in df.columns, f"Missing column: {col}"


def test_run_benchmark_multiple_seeds(tmp_path):
    cfg = {
        "datasets": [
            {"id": "blobs", "n_samples": [100], "n_features": [4], "k_targets": [3], "compactness": [0.5]}
        ],
        "algorithms": [
            {"name": "kmeans", "kind": "python", "entry": "kmeans", "params": {}},
        ],
        "seeds": [0, 1, 2],
    }
    df = run_benchmark(cfg, tmp_path / "multi_seed")
    assert len(df) == 3  # 1 algo × 3 seeds


# ---------------------------------------------------------------------------
# average_ranks
# ---------------------------------------------------------------------------

def test_average_ranks(tmp_path, multi_seed_config):
    df = run_benchmark(multi_seed_config, tmp_path / "ranks_run")
    if "ari" in df.columns:
        ranks = average_ranks(df, "ari", higher=True)
        assert set(ranks["algo"]) == {"kmeans", "birch", "minibatch_kmeans"}
        assert ranks["rank"].notna().all()


# ---------------------------------------------------------------------------
# friedman
# ---------------------------------------------------------------------------

def test_friedman_enough_tasks(tmp_path, multi_seed_config):
    df = run_benchmark(multi_seed_config, tmp_path / "friedman_run")
    if "ari" in df.columns:
        result = friedman(df, "ari", higher=True)
        # With 6 tasks (2 datasets × 3 seeds) and 3 algos the test should run
        if "error" not in result:
            assert "statistic" in result
            assert "p_value" in result
            assert result["k"] == 3


def test_friedman_no_complete_tasks():
    # All rows have NaN → should return error dict
    df = pd.DataFrame({
        "dataset_id": ["blobs"] * 2,
        "n_samples": [100] * 2,
        "n_features": [4] * 2,
        "k_target": [3] * 2,
        "compactness": [0.5] * 2,
        "seed": [0] * 2,
        "algo": ["a", "b"],
        "ari": [float("nan"), float("nan")],
    })
    result = friedman(df, "ari")
    assert "error" in result
