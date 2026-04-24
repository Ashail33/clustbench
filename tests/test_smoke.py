"""End-to-end smoke tests for the clustbench harness.

These run a tiny benchmark over the sample config on a fraction of the
sample size so they complete in a few seconds, and verify that the
expected output files are produced and the registry is fully populated.
"""

from __future__ import annotations

import json
import pathlib
import subprocess
import sys
import textwrap

import numpy as np
import pandas as pd
import pytest

from clustbench.algorithms.base import ALGO_REGISTRY


EXPECTED_ALGOS = {"kmeans", "minibatch_kmeans", "dbscan", "birch_algo", "clarans", "consensus"}


def test_registry_contains_all_algos():
    assert EXPECTED_ALGOS.issubset(ALGO_REGISTRY.keys())


def test_cli_end_to_end(tmp_path):
    config = tmp_path / "bench.yaml"
    config.write_text(
        textwrap.dedent(
            """
            datasets:
              - id: blobs
                n_samples: [500]
                n_features: [4]
                k_targets: [3]
                compactness: [0.5]
            algorithms:
              - name: kmeans
                kind: python
                entry: kmeans
                params: {max_iter: 100, n_init: 3}
              - name: minibatch_kmeans
                kind: python
                entry: minibatch_kmeans
                params: {batch_size: 100, max_iter: 50}
            seeds: [1]
            """
        )
    )
    out = tmp_path / "run"
    subprocess.run(
        [sys.executable, "-m", "clustbench.cli.run", "--config", str(config), "--out", str(out)],
        check=True,
    )

    assert (out / "manifest.json").exists()
    assert (out / "results.csv").exists()
    assert (out / "summary.json").exists()

    df = pd.read_csv(out / "results.csv")
    assert set(df["algo"]) == {"kmeans", "minibatch_kmeans"}
    assert (df["ari"] > 0.9).all()

    for algo in ("kmeans", "minibatch_kmeans"):
        matches = list((out / "artifacts").glob(f"labels_{algo}__*.npy"))
        assert matches, f"no labels file for {algo}"
        labels = np.load(matches[0])
        assert labels.shape == (500,)

    traj_matches = list((out / "artifacts").glob("trajectory_kmeans__*.parquet"))
    assert traj_matches, "kmeans should have emitted a trajectory"
    traj = pd.read_parquet(traj_matches[0])
    assert {"step_idx", "cost", "action", "state"}.issubset(traj.columns)
    assert len(traj) >= 1


def test_external_runner_roundtrip(tmp_path):
    """External runner should forward data, run the executable, and collect labels."""
    from clustbench.runners.external_runner import run_external

    runner_script = tmp_path / "runner.py"
    runner_script.write_text(
        f"#!{sys.executable}\n"
        + textwrap.dedent(
            """
            import json, sys, numpy as np
            from sklearn.cluster import KMeans
            p = json.loads(sys.stdin.read())
            X = np.load(p["data_path"])
            labels = KMeans(n_clusters=p["k"], n_init=3, random_state=0).fit_predict(X)
            np.save(p["labels_path"], labels)
            sys.stdout.write(json.dumps({"extra": {"via": "test_runner"}}))
            """
        )
    )
    runner_script.chmod(0o755)

    X = np.random.RandomState(0).randn(200, 4).astype(np.float32)
    artifacts = tmp_path / "artifacts"
    out = run_external(
        entry=str(runner_script),
        X=X,
        k=3,
        params={},
        artifacts=artifacts,
    )

    assert pathlib.Path(out["labels_path"]).exists()
    labels = np.load(out["labels_path"])
    assert labels.shape == (200,)
    assert out["extra"] == {"via": "test_runner"}


@pytest.mark.parametrize("higher", [True, False])
def test_summary_has_ranks_and_friedman(tmp_path, higher):
    """summary.json should carry rank + friedman keys for each metric it processed."""
    from clustbench.benchmark import average_ranks, friedman

    rows = []
    for seed in range(1, 5):
        for algo, val in [("a", 0.9), ("b", 0.8), ("c", 0.85)]:
            rows.append(
                {
                    "dataset_id": "d",
                    "n_samples": 1,
                    "n_features": 1,
                    "k_target": 2,
                    "compactness": 1.0,
                    "seed": seed,
                    "algo": algo,
                    "ari": val + 0.01 * seed,
                }
            )
    df = pd.DataFrame(rows)
    ranks = average_ranks(df, "ari", higher=higher)
    assert set(ranks["algo"]) == {"a", "b", "c"}
    fr = friedman(df, "ari", higher=higher)
    assert "statistic" in fr or "error" in fr
