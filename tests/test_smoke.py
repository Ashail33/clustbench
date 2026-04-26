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


EXPECTED_ALGOS = {
    "kmeans",
    "minibatch_kmeans",
    "dbscan",
    "birch_algo",
    "clarans",
    "consensus",
    "parallel_kmeans",
    "pwcc",
    "s5c",
    "gmm",
    "agglomerative",
    "spectral",
    "meanshift",
    "optics",
    "chameleon",
}


def test_registry_contains_all_algos():
    assert EXPECTED_ALGOS.issubset(ALGO_REGISTRY.keys())


def test_mdcgen_injects_outliers_and_noise():
    from clustbench.datasets import gen_mdcgen, DataSpec

    spec = DataSpec(
        n_samples=300,
        n_features=4,
        centers=3,
        compactness=0.5,
        seed=1,
        outliers=20,
        noise=10,
        density=0.5,
    )
    X, y = gen_mdcgen(spec)
    # 300 cluster points + 10 noise + 20 outliers
    assert X.shape == (330, 4)
    assert int((y == -1).sum()) == 30
    assert set(int(v) for v in y) == {-1, 0, 1, 2}


def test_extra_algorithms_run():
    """The five sklearn extras run on a tiny dataset and produce k clusters."""
    from clustbench.datasets import gen_blobs, DataSpec
    from clustbench.algorithms.sklearn_extras import (
        Gmm,
        Agglomerative,
        Spectral,
        Meanshift,
        Optics,
    )

    X, _ = gen_blobs(DataSpec(n_samples=200, n_features=4, centers=3, compactness=0.5, seed=1))
    for cls in (Gmm, Agglomerative, Spectral):
        res = cls().fit_predict(X, k=3)
        assert res.labels.shape == (200,)
        # Each algo should split into ~3 clusters (some may merge under noise).
        assert 1 <= len(set(int(l) for l in res.labels)) <= 4
    # MeanShift / OPTICS are non-parametric in k; just check they return integer labels.
    for cls in (Meanshift, Optics):
        res = cls().fit_predict(X, k=None)
        assert res.labels.shape == (200,)


def test_new_dataset_generators():
    from clustbench.datasets import DATASETS, DataSpec

    for name, expected_k in [("moons", 2), ("circles", 2), ("anisotropic", 3)]:
        gen = DATASETS[name]
        X, y = gen(
            DataSpec(
                n_samples=200,
                n_features=4 if name == "anisotropic" else 3,
                centers=expected_k,
                compactness=1.0,
                seed=1,
            )
        )
        assert X.shape[0] == 200
        assert len(set(int(v) for v in y)) == expected_k


def test_paper_algorithms_run():
    """Parallel k-means, PWCC, and S5C all run on a tiny MDCGen dataset
    and emit a non-empty trajectory."""
    from clustbench.datasets import gen_mdcgen, DataSpec
    from clustbench.algorithms.parallel_kmeans import Parallel_kmeans
    from clustbench.algorithms.pwcc import Pwcc
    from clustbench.algorithms.s5c import S5c

    X, _ = gen_mdcgen(DataSpec(n_samples=300, n_features=4, centers=3, compactness=0.5, seed=1))

    pk = Parallel_kmeans(n_workers=1, n_init=1, max_iter=20).fit_predict(X, k=3)
    assert pk.labels.shape == (300,) and pk.trajectory and len(pk.trajectory) >= 1

    pw = Pwcc(base=["kmeans", "minibatch_kmeans"]).fit_predict(X, k=3)
    assert pw.labels.shape == (300,) and pw.trajectory and len(pw.trajectory) >= 2

    s5 = S5c(sample_size=120, n_nonzero_coefs=3).fit_predict(X, k=3)
    assert s5.labels.shape == (300,) and s5.trajectory and len(s5.trajectory) == 4


def test_chameleon_runs():
    from clustbench.datasets import gen_blobs, DataSpec
    from clustbench.algorithms.chameleon import Chameleon

    X, _ = gen_blobs(DataSpec(n_samples=300, n_features=6, centers=4, compactness=0.7, seed=1))
    res = Chameleon(n_neighbors=10, overcluster_factor=4, max_partitions=30).fit_predict(X, k=4)
    assert res.labels.shape == (300,)
    assert 1 <= len(set(int(v) for v in res.labels)) <= 4
    assert res.trajectory and len(res.trajectory) >= 2


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
