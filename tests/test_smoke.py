"""End-to-end smoke tests for the clustbench harness.

These run a tiny benchmark over the sample config on a fraction of the
sample size so they complete in a few seconds, and verify that the
expected output files are produced and the registry is fully populated.
"""

from __future__ import annotations

import json
import os
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
    "mri",
    "fmm",
    "lmm",
    "amm",
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


def test_mri_runs():
    """MRI-inspired clustering produces k clusters with non-empty trajectory."""
    from clustbench.datasets import gen_blobs, DataSpec
    from clustbench.algorithms.mri import Mri

    X, y = gen_blobs(DataSpec(n_samples=300, n_features=5, centers=3, compactness=0.5, seed=1))
    res = Mri(n_neighbors=10, n_echoes=4, n_gradient_axes=2).fit_predict(X, k=3)
    assert res.labels.shape == (300,)
    assert 1 <= len(set(int(v) for v in res.labels)) <= 3
    assert res.trajectory and len(res.trajectory) >= 4
    # B0 alignment + local probe + gradient encode + RF pulse + n_echoes acquires + final kmeans.
    expected_steps = 4 + 4 + 1
    assert len(res.trajectory) == expected_steps
    # Signature feature space should be wider than the input.
    assert res.extra["n_signature_features"] >= X.shape[1]
    # The phases we documented should appear in order.
    types = [s.action.get("type") for s in res.trajectory]
    assert types[0] == "b0_align"
    assert types[1] == "local_probe"
    assert types[2] == "gradient_encode"
    assert types[3] == "rf_pulse_90"
    assert types[-1] == "signature_kmeans"


def test_fmm_runs():
    """Fourier mixture model produces k clusters with monotone EM and reports BIC."""
    from clustbench.datasets import gen_blobs, DataSpec
    from clustbench.algorithms.fmm import Fmm

    X, _ = gen_blobs(DataSpec(n_samples=300, n_features=4, centers=3, compactness=0.5, seed=1))
    res = Fmm(n_frequencies=24, n_scales=3, max_iter=10).fit_predict(X, k=3)
    assert res.labels.shape == (300,)
    assert 1 <= len(set(int(v) for v in res.labels)) <= 3
    assert res.trajectory and len(res.trajectory) >= 3
    types = [s.action.get("type") for s in res.trajectory]
    assert types[0] == "fourier_basis"
    assert types[1] == "kmeans_init"
    assert types[2] == "newton_step"
    assert res.extra["feature_dim"] == 2 * 24

    # Newton M-step should be concave EM: log-likelihood non-decreasing.
    lls = [s.state["log_likelihood"] for s in res.trajectory if s.action.get("type") == "newton_step"]
    assert all(lls[i + 1] >= lls[i] - 1e-6 for i in range(len(lls) - 1)), "EM should not decrease ll"
    # BIC should be present and a finite number.
    import math
    assert "bic" in res.extra and math.isfinite(res.extra["bic"])


def test_fmm_auto_k_via_bic():
    """k_search picks the lowest-mean-BIC k when k is not supplied, averaging
    across multiple Fourier bases."""
    from clustbench.datasets import gen_blobs, DataSpec
    from clustbench.algorithms.fmm import Fmm

    X, _ = gen_blobs(DataSpec(n_samples=300, n_features=4, centers=3, compactness=0.5, seed=1))
    fmm = Fmm(
        n_frequencies=24, n_scales=2, max_iter=8,
        k_search=(2, 5), n_basis_samples=2,
    )
    res = fmm.fit_predict(X, k=None)
    assert "k_search_best_k" in res.extra
    assert 2 <= res.extra["k_search_best_k"] <= 5
    profile = res.extra["k_search_bic_profile"]
    assert [p["k"] for p in profile] == [2, 3, 4, 5]
    # Each profile entry has the per-basis breakdown and the mean.
    assert all(len(p["bic_per_basis"]) == 2 for p in profile)
    best = min(profile, key=lambda p: p["bic_mean"])
    assert best["k"] == res.extra["k_search_best_k"]


def test_lmm_handles_non_convex_shapes():
    """LMM with k-NN Laplacian basis recovers non-convex shapes where
    Euclidean mixtures (kmeans/gmm/fmm) bottom out near zero ARI."""
    import numpy as np
    from clustbench.datasets import DATASETS, DataSpec
    from clustbench.algorithms.lmm import Lmm
    from clustbench.algorithms.fmm import Fmm
    from sklearn.metrics import adjusted_rand_score

    lmm_aris, fmm_aris = [], []
    for seed in (1, 2, 3):
        X, y = DATASETS["circles"](DataSpec(n_samples=400, n_features=2, centers=2, compactness=1.0, seed=seed))
        lmm_aris.append(adjusted_rand_score(y, Lmm(n_neighbors=10).fit_predict(X, k=2).labels))
        fmm_aris.append(adjusted_rand_score(y, Fmm().fit_predict(X, k=2).labels))
    # On circles, FMM is near zero; LMM should clearly beat it.
    assert np.mean(lmm_aris) > 0.5, f"LMM mean ARI {np.mean(lmm_aris):.3f} too low"
    assert np.mean(lmm_aris) > np.mean(fmm_aris) + 0.3, \
        f"LMM ({np.mean(lmm_aris):.3f}) should clearly beat FMM ({np.mean(fmm_aris):.3f}) on circles"

    # Trajectory and feature-dim plumbing.
    X, _ = DATASETS["moons"](DataSpec(n_samples=400, n_features=2, centers=2, compactness=1.0, seed=2))
    res = Lmm(n_neighbors=10).fit_predict(X, k=2)
    assert res.trajectory[0].action["type"] == "laplacian_basis"
    assert res.extra["feature_dim"] == 2  # n_eigvecs auto-sized to k


def test_lmm_nystrom_opt_in():
    """Nystrom mode produces sensible labels at much lower cost than full LMM."""
    from clustbench.datasets import gen_blobs, DataSpec
    from clustbench.algorithms.lmm import Lmm
    from sklearn.metrics import adjusted_rand_score

    X, y = gen_blobs(DataSpec(n_samples=600, n_features=4, centers=4, compactness=0.5, seed=1))
    res = Lmm(nystrom=True, n_landmarks=100).fit_predict(X, k=4)
    assert res.labels.shape == (600,)
    # On well-separated blobs Nystrom should match full quality.
    assert adjusted_rand_score(y, res.labels) > 0.9


def test_amm_runs():
    """Autoencoder mixture model returns sensible labels on synthetic blobs."""
    from clustbench.datasets import gen_blobs, DataSpec
    from clustbench.algorithms.amm import Amm
    from sklearn.metrics import adjusted_rand_score

    X, y = gen_blobs(DataSpec(n_samples=300, n_features=10, centers=3, compactness=0.5, seed=1))
    res = Amm(n_components=4, hidden_sizes=(32,), ae_max_iter=50).fit_predict(X, k=3)
    assert res.labels.shape == (300,)
    assert 1 <= len(set(int(v) for v in res.labels)) <= 3
    # Autoencoder is overkill for plain blobs but should still recover well.
    assert adjusted_rand_score(y, res.labels) > 0.5
    assert res.trajectory and res.trajectory[0].action.get("type") == "autoencoder_basis"
    assert res.extra["feature_dim"] == 4


@pytest.mark.skipif(
    not os.environ.get("CLUSTBENCH_TEST_DOWNLOADS"),
    reason="20-newsgroups fetch makes a network round-trip; opt in via CLUSTBENCH_TEST_DOWNLOADS=1",
)
def test_text20news_dataset():
    """gen_text20news fetches and vectorises a small slice of 20-newsgroups."""
    from clustbench.datasets import DATASETS, DataSpec
    import numpy as np

    X, y = DATASETS["text20news"](
        DataSpec(n_samples=200, n_features=64, centers=4, compactness=1.0, seed=1)
    )
    assert X.shape[0] <= 200
    assert X.shape[1] == 64
    assert set(int(v) for v in np.unique(y)) <= {0, 1, 2, 3}
    assert X.dtype == np.float32


def test_consensus_with_fmm():
    """Consensus can mix FMM with centroid- and density-based algorithms."""
    from clustbench.datasets import gen_blobs, DataSpec
    from clustbench.consensus import Consensus

    X, y = gen_blobs(DataSpec(n_samples=300, n_features=4, centers=3, compactness=0.5, seed=1))
    res = Consensus(
        base=["kmeans", "gmm", "fmm"],
        base_params={"fmm": {"n_frequencies": 24, "max_iter": 8}},
    ).fit_predict(X, k=3)
    assert res.labels.shape == (300,)
    assert 1 <= len(set(int(v) for v in res.labels)) <= 3
    assert "fmm" in res.extra["bases"]


def test_fmm_heat_kernel_learns_tau():
    """Per-cluster heat-kernel bandwidth ``tau`` is learned during EM."""
    from clustbench.datasets import gen_blobs, DataSpec
    from clustbench.algorithms.fmm import Fmm

    X, _ = gen_blobs(DataSpec(n_samples=300, n_features=4, centers=3, compactness=0.5, seed=1))
    res = Fmm(
        n_frequencies=24, n_scales=3, max_iter=10,
        learn_bandwidth=True, tau_init=0.1, tau_step=0.5,
    ).fit_predict(X, k=3)
    assert res.extra["learn_bandwidth"] is True
    # tau is bounded and finite.
    assert 0.0 <= res.extra["tau_min"] <= res.extra["tau_mean"] <= res.extra["tau_max"] < 1e4
    # tau should be reflected in the parameter count for BIC.
    assert res.extra["n_params"] == 3 * (2 * 24) + 3 + 2


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
