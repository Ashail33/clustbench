"""CLI entrypoint for clustbench."""

from __future__ import annotations
import argparse
import json
import pathlib
import platform
import time
from typing import List

import yaml
import pandas as pd

from ..benchmark import run_task, AlgoCfg, average_ranks, friedman
from .. import stats as cb_stats


def environment_info(threads: int | None) -> dict:
    """Hardware / software context needed to interpret timing numbers."""
    import hashlib
    import importlib.metadata as md
    import os
    import subprocess

    import psutil

    info: dict = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_count_logical": psutil.cpu_count(logical=True),
        "cpu_count_physical": psutil.cpu_count(logical=False),
        "memory_gb": round(psutil.virtual_memory().total / 1024 ** 3, 2),
        "thread_limit": threads,
    }
    try:
        for line in open("/proc/cpuinfo"):
            if line.startswith("model name"):
                info["cpu_model"] = line.split(":", 1)[1].strip()
                break
    except OSError:
        pass
    versions = {}
    for pkg in ("numpy", "scipy", "scikit-learn", "pandas", "networkx", "torch"):
        try:
            versions[pkg] = md.version(pkg)
        except md.PackageNotFoundError:
            pass
    info["packages"] = versions
    try:
        from threadpoolctl import threadpool_info

        info["threadpools"] = [
            {k: t.get(k) for k in ("internal_api", "num_threads", "version")}
            for t in threadpool_info()
        ]
    except Exception:
        pass
    try:
        here = pathlib.Path(__file__).resolve().parent
        info["git_commit"] = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=here, capture_output=True, text=True, timeout=5
        ).stdout.strip() or None
        info["git_dirty"] = bool(subprocess.run(
            ["git", "status", "--porcelain"], cwd=here, capture_output=True, text=True, timeout=5
        ).stdout.strip())
    except Exception:
        pass
    info["env_threads"] = {k: os.environ.get(k) for k in
                           ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")}
    return info


def main() -> None:
    parser = argparse.ArgumentParser(description="Run clustbench benchmark")
    parser.add_argument("--config", type=pathlib.Path, required=True)
    parser.add_argument("--out", type=pathlib.Path, required=True)
    args = parser.parse_args()

    cfg_text = args.config.read_text()
    cfg = yaml.safe_load(cfg_text)
    outdir = args.out
    outdir.mkdir(parents=True, exist_ok=True)

    # Validity controls (all optional; see docs/METHODOLOGY.md):
    #   timing: {repeats: 3, threads: 1}  -> median-of-N wall time, pinned BLAS/OpenMP threads
    #   strict_provenance: true           -> refuse real datasets that fell back to synthetic
    timing = cfg.get("timing", {}) or {}
    repeats = int(timing.get("repeats", 1))
    threads = timing.get("threads")
    strict = bool(cfg.get("strict_provenance", False))
    # resources: {memory_limit_gb: 6} caps this process's address space, so an
    # algorithm that tries to allocate an n x n matrix at large n raises
    # MemoryError (recorded as a failed run) instead of being OOM-killed and
    # taking the whole sweep with it.
    mem_gb = (cfg.get("resources", {}) or {}).get("memory_limit_gb")
    if mem_gb:
        import resource

        limit = int(float(mem_gb) * 1024 ** 3)
        resource.setrlimit(resource.RLIMIT_AS, (limit, limit))
    if threads is not None:
        from threadpoolctl import threadpool_limits

        threadpool_limits(limits=int(threads))

    import hashlib

    manifest = {
        "config": cfg,
        "config_sha256": hashlib.sha256(cfg_text.encode()).hexdigest(),
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "env": environment_info(int(threads) if threads is not None else None),
    }
    (outdir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    # Prepare algorithm configurations
    algos: List[AlgoCfg] = [
        AlgoCfg(name=a["name"], kind=a.get("kind", "python"), entry=a["entry"],
                params=a.get("params", {}), max_n=a.get("max_n"),
                timeout_s=a.get("timeout_s"))
        for a in cfg["algorithms"]
    ]

    if timing.get("warmup", True):
        warm_up(algos)

    rows: List[dict] = []
    for ds in cfg["datasets"]:
        dataset_id = ds["id"]
        outliers_grid = ds.get("outliers", [0])
        noise_grid = ds.get("noise", [0])
        density_grid = ds.get("density", [1.0])
        outlier_extremity_grid = ds.get("outlier_extremity", [1.0])
        for n in ds["n_samples"]:
            for d in ds["n_features"]:
                for k in ds["k_targets"]:
                    for c in ds["compactness"]:
                        for o in outliers_grid:
                            for z in noise_grid:
                                for g in density_grid:
                                    for x in outlier_extremity_grid:
                                        for seed in cfg.get("seeds", [42]):
                                            rows.extend(
                                                run_task(
                                                    dataset_id,
                                                    n,
                                                    d,
                                                    k,
                                                    c,
                                                    seed,
                                                    algos,
                                                    outdir,
                                                    outliers=int(o),
                                                    noise=int(z),
                                                    density=float(g),
                                                    outlier_extremity=float(x),
                                                    timing_repeats=repeats,
                                                    default_timeout_s=timing.get("timeout_s"),
                                                    strict_provenance=strict,
                                                )
                                            )

    df = results_frame(rows)
    write_results(df, outdir)
    write_summary(df, outdir)
    print(f"Wrote results to {outdir}")


# Metric names that collide with task-identity columns (notably
# `compactness`) get a `metric_` prefix so the experiment setting wins.
PROTECTED = {"compactness", "outliers", "noise", "density", "seed", "k_target"}


def results_frame(rows: List[dict]) -> pd.DataFrame:
    """Flatten Record dicts into the tidy results table.

    Metrics become top-level columns. ``extra`` (algorithm-specific, with a
    different shape per algorithm) is stored as a JSON string so the table
    can always be written to Parquet.
    """

    def flatten(r: dict) -> dict:
        m = r.get("metrics") or {}
        base = {k: v for k, v in r.items() if k != "metrics"}
        for k, v in m.items():
            if k in PROTECTED and k in base:
                base[f"metric_{k}"] = v
            else:
                base[k] = v
        if isinstance(base.get("extra"), (dict, list)):
            base["extra"] = json.dumps(base["extra"], default=str)
        return base

    return pd.DataFrame([flatten(dict(r)) for r in rows])


def write_results(df: pd.DataFrame, outdir: pathlib.Path) -> None:
    df.to_csv(outdir / "results.csv", index=False)
    try:
        df.to_parquet(outdir / "results.parquet")
    except Exception as e:  # CSV is still there; make the failure visible
        print(f"WARNING: could not write results.parquet ({type(e).__name__}: {e}); results.csv only")


def write_summary(df: pd.DataFrame, outdir: pathlib.Path) -> dict:
    """Average ranks, Friedman and post-hoc statistics -> ``summary.json``."""
    metrics_list = [
        ("ari", True),
        ("nmi", True),
        ("silhouette", True),
        ("davies_bouldin", False),
        ("wall_time_s", False),
        ("rss_delta_mb", False),
    ]
    summary = {}
    for metric, higher in metrics_list:
        if metric not in df.columns:
            continue
        try:
            ranks = average_ranks(df, metric, higher).to_dict(orient="records")
            fr = friedman(df, metric, higher)
            summary[metric] = {"average_ranks": ranks, "friedman": fr}
        except Exception as e:
            summary[metric] = {"error": str(e)}
    # Post-hoc statistics for the quality metrics: Nemenyi critical
    # difference, Holm-corrected pairwise Wilcoxon tests, configuration-level
    # bootstrap CIs and seed variability (clustbench.stats).
    for metric in ("ari", "nmi", "ari_inliers"):
        if metric not in df.columns or df[metric].notna().sum() == 0:
            continue
        try:
            post = cb_stats.summarize(df, metric, higher=True)
            summary.setdefault(metric, {})["posthoc"] = {
                k: v for k, v in post.items() if k not in ("average_ranks", "friedman")
            }
            if "average_ranks" not in summary[metric]:
                summary[metric]["average_ranks"] = post.get("average_ranks")
                summary[metric]["friedman"] = post.get("friedman")
        except Exception as e:
            summary.setdefault(metric, {})["posthoc"] = {"error": str(e)}
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def warm_up(algos: List[AlgoCfg]) -> None:
    """Call every Python algorithm once on a tiny dataset, untimed.

    Learned routers train (and RL checkpoints load) on first use; without a
    warm-up that one-off cost lands in the first task's wall time.
    """
    import warnings

    import numpy as np

    from ..algorithms import base as base_algos

    rng = np.random.default_rng(0)
    X = np.vstack([rng.normal(0, 0.3, (40, 4)), rng.normal(3, 0.3, (40, 4))]).astype(np.float32)
    for cfg in algos:
        if cfg.kind != "python":
            continue
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                base_algos.ALGO_REGISTRY[cfg.entry.lower()](**cfg.params).fit_predict(X, k=2)
        except Exception as e:  # the real run will log the failure
            print(f"  [warm-up] {cfg.name}: {type(e).__name__}: {str(e)[:100]}")


if __name__ == "__main__":
    main()
