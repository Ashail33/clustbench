from __future__ import annotations

import time
import psutil
import json
import pathlib
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from .datasets import DATASETS, DataSpec
from .metrics import bundle_scores
from .schemas import Record
from .algorithms import base as base_algos
from .runners.external_runner import run_external
from scipy.stats import friedmanchisquare


@dataclass
class AlgoCfg:
    name: str
    kind: str  # "python" or "external"
    params: Dict[str, Any]
    entry: str | None = None


def measure_resources(fn):
    proc = psutil.Process()
    cpu_before = proc.cpu_times()
    try:
        io_before = proc.io_counters()
    except (AttributeError, NotImplementedError, psutil.AccessDenied):
        io_before = None
    rss_before = proc.memory_info().rss
    t0 = time.perf_counter()
    payload = fn()
    elapsed = time.perf_counter() - t0
    rss_after = proc.memory_info().rss
    cpu_after = proc.cpu_times()
    try:
        io_after = proc.io_counters()
    except (AttributeError, NotImplementedError, psutil.AccessDenied):
        io_after = None
    out = {
        "wall_time_s": elapsed,
        "rss_delta_mb": (rss_after - rss_before) / (1024 ** 2),
        "cpu_user_s": cpu_after.user - cpu_before.user,
        "cpu_system_s": cpu_after.system - cpu_before.system,
    }
    if io_before is not None and io_after is not None:
        out["read_bytes"] = io_after.read_bytes - io_before.read_bytes
        out["write_bytes"] = io_after.write_bytes - io_before.write_bytes
    out.update(payload)
    return out


def run_task(
    dataset_id: str,
    n: int,
    d: int,
    k: int,
    compactness: float,
    seed: int,
    algos: List[AlgoCfg],
    outdir: pathlib.Path,
):
    gen = DATASETS[dataset_id]
    X, y = gen(DataSpec(n, d, k, compactness, seed))
    task_stub = dict(
        dataset_id=dataset_id,
        n_samples=n,
        n_features=d,
        k_target=k,
        compactness=compactness,
        seed=seed,
    )
    artifacts = outdir / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    records = []
    for cfg in algos:
        if cfg.kind == "python":
            cls = base_algos.ALGO_REGISTRY[cfg.entry.lower()]
            algo = cls(**cfg.params)
            # Capture loop variables by value to avoid late-binding closure issues
            _algo = algo
            _k = k
            _X = X
            _y = y
            _cfg_name = cfg.name
            _artifacts = artifacts

            def _inner(
                _algo=_algo,
                _X=_X,
                _k=_k,
                _y=_y,
                _cfg_name=_cfg_name,
                _artifacts=_artifacts,
            ):
                res = _algo.fit_predict(_X, k=_k)
                labels = res.labels
                npy_path = _artifacts / f"labels_{_cfg_name}.npy"
                np.save(npy_path, labels)
                m = bundle_scores(_X, labels, y_true=_y)
                return {
                    "n_clusters_found": int(len(set(labels)) - (1 if -1 in labels else 0)),
                    "metrics": m,
                    "extra": res.extra,
                    "labels_path": str(npy_path),
                }

            payload = measure_resources(_inner)
        else:
            _entry = cfg.entry
            _X = X
            _k = k
            _cfg_params = cfg.params
            _artifacts = artifacts
            _y = y

            def _inner_ext(
                _entry=_entry,
                _X=_X,
                _k=_k,
                _cfg_params=_cfg_params,
                _artifacts=_artifacts,
                _y=_y,
            ):
                out = run_external(_entry, _X, _k, _cfg_params, _artifacts)
                labels = np.load(out["labels_path"])
                m = bundle_scores(_X, labels, y_true=_y)
                return {
                    "n_clusters_found": int(len(set(labels)) - (1 if -1 in labels else 0)),
                    "metrics": m,
                    "extra": out.get("extra", {}),
                    "labels_path": out["labels_path"],
                }

            payload = measure_resources(_inner_ext)
        rec = Record(
            algo=cfg.name,
            **task_stub,
            wall_time_s=payload["wall_time_s"],
            rss_delta_mb=payload["rss_delta_mb"],
            cpu_user_s=payload["cpu_user_s"],
            cpu_system_s=payload["cpu_system_s"],
            read_bytes=payload.get("read_bytes"),
            write_bytes=payload.get("write_bytes"),
            n_clusters_found=payload["n_clusters_found"],
            metrics=payload["metrics"],
            extra=payload["extra"],
            labels_path=payload["labels_path"],
        )
        json_path = artifacts / f"metrics_{cfg.name}.json"
        json_path.write_text(json.dumps(rec.model_dump(), indent=2, default=float))
        records.append(rec.model_dump())
    return records


def average_ranks(df: pd.DataFrame, metric: str, higher=True) -> pd.DataFrame:
    keys = [
        "dataset_id",
        "n_samples",
        "n_features",
        "k_target",
        "compactness",
        "seed",
    ]
    tasks = df[keys].drop_duplicates()
    ranks: List[Dict[str, Any]] = []
    for _, t in tasks.iterrows():
        mask = (df[keys] == t.values).all(axis=1)
        s = df[mask][["algo", metric]].copy()
        if not higher:
            s[metric] = -s[metric]
        r = s.set_index("algo")[metric].rank(ascending=False, method="average")
        ranks.extend([{"algo": a, "rank": float(rk)} for a, rk in r.items()])
    return (
        pd.DataFrame(ranks)
        .groupby("algo", as_index=False)
        .mean()
        .sort_values("rank")
    )


def friedman(df: pd.DataFrame, metric: str, higher=True):
    keys = [
        "dataset_id",
        "n_samples",
        "n_features",
        "k_target",
        "compactness",
        "seed",
    ]
    algos = sorted(df["algo"].unique())
    tasks = df[keys].drop_duplicates().reset_index(drop=True)
    M: List[List[float]] = []
    for _, t in tasks.iterrows():
        mask = (df[keys] == t.values).all(axis=1)
        row = []
        for a in algos:
            v = df[mask & (df["algo"] == a)][metric].values
            row.append(v[0] if len(v) else np.nan)
        M.append(row)
    M = np.array(M, dtype=float)
    if not higher:
        M = -M
    M = M[~np.isnan(M).any(axis=1)]
    if M.shape[0] == 0:
        return {"error": "no complete tasks"}
    stat, p = friedmanchisquare(*[M[:, i] for i in range(M.shape[1])])
    return {
        "statistic": float(stat),
        "p_value": float(p),
        "k": len(algos),
        "n_tasks": int(M.shape[0]),
    }


def _flatten_record(r: dict) -> dict:
    """Hoist nested ``metrics`` dict to top-level columns."""
    m = r.get("metrics", {})
    base = {k: v for k, v in r.items() if k != "metrics"}
    base.update(m)
    return base


def run_benchmark(
    config: dict,
    outdir: str | pathlib.Path,
    *,
    firebase_bucket: Optional[str] = None,
    firebase_credentials: Optional[str] = None,
) -> pd.DataFrame:
    """Run a full benchmark programmatically.

    Parameters
    ----------
    config:
        Benchmark configuration dict with the same structure as a YAML config
        file (see ``configs/benchmark.sample.yaml`` for the format).
    outdir:
        Directory where results will be written.
    firebase_bucket:
        Optional Firebase Storage bucket name.  When provided the run results
        are uploaded to Firebase after being written locally.
    firebase_credentials:
        Path to a Firebase service-account JSON key file.  Falls back to the
        ``GOOGLE_APPLICATION_CREDENTIALS`` environment variable or Application
        Default Credentials when omitted.

    Returns
    -------
    pd.DataFrame
        Tidy results table with one row per algorithm × dataset × seed.

    Example
    -------
    >>> import yaml
    >>> from clustbench.benchmark import run_benchmark
    >>> cfg = yaml.safe_load(open("configs/benchmark.sample.yaml"))
    >>> df = run_benchmark(cfg, "runs/my_run")
    >>> print(df[["algo", "ari", "nmi"]].groupby("algo").mean())
    """
    import platform

    outdir = pathlib.Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "config": config,
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "env": {
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
    }
    (outdir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    algos: List[AlgoCfg] = [
        AlgoCfg(
            name=a["name"],
            kind=a.get("kind", "python"),
            entry=a["entry"],
            params=a.get("params", {}),
        )
        for a in config["algorithms"]
    ]

    rows: List[dict] = []
    for ds in config["datasets"]:
        dataset_id = ds["id"]
        for n in ds["n_samples"]:
            for d in ds["n_features"]:
                for k in ds["k_targets"]:
                    for c in ds["compactness"]:
                        for seed in config.get("seeds", [42]):
                            rows.extend(
                                run_task(dataset_id, n, d, k, c, seed, algos, outdir)
                            )

    flat_rows = [_flatten_record(r) for r in rows]
    df = pd.DataFrame(flat_rows)

    try:
        df.to_parquet(outdir / "results.parquet")
    except Exception:
        pass
    df.to_csv(outdir / "results.csv", index=False)

    metrics_cfg = [
        ("ari", True),
        ("nmi", True),
        ("silhouette", True),
        ("davies_bouldin", False),
        ("wall_time_s", False),
        ("rss_delta_mb", False),
    ]
    summary: dict = {}
    for metric, higher in metrics_cfg:
        if metric not in df.columns:
            continue
        try:
            ranks = average_ranks(df, metric, higher).to_dict(orient="records")
            fr = friedman(df, metric, higher)
            summary[metric] = {"average_ranks": ranks, "friedman": fr}
        except Exception as e:
            summary[metric] = {"error": str(e)}
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2))

    if firebase_bucket:
        from .storage.firebase_storage import FirebaseStorageClient

        client = FirebaseStorageClient(
            bucket_name=firebase_bucket,
            credentials_path=firebase_credentials,
        )
        run_name = outdir.name
        uploaded = client.upload_run(outdir, run_name=run_name)
        upload_log = outdir / "firebase_upload.json"
        upload_log.write_text(json.dumps({"uploaded": uploaded}, indent=2))

    return df
