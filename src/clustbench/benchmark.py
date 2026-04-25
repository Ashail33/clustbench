from __future__ import annotations
import time
import uuid
import psutil
import json
import pathlib
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from .datasets import DATASETS, DataSpec
from .metrics import bundle_scores
from .schemas import Record, StepRecord
from .algorithms import base as base_algos
from .algorithms.base import Step
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
    io_before = proc.io_counters() if proc.io_counters() else None
    rss_before = proc.memory_info().rss
    t0 = time.perf_counter()
    payload = fn()
    elapsed = time.perf_counter() - t0
    rss_after = proc.memory_info().rss
    cpu_after = proc.cpu_times()
    io_after = proc.io_counters() if proc.io_counters() else None
    out = {
        "wall_time_s": elapsed,
        "rss_delta_mb": (rss_after - rss_before) / (1024 ** 2),
        "cpu_user_s": cpu_after.user - cpu_before.user,
        "cpu_system_s": cpu_after.system - cpu_before.system,
    }
    if io_before and io_after:
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
    outliers: int = 0,
    noise: int = 0,
    density: float = 1.0,
):
    gen = DATASETS[dataset_id]
    X, y = gen(
        DataSpec(
            n_samples=n,
            n_features=d,
            centers=k,
            compactness=compactness,
            seed=seed,
            outliers=outliers,
            noise=noise,
            density=density,
        )
    )
    task_stub = dict(
        dataset_id=dataset_id,
        n_samples=n,
        n_features=d,
        k_target=k,
        compactness=compactness,
        seed=seed,
        outliers=outliers,
        noise=noise,
        density=density,
    )
    artifacts = outdir / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    records = []
    task_suffix = (
        f"{dataset_id}_n{n}_d{d}_k{k}_c{compactness}"
        f"_o{outliers}_z{noise}_g{density}_s{seed}"
    )
    for cfg in algos:
        if cfg.kind == "python":
            cls = base_algos.ALGO_REGISTRY[cfg.entry.lower()]
            algo = cls(**cfg.params)

            def _inner():
                res = algo.fit_predict(X, k=k)
                labels = res.labels
                npy_path = artifacts / f"labels_{cfg.name}__{task_suffix}.npy"
                np.save(npy_path, labels)
                m = bundle_scores(X, labels, y_true=y)
                return {
                    "n_clusters_found": int(len(set(labels)) - (1 if -1 in labels else 0)),
                    "metrics": m,
                    "extra": res.extra,
                    "labels_path": str(npy_path),
                    "trajectory": res.trajectory,
                }

            payload = measure_resources(_inner)
        else:

            def _inner_ext():
                out = run_external(cfg.entry, X, k, cfg.params, artifacts)
                labels = np.load(out["labels_path"])
                m = bundle_scores(X, labels, y_true=y)
                return {
                    "n_clusters_found": int(len(set(labels)) - (1 if -1 in labels else 0)),
                    "metrics": m,
                    "extra": out.get("extra", {}),
                    "labels_path": out["labels_path"],
                    "trajectory": None,
                }

            payload = measure_resources(_inner_ext)

        trajectory: Optional[List[Step]] = payload.get("trajectory")
        run_id = uuid.uuid4().hex
        trajectory_path: Optional[str] = None
        n_steps: Optional[int] = None
        if trajectory:
            traj_rows = [
                StepRecord(
                    run_id=run_id,
                    algo=cfg.name,
                    **task_stub,
                    step_idx=s.step_idx,
                    cost=s.cost,
                    delta_cost=s.delta_cost,
                    accepted=s.accepted,
                    action=s.action,
                    state=s.state,
                ).model_dump()
                for s in trajectory
            ]
            traj_df = pd.DataFrame(traj_rows)
            traj_stem = f"trajectory_{cfg.name}__{task_suffix}"
            traj_out = artifacts / f"{traj_stem}.parquet"
            try:
                traj_df.to_parquet(traj_out)
            except Exception:
                traj_out = artifacts / f"{traj_stem}.csv"
                traj_df.to_csv(traj_out, index=False)
            trajectory_path = str(traj_out)
            n_steps = len(trajectory)

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
            n_steps=n_steps,
            metrics=payload["metrics"],
            extra=payload["extra"],
            labels_path=payload["labels_path"],
            trajectory_path=trajectory_path,
        )
        json_path = artifacts / f"metrics_{cfg.name}__{task_suffix}.json"
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
