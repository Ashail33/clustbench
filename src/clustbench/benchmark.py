from __future__ import annotations
import time
import uuid
import warnings
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
from . import stats as _stats

# Datasets that are supposed to be real data. If their generator falls back
# to a synthetic stand-in (e.g. a download failed), the row is tagged
# ``dataset_source="synthetic_fallback"`` and, under ``strict_provenance``,
# the task is refused instead of silently benchmarked.
REAL_DATASETS = {
    "iris", "wine", "breast_cancer", "digits", "olivetti_faces", "glass",
    "vehicle", "segment", "yeast", "ecoli", "ts_ecg200", "ts_trace",
    "graph_karate", "text20news",
}


class ProvenanceError(RuntimeError):
    """A real dataset fell back to synthetic data under strict provenance."""


def generate_with_provenance(dataset_id: str, spec: DataSpec, strict: bool = False):
    """Generate ``(X, y, source)`` and detect synthetic fallbacks.

    Generators announce a fallback with a ``warnings.warn`` mentioning
    "fallback"; we capture those instead of letting them scroll past.
    """
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        X, y = DATASETS[dataset_id](spec)
    fell_back = any("fallback" in str(w.message).lower() for w in caught)
    if dataset_id in REAL_DATASETS:
        source = "synthetic_fallback" if fell_back else "real"
    else:
        source = "synthetic"
    if fell_back and strict:
        msgs = "; ".join(str(w.message) for w in caught)
        raise ProvenanceError(f"{dataset_id}: {msgs}")
    for w in caught:  # re-emit so logs still show them
        warnings.warn_explicit(w.message, w.category, w.filename, w.lineno)
    return X, y, source


@dataclass
class AlgoCfg:
    name: str
    kind: str  # "python" or "external"
    params: Dict[str, Any]
    entry: str | None = None
    # Skip this algorithm on tasks with more than ``max_n`` points (e.g. the
    # O(n^2) methods in a large-n scaling sweep). Skips are logged, not
    # silently dropped.
    max_n: int | None = None
    # Wall-clock budget for one fit_predict call. A run that exceeds it is
    # recorded as an ``error_*.json`` with ``error_type="Timeout"`` (a
    # "did not finish" in scaling reports) instead of stalling the sweep.
    timeout_s: float | None = None


def measure_resources(fn, repeats: int = 1):
    """Run ``fn`` and measure wall time, RSS delta, CPU and I/O.

    With ``repeats > 1`` the call is repeated and ``wall_time_s`` is the
    *median* wall time (all repeats kept in ``wall_time_repeats``); the
    payload (labels, metrics) is taken from the first call. The other
    resource numbers come from the first call too.
    """
    out = _measure_once(fn)
    if repeats > 1:
        times = [out["wall_time_s"]]
        for _ in range(repeats - 1):
            t0 = time.perf_counter()
            fn()
            times.append(time.perf_counter() - t0)
        out["wall_time_repeats"] = times
        out["wall_time_s"] = float(np.median(times))
    return out


def _measure_once(fn):
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


class AlgoTimeout(Exception):
    """fit_predict exceeded its wall-clock budget."""


class _time_limit:
    """SIGALRM-based time limit (Unix, main thread). Long C/BLAS calls are
    interrupted when they next return to Python, so overruns are possible
    but bounded by the longest single native call."""

    def __init__(self, seconds: float | None) -> None:
        self.seconds = seconds

    def __enter__(self):
        import signal
        import threading

        self._armed = bool(self.seconds) and threading.current_thread() is threading.main_thread()
        if self._armed:
            def _raise(signum, frame):
                raise AlgoTimeout(f"exceeded {self.seconds:g}s")

            self._old = signal.signal(signal.SIGALRM, _raise)
            signal.setitimer(signal.ITIMER_REAL, float(self.seconds))
        return self

    def __exit__(self, *exc):
        if self._armed:
            import signal

            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, self._old)
        return False


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
    outlier_extremity: float = 1.0,
    timing_repeats: int = 1,
    default_timeout_s: float | None = None,
    strict_provenance: bool = False,
):
    X, y, dataset_source = generate_with_provenance(
        dataset_id,
        DataSpec(
            n_samples=n,
            n_features=d,
            centers=k,
            compactness=compactness,
            seed=seed,
            outliers=outliers,
            noise=noise,
            density=density,
            outlier_extremity=outlier_extremity,
        ),
        strict=strict_provenance,
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
        outlier_extremity=outlier_extremity,
        dataset_source=dataset_source,
    )
    artifacts = outdir / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    records = []
    task_suffix = (
        f"{dataset_id}_n{n}_d{d}_k{k}_c{compactness}"
        f"_o{outliers}_z{noise}_g{density}_x{outlier_extremity}_s{seed}"
    )
    n_points = int(X.shape[0])
    for cfg in algos:
        if cfg.max_n is not None and n_points > cfg.max_n:
            skip_path = artifacts / f"skipped_{cfg.name}__{task_suffix}.json"
            skip_path.write_text(json.dumps({
                "algo": cfg.name, **task_stub,
                "reason": f"n={n_points} exceeds max_n={cfg.max_n}",
            }, indent=2, default=str))
            print(f"  [skip] {cfg.name} on {task_suffix}: n={n_points} > max_n={cfg.max_n}")
            continue
        if cfg.kind == "python":
            cls = base_algos.ALGO_REGISTRY[cfg.entry.lower()]
            algo = cls(**cfg.params)

            def _fit():
                return {"_res": algo.fit_predict(X, k=k)}

            fit = lambda: measure_resources(_fit, repeats=timing_repeats)  # noqa: E731
            labels_of = lambda payload: payload["_res"].labels  # noqa: E731
            extra_of = lambda payload: payload["_res"].extra  # noqa: E731
            traj_of = lambda payload: payload["_res"].trajectory  # noqa: E731
            kind_tag = ""
        else:

            def _fit_ext():
                return {"_out": run_external(cfg.entry, X, k, cfg.params, artifacts)}

            fit = lambda: measure_resources(_fit_ext, repeats=1)  # noqa: E731
            labels_of = lambda payload: np.load(payload["_out"]["labels_path"])  # noqa: E731
            extra_of = lambda payload: payload["_out"].get("extra", {})  # noqa: E731
            traj_of = lambda payload: None  # noqa: E731
            kind_tag = " (external)"

        try:
            # Only the algorithm call is timed; saving labels and scoring
            # (silhouette is O(n^2)) happen outside the measured window.
            with _time_limit(cfg.timeout_s if cfg.timeout_s is not None else default_timeout_s):
                payload = fit()
            labels = np.asarray(labels_of(payload))
            npy_path = artifacts / f"labels_{cfg.name}__{task_suffix}.npy"
            np.save(npy_path, labels)
            payload.update({
                "n_clusters_found": int(len(set(labels.tolist())) - (1 if -1 in labels else 0)),
                "metrics": bundle_scores(X, labels, y_true=y),
                "extra": extra_of(payload),
                "labels_path": str(npy_path),
                "trajectory": traj_of(payload),
            })
        except Exception as e:
            # One algorithm crashing must not kill the whole run.
            # Write a sentinel JSON so the dashboard knows the algo
            # was attempted on this task and skip the rest.
            err_path = artifacts / f"error_{cfg.name}__{task_suffix}.json"
            err_path.write_text(json.dumps({
                "algo": cfg.name, **task_stub,
                "error_type": "Timeout" if isinstance(e, AlgoTimeout) else type(e).__name__,
                "error_message": str(e)[:500],
            }, indent=2, default=str))
            print(f"  [skip] {cfg.name}{kind_tag} on {task_suffix}: {type(e).__name__}: {str(e)[:120]}")
            continue

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
            wall_time_repeats=payload.get("wall_time_repeats"),
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
    keys = [c for c in [
        "dataset_id",
        "n_samples",
        "n_features",
        "k_target",
        "compactness",
        "outliers",
        "noise",
        "density",
        "outlier_extremity",
        "seed",
    ] if c in df.columns]
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
    keys = [c for c in [
        "dataset_id",
        "n_samples",
        "n_features",
        "k_target",
        "compactness",
        "outliers",
        "noise",
        "density",
        "outlier_extremity",
        "seed",
    ] if c in df.columns]
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
