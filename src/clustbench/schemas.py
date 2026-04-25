from __future__ import annotations
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional


class Task(BaseModel):
    dataset_id: str
    n_samples: int
    n_features: int
    k_target: Optional[int] = None
    compactness: float
    seed: int


class AlgoResultModel(BaseModel):
    labels_path: str
    extra: Dict[str, Any] = Field(default_factory=dict)


class MetricBundle(BaseModel):
    ari: float | None = None
    nmi: float | None = None
    silhouette: float | None = None
    davies_bouldin: float | None = None
    dunn: float | None = None
    compactness: float | None = None
    separation: float | None = None


class StepRecord(BaseModel):
    """One row of a trajectory table.

    Carries the run identifiers so trajectories from many runs can be
    concatenated into a single table and queried by (run_id, algo, seed, …).
    """

    run_id: str
    algo: str
    dataset_id: str
    n_samples: int
    n_features: int
    k_target: int | None = None
    compactness: float
    seed: int
    outliers: int | None = None
    noise: int | None = None
    density: float | None = None
    step_idx: int
    cost: float
    delta_cost: float | None = None
    accepted: bool = True
    action: Dict[str, Any] = Field(default_factory=dict)
    state: Dict[str, Any] = Field(default_factory=dict)


class Record(BaseModel):
    algo: str
    dataset_id: str
    n_samples: int
    n_features: int
    k_target: int | None = None
    compactness: float
    seed: int
    outliers: int | None = None
    noise: int | None = None
    density: float | None = None
    wall_time_s: float
    rss_delta_mb: float
    cpu_user_s: float
    cpu_system_s: float
    read_bytes: int | None = None
    write_bytes: int | None = None
    n_clusters_found: int | None = None
    n_steps: int | None = None
    metrics: MetricBundle
    extra: Dict[str, Any] = Field(default_factory=dict)
    labels_path: str
    trajectory_path: str | None = None
