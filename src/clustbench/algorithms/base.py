"""Base classes and registry for clustering algorithms."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Type
import numpy as np


@dataclass
class Step:
    """One step of an iterative clustering algorithm.

    A trajectory of these forms a state-action sequence that a latent-space
    model can learn over. Everything except ``step_idx`` and ``cost`` is
    optional so different algorithms can populate what's meaningful to them.

    Fields
    ------
    step_idx : int
        Monotonic step counter within this run (starts at 0).
    cost : float
        Scalar objective for the current state (e.g., inertia, medoid cost,
        negative silhouette). Lower is better by convention.
    delta_cost : float | None
        ``cost_after - cost_before`` for the action taken this step. Useful
        as a reward signal.
    accepted : bool
        Whether the candidate action was accepted (e.g., swap that reduced
        cost). For deterministic algos like k-means, set to True.
    action : dict
        Algorithm-specific description of what changed. Examples:
        ``{"type": "swap", "out": 17, "in": 42}`` for CLARANS,
        ``{"type": "reassign"}`` for k-means EM iterations.
    state : dict
        A compact snapshot of the state. Examples: ``{"medoids": [...]}``
        for CLARANS, ``{"centroids": [[...]]}`` for k-means. Kept numeric
        so it serializes cleanly to Parquet.
    """

    step_idx: int
    cost: float
    delta_cost: Optional[float] = None
    accepted: bool = True
    action: Dict[str, Any] = field(default_factory=dict)
    state: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AlgoResult:
    """Result returned by a clustering algorithm implementation."""
    labels: np.ndarray
    extra: Dict[str, Any]
    trajectory: Optional[List[Step]] = None


class Algorithm:
    """Abstract base class for clustering algorithms.

    Implementations must override :meth:`fit_predict` to return an ``AlgoResult``.
    """

    name: str = "algorithm"

    def fit_predict(self, X: np.ndarray, k: Optional[int] = None) -> AlgoResult:
        raise NotImplementedError


ALGO_REGISTRY: Dict[str, Type[Algorithm]] = {}


def register(cls: Type[Algorithm]) -> Type[Algorithm]:
    """Class decorator to register a clustering algorithm under its lower-case class name."""
    ALGO_REGISTRY[cls.__name__.lower()] = cls
    return cls
