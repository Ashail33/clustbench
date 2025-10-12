"""Base classes and registry for clustering algorithms."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Optional, Type
import numpy as np


@dataclass
class AlgoResult:
    """Result returned by a clustering algorithm implementation."""
    labels: np.ndarray
    extra: Dict[str, Any]


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
