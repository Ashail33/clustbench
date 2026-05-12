"""Base classes for regression algorithms — mirror of clustering side."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Type

import numpy as np

from ..algorithms.base import Step


@dataclass
class RegResult:
    """Result returned by a regressor."""

    predictions: np.ndarray
    extra: Dict[str, Any] = field(default_factory=dict)
    trajectory: Optional[List[Step]] = None


class Regressor:
    """Abstract base for regression algorithms.

    Implementations must override :meth:`fit` and :meth:`predict`. A
    default :meth:`fit_predict` is provided.
    """

    name: str = "regressor"

    def fit(self, X: np.ndarray, y: np.ndarray) -> "Regressor":
        raise NotImplementedError

    def predict(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def fit_predict(self, X: np.ndarray, y: np.ndarray) -> RegResult:
        self.fit(X, y)
        preds = self.predict(X)
        return RegResult(predictions=preds, extra={}, trajectory=None)


REG_REGISTRY: Dict[str, Type[Regressor]] = {}


def register_regressor(cls: Type[Regressor]) -> Type[Regressor]:
    REG_REGISTRY[cls.__name__.lower()] = cls
    return cls
