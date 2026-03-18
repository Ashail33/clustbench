"""Clustering algorithms package.

Importing this package registers all built-in algorithm implementations
in the :data:`ALGO_REGISTRY`.
"""

from .base import Algorithm, AlgoResult, ALGO_REGISTRY, register
from . import kmeans, minibatch_kmeans, dbscan, birch, clarans, consensus

__all__ = ["Algorithm", "AlgoResult", "ALGO_REGISTRY", "register"]
