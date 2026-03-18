"""Top‑level package for clustbench.

This package exposes dataset generators, metrics functions, algorithm registry,
benchmarking harness, and CLI entry points.
"""

from . import datasets, metrics, benchmark, algorithms
from .benchmark import run_benchmark

__all__ = ["datasets", "metrics", "benchmark", "algorithms", "run_benchmark"]
