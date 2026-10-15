"""Single source of training data for the learned routers.

Every learned router trains on the benchmark's own results table. Before this
module each router hard-coded its own path (``learned_router_v8`` even pointed
at a git-ignored run directory, so a fresh checkout trained it on nothing).
Routing all of them through :func:`load_training_rows` gives one place to

* point routers at a different results file
  (``CLUSTBENCH_ROUTER_TRAINING=/path/results.json`` or ``.csv``), and
* hold data out of training
  (``CLUSTBENCH_ROUTER_EXCLUDE_DATASETS=moons,circles``), which is what the
  leave-one-dataset-family-out evaluation in ``scripts/router_lofo.py`` uses
  to measure routers on dataset families they have never seen.

Routers cache their training set per process, so set the environment
variables before the router module is first used (the LOFO script runs each
fold in a fresh subprocess for this reason).
"""
from __future__ import annotations

import csv
import json
import math
import os
import pathlib
from typing import Any, Dict, List

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
DEFAULT_PATH = REPO_ROOT / "docs" / "data" / "results.json"

ENV_PATH = "CLUSTBENCH_ROUTER_TRAINING"
ENV_EXCLUDE = "CLUSTBENCH_ROUTER_EXCLUDE_DATASETS"


def training_path() -> pathlib.Path:
    p = os.environ.get(ENV_PATH)
    return pathlib.Path(p) if p else DEFAULT_PATH


def excluded_datasets() -> set[str]:
    raw = os.environ.get(ENV_EXCLUDE, "")
    return {s.strip() for s in raw.split(",") if s.strip()}


def _coerce(v: Any) -> Any:
    if isinstance(v, str):
        if v == "":
            return None
        try:
            f = float(v)
        except ValueError:
            return v
        if math.isnan(f):
            return None
        return int(f) if f.is_integer() and "." not in v and "e" not in v.lower() else f
    return v


def load_training_rows() -> List[Dict[str, Any]]:
    """Return result rows (dicts) for router training, minus held-out datasets.

    Returns an empty list if the file is missing or unreadable, which every
    router already treats as "no training data".
    """
    path = training_path()
    if not path.exists():
        return []
    try:
        if path.suffix == ".csv":
            with path.open(newline="") as f:
                rows = [{k: _coerce(v) for k, v in r.items()} for r in csv.DictReader(f)]
        else:
            rows = json.loads(path.read_text())
    except Exception:
        return []
    drop = excluded_datasets()
    if drop:
        rows = [r for r in rows if r.get("dataset_id") not in drop]
    return rows
