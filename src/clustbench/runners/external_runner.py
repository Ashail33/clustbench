"""External algorithm runner using a JSON-over-stdin/stdout protocol.

Protocol
--------
Clustbench launches the executable given by ``entry`` and writes a JSON
payload to its stdin:

    {
        "data_path": "<abs path to .npy file with X>",
        "k": <int or null>,
        "params": {...},
        "artifacts_dir": "<abs path>",
        "labels_path": "<abs path to the .npy file the runner must write>"
    }

The executable must write the predicted integer labels to ``labels_path``
(as a NumPy ``.npy`` array) and may print a JSON response to stdout:

    {"extra": {...optional algo-specific metadata...}}

This module handles the handshake and returns a dict with ``labels_path``
and ``extra`` so the caller can load the labels.
"""

from __future__ import annotations

import json
import os
import pathlib
import subprocess
import sys
import tempfile
from typing import Any, Dict, Optional

import numpy as np


def run_external(
    entry: str,
    X: np.ndarray,
    k: Optional[int],
    params: Dict[str, Any],
    artifacts: pathlib.Path,
    timeout: Optional[float] = None,
) -> Dict[str, Any]:
    """Invoke an external clustering executable and return its result.

    Parameters
    ----------
    entry : str
        Path to the executable (or command on PATH) to invoke.
    X : np.ndarray
        Input data; written to a temporary ``.npy`` file.
    k : int | None
        Target number of clusters (may be None for density-based algos).
    params : dict
        Algorithm-specific parameters, forwarded verbatim.
    artifacts : pathlib.Path
        Output directory where the labels file will be placed.
    timeout : float | None
        Optional subprocess timeout in seconds.
    """
    artifacts = pathlib.Path(artifacts)
    artifacts.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = pathlib.Path(tmp)
        data_path = tmp_dir / "X.npy"
        np.save(data_path, X)

        # Name the labels file by a sanitized entry key so parallel runs
        # of different external algos don't clobber each other.
        safe_name = pathlib.Path(entry).name.replace(" ", "_")
        labels_path = artifacts / f"labels_external_{safe_name}.npy"

        payload = {
            "data_path": str(data_path),
            "k": k,
            "params": params,
            "artifacts_dir": str(artifacts),
            "labels_path": str(labels_path),
        }

        cmd = [entry]
        if os.name == "nt" and pathlib.Path(entry).suffix.lower() == ".py":
            cmd = [sys.executable, entry]

        try:
            proc = subprocess.run(
                cmd,
                input=json.dumps(payload),
                capture_output=True,
                text=True,
                timeout=timeout,
                check=True,
            )
        except subprocess.CalledProcessError as e:
            raise RuntimeError(
                f"External runner '{entry}' failed with exit code {e.returncode}.\n"
                f"stderr:\n{e.stderr}"
            ) from e

        extra: Dict[str, Any] = {}
        stdout = proc.stdout.strip()
        if stdout:
            try:
                resp = json.loads(stdout)
                if isinstance(resp, dict):
                    extra = resp.get("extra", {}) or {}
            except json.JSONDecodeError:
                # Non-JSON stdout is allowed; just ignore it.
                pass

        if not labels_path.exists():
            raise RuntimeError(
                f"External runner '{entry}' did not write labels to {labels_path}"
            )

    return {"labels_path": str(labels_path), "extra": extra}
