"""External algorithm runner using a JSON-over-stdin/stdout protocol.

Any external executable can be plugged in as a clustering algorithm.  The
runner calls the executable, writes a JSON payload to its stdin, and reads a
JSON response from its stdout.  The response must include a ``labels_path``
key pointing to a ``.npy`` file containing the predicted integer labels.

Protocol
--------
**stdin** – JSON object with keys::

    {
        "X_path": "<path to .npy float32 array of shape (n, d)>",
        "k": <int or null>,
        "params": {<extra params from config>},
        "artifacts_dir": "<directory where the executable should write outputs>"
    }

**stdout** – JSON object with keys::

    {
        "labels_path": "<path to .npy int array of shape (n,)>",
        "extra": {<optional dict of extra metadata>}
    }

Any non-zero exit code or JSON parse failure is raised as a ``RuntimeError``.
"""

from __future__ import annotations

import json
import pathlib
import subprocess
import tempfile

import numpy as np


def run_external(
    entry: str,
    X: np.ndarray,
    k: int | None,
    params: dict,
    artifacts_dir: pathlib.Path,
) -> dict:
    """Run an external clustering executable and return its response dict.

    Parameters
    ----------
    entry:
        Path to the executable (or any shell-runnable command).
    X:
        Feature matrix, shape ``(n_samples, n_features)``.
    k:
        Target number of clusters, or ``None`` if the algorithm should
        determine *k* automatically.
    params:
        Extra parameters forwarded to the executable via the JSON payload.
    artifacts_dir:
        Directory where the executable should write output files.

    Returns
    -------
    dict
        Parsed JSON response from the executable.  Must contain at least
        ``labels_path``.
    """
    artifacts_dir = pathlib.Path(artifacts_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.NamedTemporaryFile(suffix=".npy", dir=artifacts_dir, delete=False) as f:
        x_path = f.name
    np.save(x_path, X.astype(np.float32))

    payload = json.dumps({
        "X_path": x_path,
        "k": k,
        "params": params,
        "artifacts_dir": str(artifacts_dir),
    })

    try:
        result = subprocess.run(
            entry,
            input=payload,
            capture_output=True,
            text=True,
            shell=True,
            timeout=3600,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"External algorithm '{entry}' timed out after 3600 s") from exc

    if result.returncode != 0:
        raise RuntimeError(
            f"External algorithm '{entry}' exited with code {result.returncode}.\n"
            f"stderr:\n{result.stderr}"
        )

    try:
        response = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"External algorithm '{entry}' returned invalid JSON.\n"
            f"stdout:\n{result.stdout}"
        ) from exc

    if "labels_path" not in response:
        raise RuntimeError(
            f"External algorithm '{entry}' response missing required key 'labels_path'.\n"
            f"Got: {response}"
        )

    return response
