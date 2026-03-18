"""Firebase Cloud Storage backend for clustbench results.

Usage
-----
Install the optional dependency::

    pip install "clustbench[firebase]"

Then either pass ``credentials_path`` explicitly or set the
``GOOGLE_APPLICATION_CREDENTIALS`` environment variable to the path of a
service-account JSON key file.

Example
-------
>>> from clustbench.storage import FirebaseStorageClient
>>> client = FirebaseStorageClient(bucket_name="my-project.appspot.com")
>>> client.upload_run(local_dir="runs/demo", run_name="demo")
Uploaded 5 file(s) to gs://my-project.appspot.com/clustbench-runs/demo/
"""

from __future__ import annotations

import os
import pathlib
from typing import Sequence

try:
    import firebase_admin
    from firebase_admin import credentials, storage as fb_storage

    _FIREBASE_AVAILABLE = True
except ImportError:  # pragma: no cover
    _FIREBASE_AVAILABLE = False


_DEFAULT_PREFIX = "clustbench-runs"
_INITIALIZED_APPS: dict[str, object] = {}  # bucket -> Firebase App


def _init_app(bucket_name: str, credentials_path: str | None) -> object:
    """Initialise (or reuse) a Firebase app for *bucket_name*."""
    if not _FIREBASE_AVAILABLE:
        raise ImportError(
            "firebase-admin is required for Firebase Storage support. "
            "Install it with: pip install 'clustbench[firebase]'"
        )
    if bucket_name in _INITIALIZED_APPS:
        return _INITIALIZED_APPS[bucket_name]

    cred_path = credentials_path or os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    if cred_path:
        cred = credentials.Certificate(cred_path)
    else:
        cred = credentials.ApplicationDefault()

    app_name = f"clustbench-{bucket_name}"
    try:
        app = firebase_admin.get_app(app_name)
    except ValueError:
        app = firebase_admin.initialize_app(
            cred,
            options={"storageBucket": bucket_name},
            name=app_name,
        )
    _INITIALIZED_APPS[bucket_name] = app
    return app


class FirebaseStorageClient:
    """Upload clustbench run artifacts to a Firebase / Cloud Storage bucket.

    Parameters
    ----------
    bucket_name:
        The Firebase Storage bucket name (e.g. ``"my-project.appspot.com"``).
    credentials_path:
        Path to a service-account JSON key file.  If ``None`` the SDK falls
        back to ``GOOGLE_APPLICATION_CREDENTIALS`` or Application Default
        Credentials (ADC).
    prefix:
        Storage path prefix inside the bucket.  Defaults to
        ``"clustbench-runs"``.
    """

    def __init__(
        self,
        bucket_name: str,
        credentials_path: str | None = None,
        prefix: str = _DEFAULT_PREFIX,
    ) -> None:
        self.bucket_name = bucket_name
        self.credentials_path = credentials_path
        self.prefix = prefix
        self._app = _init_app(bucket_name, credentials_path)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def upload_run(
        self,
        local_dir: str | pathlib.Path,
        run_name: str,
        extensions: Sequence[str] = (".parquet", ".csv", ".json"),
        include_artifacts: bool = False,
    ) -> list[str]:
        """Upload all result files from *local_dir* to Firebase Storage.

        Parameters
        ----------
        local_dir:
            Local directory produced by a clustbench run (the ``--out`` path).
        run_name:
            Sub-path inside the storage prefix (e.g. ``"2024-01-01_kmeans"``).
        extensions:
            File extensions to upload.  Label ``.npy`` artifact files are
            excluded by default because they can be large; set
            ``include_artifacts=True`` to include them.
        include_artifacts:
            When ``True``, also upload ``.npy`` label arrays from the
            ``artifacts/`` sub-directory.

        Returns
        -------
        list[str]
            GCS paths (``gs://…``) of the uploaded blobs.
        """
        local_dir = pathlib.Path(local_dir)
        if not local_dir.is_dir():
            raise FileNotFoundError(f"Run directory not found: {local_dir}")

        bucket = fb_storage.bucket(app=self._app)
        uploaded: list[str] = []

        patterns = list(extensions)
        if include_artifacts:
            patterns.append(".npy")

        for file in local_dir.rglob("*"):
            if not file.is_file():
                continue
            if file.suffix not in patterns:
                continue
            if not include_artifacts and file.suffix == ".npy":
                continue

            relative = file.relative_to(local_dir)
            blob_path = f"{self.prefix}/{run_name}/{relative}"
            blob = bucket.blob(blob_path)
            blob.upload_from_filename(str(file))
            gcs_uri = f"gs://{self.bucket_name}/{blob_path}"
            uploaded.append(gcs_uri)

        return uploaded

    def upload_file(self, local_path: str | pathlib.Path, storage_path: str) -> str:
        """Upload a single file to *storage_path* inside the bucket.

        Parameters
        ----------
        local_path:
            Local file to upload.
        storage_path:
            Destination path inside the bucket (relative, without the prefix).

        Returns
        -------
        str
            GCS URI (``gs://…``) of the uploaded blob.
        """
        local_path = pathlib.Path(local_path)
        bucket = fb_storage.bucket(app=self._app)
        blob_path = f"{self.prefix}/{storage_path}"
        blob = bucket.blob(blob_path)
        blob.upload_from_filename(str(local_path))
        return f"gs://{self.bucket_name}/{blob_path}"

    def list_runs(self) -> list[str]:
        """Return the unique run names stored under :attr:`prefix`.

        Returns
        -------
        list[str]
            Sorted list of run names.
        """
        bucket = fb_storage.bucket(app=self._app)
        blobs = bucket.list_blobs(prefix=f"{self.prefix}/")
        runs: set[str] = set()
        for blob in blobs:
            parts = blob.name[len(self.prefix) + 1:].split("/")
            if parts:
                runs.add(parts[0])
        return sorted(runs)
