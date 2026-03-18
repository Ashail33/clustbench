"""Tests for the Firebase Storage client (using mocks – no real Firebase connection)."""

from __future__ import annotations

import json
import pathlib
from unittest.mock import MagicMock, patch, call

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_run_dir(tmp_path: pathlib.Path) -> pathlib.Path:
    """Create a fake run directory with the files clustbench normally produces."""
    run_dir = tmp_path / "my_run"
    run_dir.mkdir()
    (run_dir / "results.csv").write_text("algo,ari\nkmeans,0.99\n")
    (run_dir / "results.parquet").write_bytes(b"fake parquet bytes")
    (run_dir / "summary.json").write_text("{}")
    artifacts = run_dir / "artifacts"
    artifacts.mkdir()
    (artifacts / "labels_kmeans.npy").write_bytes(b"fake npy")
    (artifacts / "metrics_kmeans.json").write_text("{}")
    return run_dir


# ---------------------------------------------------------------------------
# FirebaseStorageClient – happy-path with mocked firebase_admin
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_firebase_admin(monkeypatch):
    """Patch the entire firebase_admin ecosystem so no real SDK is needed."""
    fake_app = MagicMock(name="fake_firebase_app")
    fake_bucket = MagicMock(name="fake_bucket")
    fake_blob = MagicMock(name="fake_blob")

    # blob() returns a blob, upload_from_filename is a no-op
    fake_bucket.blob.return_value = fake_blob
    fake_bucket.list_blobs.return_value = []

    fake_storage_module = MagicMock()
    fake_storage_module.bucket.return_value = fake_bucket

    fake_credentials_module = MagicMock()
    fake_credentials_module.Certificate.return_value = MagicMock()
    fake_credentials_module.ApplicationDefault.return_value = MagicMock()

    fake_firebase_admin_module = MagicMock()
    fake_firebase_admin_module.get_app.side_effect = ValueError("no app")
    fake_firebase_admin_module.initialize_app.return_value = fake_app

    import clustbench.storage.firebase_storage as fs_module
    monkeypatch.setattr(fs_module, "_FIREBASE_AVAILABLE", True)
    monkeypatch.setattr(fs_module, "_INITIALIZED_APPS", {})
    monkeypatch.setattr(fs_module, "firebase_admin", fake_firebase_admin_module)
    monkeypatch.setattr(fs_module, "credentials", fake_credentials_module)
    monkeypatch.setattr(fs_module, "fb_storage", fake_storage_module)

    return {
        "app": fake_app,
        "bucket": fake_bucket,
        "blob": fake_blob,
        "storage": fake_storage_module,
    }


def test_firebase_client_initialises(mock_firebase_admin):
    from clustbench.storage.firebase_storage import FirebaseStorageClient
    client = FirebaseStorageClient(
        bucket_name="test-bucket.appspot.com",
        credentials_path="/fake/creds.json",
    )
    assert client.bucket_name == "test-bucket.appspot.com"
    assert client.prefix == "clustbench-runs"


def test_upload_run_returns_gcs_uris(tmp_path, mock_firebase_admin):
    from clustbench.storage.firebase_storage import FirebaseStorageClient
    run_dir = _make_run_dir(tmp_path)

    client = FirebaseStorageClient(bucket_name="test-bucket.appspot.com")
    uris = client.upload_run(run_dir, run_name="my_run")

    assert len(uris) > 0
    for uri in uris:
        assert uri.startswith("gs://test-bucket.appspot.com/")


def test_upload_run_excludes_npy_by_default(tmp_path, mock_firebase_admin):
    from clustbench.storage.firebase_storage import FirebaseStorageClient
    run_dir = _make_run_dir(tmp_path)

    client = FirebaseStorageClient(bucket_name="test-bucket.appspot.com")
    uris = client.upload_run(run_dir, run_name="my_run", include_artifacts=False)

    assert all(".npy" not in uri for uri in uris)


def test_upload_run_includes_npy_when_requested(tmp_path, mock_firebase_admin):
    from clustbench.storage.firebase_storage import FirebaseStorageClient
    run_dir = _make_run_dir(tmp_path)

    client = FirebaseStorageClient(bucket_name="test-bucket.appspot.com")
    uris = client.upload_run(run_dir, run_name="my_run", include_artifacts=True)

    assert any(".npy" in uri for uri in uris)


def test_upload_run_missing_directory_raises(tmp_path, mock_firebase_admin):
    from clustbench.storage.firebase_storage import FirebaseStorageClient
    client = FirebaseStorageClient(bucket_name="test-bucket.appspot.com")
    with pytest.raises(FileNotFoundError):
        client.upload_run(tmp_path / "nonexistent", run_name="x")


def test_upload_file(tmp_path, mock_firebase_admin):
    from clustbench.storage.firebase_storage import FirebaseStorageClient
    local_file = tmp_path / "summary.json"
    local_file.write_text("{}")

    client = FirebaseStorageClient(bucket_name="test-bucket.appspot.com")
    uri = client.upload_file(local_file, storage_path="run1/summary.json")

    assert uri == "gs://test-bucket.appspot.com/clustbench-runs/run1/summary.json"


def test_list_runs(tmp_path, mock_firebase_admin):
    from clustbench.storage.firebase_storage import FirebaseStorageClient

    # Patch list_blobs to return fake blobs with names
    fake_blob1 = MagicMock()
    fake_blob1.name = "clustbench-runs/run_a/results.csv"
    fake_blob2 = MagicMock()
    fake_blob2.name = "clustbench-runs/run_b/results.csv"
    fake_blob3 = MagicMock()
    fake_blob3.name = "clustbench-runs/run_a/summary.json"
    mock_firebase_admin["bucket"].list_blobs.return_value = [fake_blob1, fake_blob2, fake_blob3]

    client = FirebaseStorageClient(bucket_name="test-bucket.appspot.com")
    runs = client.list_runs()
    assert sorted(runs) == ["run_a", "run_b"]


# ---------------------------------------------------------------------------
# Missing firebase-admin raises ImportError
# ---------------------------------------------------------------------------

def test_import_error_without_firebase_admin(monkeypatch):
    import clustbench.storage.firebase_storage as fs_module
    monkeypatch.setattr(fs_module, "_FIREBASE_AVAILABLE", False)
    monkeypatch.setattr(fs_module, "_INITIALIZED_APPS", {})

    from clustbench.storage.firebase_storage import FirebaseStorageClient
    with pytest.raises(ImportError, match="firebase-admin"):
        FirebaseStorageClient(bucket_name="test-bucket.appspot.com")
