"""Storage backends for persisting benchmark results.

Currently supported backends:
- :mod:`clustbench.storage.firebase_storage` – Google Firebase / Cloud Storage
"""

from .firebase_storage import FirebaseStorageClient

__all__ = ["FirebaseStorageClient"]
