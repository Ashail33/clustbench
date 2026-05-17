"""Real-world dataset adapters for the benchmark harness.

A companion to :mod:`clustbench.datasets`. Where ``datasets.py`` favours
*synthetic* generators (blobs, MDCGen, inverse-PCA, moons, circles), this
module curates a set of *real* clustering benchmarks of varied size,
dimensionality, and domain — biology, image, physics, social, chemistry.

Loaders fall into three buckets:

- **sklearn-bundled.** Tiny, always-on. ``load_iris`` etc.
- **OpenML fetch.** Network round-trip on first call, cached in a
  process-local dict so re-running in the same Python session is free.
  When OpenML is unreachable (no network, sandbox) we fall back to a
  shape-matched ``SyntheticFallback`` so the registry still works in CI
  — a warning is emitted via :func:`warnings.warn`.
- **Face images.** ``fetch_olivetti_faces`` for a 4096-dim image cluster.

Public API
----------

- :class:`DatasetMetadata` — per-dataset record (name, source, shape,
  domain, license, notes).
- :data:`REAL_DATASETS` — ``{id: gen_fn(spec)}`` for use with the same
  ``DataSpec`` machinery as ``datasets.py``.
- :data:`REAL_METADATA` — ``{id: DatasetMetadata}`` parallel to
  ``REAL_DATASETS``.

Each ``gen_xxx(spec)`` returns ``(X: float32, y: int64)`` and is safe to
call repeatedly: OpenML results are cached and SyntheticFallback uses
``spec.seed`` so it's reproducible.
"""

from __future__ import annotations

import warnings
from dataclasses import asdict, dataclass

import numpy as np

from clustbench.datasets import DataSpec, _inject_noise, _inject_outliers


# ---------------------------------------------------------------------------
# DatasetMetadata
# ---------------------------------------------------------------------------

@dataclass
class DatasetMetadata:
    """Per-dataset record kept alongside each generator.

    Captures the *nominal* shape — i.e. the dataset's natural
    (n_samples, n_features, n_classes) before any DataSpec subsampling /
    random projection / outlier injection. Use the metadata to drive
    bench-grid choices (e.g. pick the dataset's native ``k`` rather than
    a guessed one).
    """

    name: str
    source: str        # "sklearn" | "openml" | "synthetic" | "local"
    n_samples: int
    n_features: int
    n_classes: int
    domain: str        # "biology"|"physics"|"image"|"text"|"social"|"chemistry"|"other"
    license: str = "varies"
    notes: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# OpenML cache + fetcher
# ---------------------------------------------------------------------------

# Process-local cache: lookups in the same Python process are free.
_OPENML_CACHE: dict = {}


def _openml_fetcher(data_id_or_name, *, prefer_id: int | None = None,
                    name: str | None = None):
    """Fetch (and cache) an OpenML dataset, falling back to ``None`` offline.

    Parameters
    ----------
    data_id_or_name : int or str
        Used as the cache key.
    prefer_id : int, optional
        If given, try ``fetch_openml(data_id=...)`` first. Recommended —
        IDs are stable, names sometimes ambiguous.
    name : str, optional
        Fallback name passed to ``fetch_openml(name=...)`` if the
        ``data_id`` lookup fails or no ID was supplied.

    Returns
    -------
    bunch or None
        sklearn-style bunch with ``.data`` and ``.target`` on success, or
        ``None`` if every lookup failed (caller should synthesize a fallback).

    Notes
    -----
    Emits ``warnings.warn`` on failure so CI logs show the fallback.
    """
    cache_key = data_id_or_name
    if cache_key in _OPENML_CACHE:
        return _OPENML_CACHE[cache_key]

    try:
        from sklearn.datasets import fetch_openml
    except Exception as e:  # pragma: no cover — sklearn always present
        warnings.warn(f"sklearn.fetch_openml import failed: {e}; "
                      f"falling back to synthetic for {cache_key}.")
        return None

    last_err = None
    if prefer_id is not None:
        try:
            bunch = fetch_openml(data_id=prefer_id, as_frame=False,
                                 parser="liac-arff")
            _OPENML_CACHE[cache_key] = bunch
            return bunch
        except Exception as e:
            last_err = e
    if name is not None:
        try:
            bunch = fetch_openml(name=name, as_frame=False,
                                 parser="liac-arff", version=1)
            _OPENML_CACHE[cache_key] = bunch
            return bunch
        except Exception as e:
            last_err = e

    warnings.warn(
        f"OpenML fetch failed for {cache_key} ({last_err!r}); "
        f"falling back to synthetic."
    )
    return None


# ---------------------------------------------------------------------------
# Synthetic fallback shaped to match a target real dataset
# ---------------------------------------------------------------------------

def _synthetic_fallback(spec: DataSpec, *, n_nominal: int, d_nominal: int,
                        k_nominal: int):
    """Produce a Gaussian-blob fallback when a real fetch is unreachable.

    Honours ``spec.n_samples`` / ``spec.n_features`` if smaller than the
    dataset's nominal shape (mirrors ``_from_sklearn_loader`` behaviour),
    otherwise uses the nominal shape so downstream code sees roughly
    what it would have seen with the real data.
    """
    from sklearn.datasets import make_blobs

    n = spec.n_samples if 0 < spec.n_samples < n_nominal else n_nominal
    d = spec.n_features if 0 < spec.n_features < d_nominal else d_nominal
    k = max(2, k_nominal)
    X, y = make_blobs(
        n_samples=n, n_features=d, centers=k, cluster_std=1.0,
        random_state=spec.seed,
    )
    rng = np.random.default_rng(spec.seed)
    X = X.astype(np.float32)
    y = y.astype(np.int64)
    X, y = _inject_noise(X, y, spec.noise, rng)
    X, y = _inject_outliers(X, y, spec.outliers, rng)
    perm = rng.permutation(len(X))
    return X[perm].astype(np.float32), y[perm]


# ---------------------------------------------------------------------------
# Generic sklearn-bunch adapter (mirrors datasets._from_sklearn_loader)
# ---------------------------------------------------------------------------

def _from_sklearn_bunch(bunch, spec: DataSpec):
    """Subsample / random-project / inject noise from an sklearn bunch.

    The same shape-honouring pipeline as ``datasets._from_sklearn_loader``,
    but works directly on a pre-fetched bunch (so OpenML caching works).
    """
    X_full = np.asarray(bunch.data)
    y_raw = np.asarray(bunch.target)

    # OpenML targets are sometimes strings ("class_1"); encode to ints.
    if y_raw.dtype.kind in ("U", "O", "S"):
        _, y_full = np.unique(y_raw, return_inverse=True)
        y_full = y_full.astype(np.int64)
    else:
        y_full = y_raw.astype(np.int64)

    rng = np.random.default_rng(spec.seed)

    if 0 < spec.n_samples < len(X_full):
        idx = rng.choice(len(X_full), size=spec.n_samples, replace=False)
        X_full, y_full = X_full[idx], y_full[idx]

    d_actual = X_full.shape[1]
    if 0 < spec.n_features < d_actual:
        proj = rng.normal(size=(d_actual, spec.n_features)) / np.sqrt(d_actual)
        X_full = X_full.astype(np.float32) @ proj.astype(np.float32)

    X = X_full.astype(np.float32)
    y = y_full.astype(np.int64)

    X, y = _inject_noise(X, y, spec.noise, rng)
    X, y = _inject_outliers(X, y, spec.outliers, rng)
    perm = rng.permutation(len(X))
    return X[perm].astype(np.float32), y[perm]


# ---------------------------------------------------------------------------
# sklearn-bundled datasets (metadata only — generators exist in datasets.py)
# ---------------------------------------------------------------------------

def gen_iris(spec: DataSpec):
    """Replicate ``datasets.gen_iris`` so the real-dataset registry is self-contained."""
    from sklearn.datasets import load_iris
    return _from_sklearn_bunch(load_iris(), spec)


def gen_wine(spec: DataSpec):
    from sklearn.datasets import load_wine
    return _from_sklearn_bunch(load_wine(), spec)


def gen_breast_cancer(spec: DataSpec):
    from sklearn.datasets import load_breast_cancer
    return _from_sklearn_bunch(load_breast_cancer(), spec)


def gen_digits(spec: DataSpec):
    from sklearn.datasets import load_digits
    return _from_sklearn_bunch(load_digits(), spec)


def gen_olivetti_faces(spec: DataSpec):
    """Olivetti face images: 400 samples, 4096 dims (64x64), k=40.

    High-d image data — a torture test for distance-based clustering.
    ``_from_sklearn_bunch`` will random-project to ``spec.n_features`` if
    that's smaller than 4096 (recommended for centroid-based methods).
    """
    try:
        from sklearn.datasets import fetch_olivetti_faces
        bunch = fetch_olivetti_faces(shuffle=False, random_state=spec.seed)
    except Exception as e:
        warnings.warn(f"fetch_olivetti_faces failed: {e}; falling back to synthetic.")
        return _synthetic_fallback(spec, n_nominal=400, d_nominal=4096, k_nominal=40)
    # sklearn's olivetti uses ``.images``/``.data``/``.target``; ``.target`` is the
    # person ID (0..39).
    return _from_sklearn_bunch(bunch, spec)


# ---------------------------------------------------------------------------
# OpenML-fetched datasets
# ---------------------------------------------------------------------------

def gen_glass(spec: DataSpec):
    """UCI Glass identification: 214 samples, 9 features, 7 classes (chemistry)."""
    bunch = _openml_fetcher("glass", prefer_id=41, name="glass")
    if bunch is None:
        return _synthetic_fallback(spec, n_nominal=214, d_nominal=9, k_nominal=7)
    return _from_sklearn_bunch(bunch, spec)


def gen_vehicle(spec: DataSpec):
    """Statlog vehicle silhouettes: 846 samples, 18 features, 4 classes (image)."""
    bunch = _openml_fetcher("vehicle", prefer_id=54, name="vehicle")
    if bunch is None:
        return _synthetic_fallback(spec, n_nominal=846, d_nominal=18, k_nominal=4)
    return _from_sklearn_bunch(bunch, spec)


def gen_segment(spec: DataSpec):
    """UCI Image Segmentation: 2310 samples, 19 features, 7 classes."""
    bunch = _openml_fetcher("segment", prefer_id=40984, name="segment")
    if bunch is None:
        return _synthetic_fallback(spec, n_nominal=2310, d_nominal=19, k_nominal=7)
    return _from_sklearn_bunch(bunch, spec)


def gen_yeast(spec: DataSpec):
    """UCI Yeast protein localization: 1484 samples, 8 features, 10 classes."""
    bunch = _openml_fetcher("yeast", prefer_id=181, name="yeast")
    if bunch is None:
        return _synthetic_fallback(spec, n_nominal=1484, d_nominal=8, k_nominal=10)
    return _from_sklearn_bunch(bunch, spec)


def gen_ecoli(spec: DataSpec):
    """UCI E.coli protein localization: 336 samples, 7 features, 8 classes."""
    bunch = _openml_fetcher("ecoli", prefer_id=40671, name="ecoli")
    if bunch is None:
        return _synthetic_fallback(spec, n_nominal=336, d_nominal=7, k_nominal=8)
    return _from_sklearn_bunch(bunch, spec)


# ---------------------------------------------------------------------------
# Registries
# ---------------------------------------------------------------------------

REAL_DATASETS = {
    # sklearn-bundled
    "iris": gen_iris,
    "wine": gen_wine,
    "breast_cancer": gen_breast_cancer,
    "digits": gen_digits,
    "olivetti_faces": gen_olivetti_faces,
    # OpenML
    "glass": gen_glass,
    "vehicle": gen_vehicle,
    "segment": gen_segment,
    "yeast": gen_yeast,
    "ecoli": gen_ecoli,
}


REAL_METADATA = {
    "iris": DatasetMetadata(
        name="iris", source="sklearn",
        n_samples=150, n_features=4, n_classes=3,
        domain="biology", license="public-domain",
        notes="Balanced; classic clustering benchmark.",
    ),
    "wine": DatasetMetadata(
        name="wine", source="sklearn",
        n_samples=178, n_features=13, n_classes=3,
        domain="chemistry", license="CC-BY-4.0",
        notes="Moderate-d; scale-sensitive.",
    ),
    "breast_cancer": DatasetMetadata(
        name="breast_cancer", source="sklearn",
        n_samples=569, n_features=30, n_classes=2,
        domain="biology", license="CC-BY-4.0",
        notes="Binary; high-d for n.",
    ),
    "digits": DatasetMetadata(
        name="digits", source="sklearn",
        n_samples=1797, n_features=64, n_classes=10,
        domain="image", license="CC-BY-4.0",
        notes="8x8 grayscale; dense.",
    ),
    "olivetti_faces": DatasetMetadata(
        name="olivetti_faces", source="sklearn",
        n_samples=400, n_features=4096, n_classes=40,
        domain="image", license="research-use",
        notes="64x64 faces; very high-d, low n per class (10).",
    ),
    "glass": DatasetMetadata(
        name="glass", source="openml",
        n_samples=214, n_features=9, n_classes=7,
        domain="chemistry", license="CC-BY-4.0",
        notes="Highly imbalanced; small n.",
    ),
    "vehicle": DatasetMetadata(
        name="vehicle", source="openml",
        n_samples=846, n_features=18, n_classes=4,
        domain="image", license="CC-BY-4.0",
        notes="Silhouette features; moderately balanced.",
    ),
    "segment": DatasetMetadata(
        name="segment", source="openml",
        n_samples=2310, n_features=19, n_classes=7,
        domain="image", license="CC-BY-4.0",
        notes="Balanced (330/class); dense.",
    ),
    "yeast": DatasetMetadata(
        name="yeast", source="openml",
        n_samples=1484, n_features=8, n_classes=10,
        domain="biology", license="CC-BY-4.0",
        notes="Highly imbalanced; low-d, hard.",
    ),
    "ecoli": DatasetMetadata(
        name="ecoli", source="openml",
        n_samples=336, n_features=7, n_classes=8,
        domain="biology", license="CC-BY-4.0",
        notes="Highly imbalanced; tiny classes (2-5).",
    ),
}


__all__ = [
    "DatasetMetadata",
    "REAL_DATASETS",
    "REAL_METADATA",
    "gen_iris", "gen_wine", "gen_breast_cancer", "gen_digits",
    "gen_olivetti_faces",
    "gen_glass", "gen_vehicle", "gen_segment", "gen_yeast", "gen_ecoli",
]
