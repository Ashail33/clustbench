"""Dataset generators for the benchmark harness.

The :class:`DataSpec` carries the core knobs every generator understands —
``n_samples``, ``n_features``, ``centers``, ``compactness``, ``seed`` — plus
optional MDCGen-style parameters (``outliers``, ``noise``, ``density``) that
let us reproduce the experimental design in
`Maharaj (2024) <paper/clustering-review-maharaj-2024.pdf>`_.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.datasets import make_blobs, make_circles, make_classification, make_moons


@dataclass
class DataSpec:
    n_samples: int
    n_features: int
    centers: int
    compactness: float
    seed: int
    outliers: int = 0
    noise: int = 0
    density: float = 1.0
    # How far outliers are placed from the cluster envelope. 1.0 reproduces
    # the original ``_inject_outliers`` behaviour (uniform box at 1.5x data
    # range); higher values move outliers exponentially further out.
    outlier_extremity: float = 1.0


def _inject_outliers(X: np.ndarray, y: np.ndarray, n_outliers: int, rng: np.random.Generator,
                     extremity: float = 1.0):
    """Append uniform-random outliers far from the cluster envelope.

    Outliers are drawn from a uniform box that sits at ``extremity * 1.5x``
    the data range, so they're well outside the cluster cloud. Their
    ground-truth label is ``-1`` (the de-facto noise label used by
    DBSCAN-style algorithms). ``extremity=1.0`` reproduces the original
    behaviour; ``extremity=3.0`` puts outliers 3x further from the data.
    """
    if n_outliers <= 0:
        return X, y
    lo, hi = X.min(axis=0), X.max(axis=0)
    span = (hi - lo) + 1e-9
    pad = 0.5 * float(extremity)
    box_lo = lo - pad * span
    box_hi = hi + pad * span
    extra = rng.uniform(box_lo, box_hi, size=(n_outliers, X.shape[1])).astype(X.dtype)
    extra_y = -np.ones(n_outliers, dtype=y.dtype)
    return np.vstack([X, extra]), np.concatenate([y, extra_y])


def _inject_noise(X: np.ndarray, y: np.ndarray, n_noise: int, rng: np.random.Generator):
    """Append Gaussian noise points centred on the data centroid.

    Distinct from outliers: noise points sit *inside* the cluster cloud and
    are harder to separate. Label ``-1``.
    """
    if n_noise <= 0:
        return X, y
    mu = X.mean(axis=0)
    sigma = X.std(axis=0) + 1e-9
    extra = (mu + rng.normal(size=(n_noise, X.shape[1])) * sigma).astype(X.dtype)
    extra_y = -np.ones(n_noise, dtype=y.dtype)
    return np.vstack([X, extra]), np.concatenate([y, extra_y])


def gen_blobs(spec: DataSpec):
    X, y = make_blobs(
        n_samples=spec.n_samples,
        n_features=spec.n_features,
        centers=spec.centers,
        cluster_std=spec.compactness,
        shuffle=True,
        random_state=spec.seed,
    )
    return X.astype(np.float32), y


def gen_mixed(spec: DataSpec):
    X, y = make_classification(
        n_samples=spec.n_samples,
        n_features=spec.n_features,
        n_informative=max(2, spec.n_features // 2),
        n_redundant=spec.n_features // 4,
        n_clusters_per_class=1,
        n_classes=min(8, spec.centers),
        class_sep=1.0 / max(0.2, spec.compactness),
        random_state=spec.seed,
    )
    return X.astype(np.float32), y


def gen_mdcgen(spec: DataSpec):
    """MDCGen-style synthetic dataset generator (Lopez et al., reproduced).

    Generates ``centers`` Gaussian clusters whose spread is modulated by
    ``density`` (lower density => looser clusters), then optionally injects
    ``noise`` in-cloud points and ``outliers`` out-of-cloud points. Cluster
    sizes are drawn from a Dirichlet so they're not exactly equal — a
    closer match to the kind of imbalance MDCGen produces.

    The ``compactness`` knob is multiplied into the per-cluster covariance
    so existing configs that tune compactness keep behaving the same way.
    """
    rng = np.random.default_rng(spec.seed)
    k = max(1, spec.centers)
    d = spec.n_features

    # Cluster centres in a unit cube scaled by sqrt(d) so per-feature spacing is constant.
    centers = rng.uniform(-1.0, 1.0, size=(k, d)) * np.sqrt(d)
    proportions = rng.dirichlet(np.ones(k) * 4.0)
    sizes = (proportions * spec.n_samples).astype(int)
    sizes[-1] = spec.n_samples - sizes[:-1].sum()

    # density in (0, 1] -> std multiplier; lower density = bigger std.
    density = max(0.01, float(spec.density))
    base_std = max(0.05, float(spec.compactness)) / np.sqrt(density)

    X_parts = []
    y_parts = []
    for j in range(k):
        cov_diag = (base_std * (0.5 + rng.random(d))).astype(np.float32)
        pts = rng.normal(loc=centers[j], scale=cov_diag, size=(sizes[j], d)).astype(np.float32)
        X_parts.append(pts)
        y_parts.append(np.full(sizes[j], j, dtype=np.int64))
    X = np.vstack(X_parts)
    y = np.concatenate(y_parts)

    # Shuffle the cluster points.
    perm = rng.permutation(len(X))
    X, y = X[perm], y[perm]

    X, y = _inject_noise(X, y, spec.noise, rng)
    X, y = _inject_outliers(X, y, spec.outliers, rng, extremity=getattr(spec, "outlier_extremity", 1.0))

    # Final shuffle so cluster, noise and outlier rows are interleaved.
    perm = rng.permutation(len(X))
    return X[perm].astype(np.float32), y[perm]


def gen_moons(spec: DataSpec):
    """Two interleaving half-moons. Non-convex; trips up centroid-based methods."""
    X, y = make_moons(n_samples=spec.n_samples, noise=0.1 * float(spec.compactness), random_state=spec.seed)
    if spec.n_features > 2:
        rng = np.random.default_rng(spec.seed)
        pad = rng.normal(scale=0.05 * float(spec.compactness), size=(X.shape[0], spec.n_features - 2))
        X = np.hstack([X, pad])
    return X.astype(np.float32), y.astype(np.int64)


def gen_circles(spec: DataSpec):
    """Concentric circles. ``compactness`` controls noise; ``centers`` is forced to 2."""
    X, y = make_circles(
        n_samples=spec.n_samples,
        noise=0.05 * float(spec.compactness),
        factor=0.5,
        random_state=spec.seed,
    )
    if spec.n_features > 2:
        rng = np.random.default_rng(spec.seed)
        pad = rng.normal(scale=0.05 * float(spec.compactness), size=(X.shape[0], spec.n_features - 2))
        X = np.hstack([X, pad])
    return X.astype(np.float32), y.astype(np.int64)


def gen_anisotropic(spec: DataSpec):
    """Sheared blobs. Tests robustness to non-spherical cluster shape."""
    X, y = make_blobs(
        n_samples=spec.n_samples,
        n_features=spec.n_features,
        centers=spec.centers,
        cluster_std=spec.compactness,
        random_state=spec.seed,
    )
    rng = np.random.default_rng(spec.seed)
    transform = rng.normal(size=(spec.n_features, spec.n_features))
    transform = transform / np.linalg.norm(transform, axis=0, keepdims=True)
    return (X @ transform).astype(np.float32), y.astype(np.int64)


def gen_text20news(spec: DataSpec):
    """20-newsgroups TF-IDF features, truncated to ``spec.n_features`` dims via SVD.

    Real-world high-dimensional sparse data — the regime where graph-based
    spectral clustering (LMM) tends to degrade and dense embedding methods
    like AMM have a fair shot. ``spec.centers`` selects how many topic
    categories to sample from (uses a deterministic prefix of the 20
    standard category names). ``spec.n_samples`` caps the sample size
    after fetching; the function uses a random subset of the fetched
    corpus to honour it.

    Network round-trip on first call (sklearn caches afterwards).
    """
    from sklearn.datasets import fetch_20newsgroups
    from sklearn.decomposition import TruncatedSVD
    from sklearn.feature_extraction.text import TfidfVectorizer

    all_categories = [
        "alt.atheism", "comp.graphics", "comp.os.ms-windows.misc",
        "comp.sys.ibm.pc.hardware", "comp.sys.mac.hardware",
        "comp.windows.x", "misc.forsale", "rec.autos",
        "rec.motorcycles", "rec.sport.baseball", "rec.sport.hockey",
        "sci.crypt", "sci.electronics", "sci.med", "sci.space",
        "soc.religion.christian", "talk.politics.guns",
        "talk.politics.mideast", "talk.politics.misc", "talk.religion.misc",
    ]
    n_cat = max(2, min(int(spec.centers), len(all_categories)))
    categories = all_categories[:n_cat]

    bunch = fetch_20newsgroups(
        subset="train",
        categories=categories,
        shuffle=True,
        random_state=spec.seed,
        remove=("headers", "footers", "quotes"),
    )
    texts = bunch.data
    y_full = bunch.target.astype(np.int64)

    # Subsample to the requested n_samples if the corpus is larger.
    if len(texts) > spec.n_samples > 0:
        rng = np.random.default_rng(spec.seed)
        idx = rng.choice(len(texts), size=spec.n_samples, replace=False)
        texts = [texts[i] for i in idx]
        y_full = y_full[idx]

    # TF-IDF → SVD truncation. SVD gives a dense, signed embedding which
    # downstream methods (kmeans / FMM / LMM / AMM) can all consume.
    vec = TfidfVectorizer(max_features=20000, stop_words="english",
                          min_df=2, max_df=0.95)
    tfidf = vec.fit_transform(texts)
    n_components = max(2, min(int(spec.n_features), tfidf.shape[1] - 1,
                              tfidf.shape[0] - 1))
    svd = TruncatedSVD(n_components=n_components, random_state=spec.seed)
    X = svd.fit_transform(tfidf).astype(np.float32)
    return X, y_full


def _from_sklearn_loader(loader, spec: DataSpec):
    """Adapter: wrap a sklearn ``load_*`` bunch so it honours ``DataSpec``.

    The synthetic generators all return ``(X, y)`` shaped by ``spec`` —
    real datasets have a fixed shape, so this adapter:

    - Loads the bunch once.
    - Optionally subsamples to ``spec.n_samples`` (with a per-seed RNG).
    - Optionally truncates feature columns via random projection if the
      caller asked for fewer dims than the dataset has. (Keeps the
      benchmark grid's ``n_features`` axis honest.)
    - Injects ``spec.outliers`` and ``spec.noise`` points the same way
      ``gen_mdcgen`` does.

    Always returns ``np.float32`` X and ``np.int64`` y, label ``-1`` for
    injected outliers / noise.
    """
    bunch = loader()
    X_full, y_full = bunch.data, bunch.target

    rng = np.random.default_rng(spec.seed)

    # Subsample to spec.n_samples if smaller than the dataset (else use all).
    if 0 < spec.n_samples < len(X_full):
        idx = rng.choice(len(X_full), size=spec.n_samples, replace=False)
        X_full, y_full = X_full[idx], y_full[idx]

    # Trim or random-project to spec.n_features if requested.
    d_actual = X_full.shape[1]
    if 0 < spec.n_features < d_actual:
        proj = rng.normal(size=(d_actual, spec.n_features)) / np.sqrt(d_actual)
        X_full = X_full.astype(np.float32) @ proj.astype(np.float32)

    X = X_full.astype(np.float32)
    y = y_full.astype(np.int64)

    X, y = _inject_noise(X, y, spec.noise, rng)
    X, y = _inject_outliers(X, y, spec.outliers, rng, extremity=getattr(spec, "outlier_extremity", 1.0))
    perm = rng.permutation(len(X))
    return X[perm].astype(np.float32), y[perm]


def gen_inverse_pca(spec: DataSpec):
    """High-ambient-dim data with a controlled low-intrinsic-dim structure.

    Uses the ``inverse_pca`` library to *prescribe* an orthonormal basis +
    spectrum + noise, then samples cluster blobs in the low-dim latent
    space before projecting them up to ``spec.n_features`` ambient
    dimensions.

    Concretely:

    1. Build an ``InversePCAGenerator`` with ``n_features = spec.n_features``
       ambient, ``n_components = max(2, min(spec.centers, spec.n_features //
       4))`` latent dims. Spectrum is power-decay (alpha 1.5) so the first
       latent component carries the most variance — realistic for things
       like text TF-IDF or expression data.
    2. Sample ``spec.centers`` cluster means uniformly in the latent space
       (scaled by sqrt(n_components) so they're well-separated relative to
       intra-cluster spread).
    3. For each cluster, draw a Dirichlet-imbalanced number of latent
       vectors centred on the cluster mean with intra-cluster std =
       ``compactness * 0.4`` (lower density → looser clusters per the
       MDCGen analogy).
    4. Project everything to ambient via ``generator.transform(Z)`` and
       add the generator's isotropic noise.
    5. Inject ``spec.outliers`` and ``spec.noise`` points the same way
       ``gen_mdcgen`` does.

    Returns a regime that exposes the *gap* between dense centroid-based
    methods (kmeans family, gmm) and basis-learning methods (lmm, amm,
    s5c) — the latter should win because the data is genuinely low-rank.
    """
    from inverse_pca import InversePCAGenerator

    rng = np.random.default_rng(spec.seed)
    d_amb = max(2, int(spec.n_features))
    k = max(1, int(spec.centers))
    # Latent dimensionality: enough to separate k clusters, capped by
    # ambient dim. Floor at 2 so the latent space is non-trivial.
    n_latent = max(2, min(int(k), d_amb // 4))

    gen = InversePCAGenerator(
        n_features=d_amb,
        n_components=n_latent,
        spectrum="power",
        spectrum_decay=1.5,
        latent_dist="gaussian",
        noise_std=0.05 * float(spec.compactness),
        random_state=spec.seed,
    )

    # Cluster means in the latent space.
    centers_lat = rng.uniform(
        -1.0, 1.0, size=(k, n_latent)
    ) * np.sqrt(n_latent)
    # Cluster sizes — Dirichlet-imbalanced like gen_mdcgen.
    proportions = rng.dirichlet(np.ones(k) * 4.0)
    sizes = (proportions * spec.n_samples).astype(int)
    sizes[-1] = spec.n_samples - sizes[:-1].sum()

    intra_std = max(0.05, 0.4 * float(spec.compactness)) / np.sqrt(
        max(0.01, float(spec.density))
    )

    Z_parts = []
    y_parts = []
    for j in range(k):
        Zj = centers_lat[j] + rng.normal(
            scale=intra_std, size=(sizes[j], n_latent)
        )
        Z_parts.append(Zj)
        y_parts.append(np.full(sizes[j], j, dtype=np.int64))
    Z = np.vstack(Z_parts).astype(np.float32)
    y = np.concatenate(y_parts)

    # Project to ambient via the generator's basis, then add its noise.
    X = gen.transform(Z).astype(np.float32)
    if gen.noise_std > 0:
        X = X + rng.normal(
            scale=gen.noise_std, size=X.shape
        ).astype(np.float32)

    perm = rng.permutation(len(X))
    X, y = X[perm], y[perm]

    X, y = _inject_noise(X, y, spec.noise, rng)
    X, y = _inject_outliers(X, y, spec.outliers, rng, extremity=getattr(spec, "outlier_extremity", 1.0))
    perm = rng.permutation(len(X))
    return X[perm].astype(np.float32), y[perm]


def gen_iris(spec: DataSpec):
    """Fisher's iris dataset (n=150, d=4, k=3). Classic clustering benchmark."""
    from sklearn.datasets import load_iris
    return _from_sklearn_loader(load_iris, spec)


def gen_wine(spec: DataSpec):
    """UCI Wine dataset (n=178, d=13, k=3). Higher dim than iris, similar k."""
    from sklearn.datasets import load_wine
    return _from_sklearn_loader(load_wine, spec)


def gen_breast_cancer(spec: DataSpec):
    """UCI Wisconsin breast-cancer (n=569, d=30, k=2). Binary, high-d."""
    from sklearn.datasets import load_breast_cancer
    return _from_sklearn_loader(load_breast_cancer, spec)


def gen_digits(spec: DataSpec):
    """UCI handwritten digits (n=1797, d=64, k=10). Higher k, higher d.
    Use a smaller ``n_features`` to project down; otherwise some O(n*d²)
    algos get slow."""
    from sklearn.datasets import load_digits
    return _from_sklearn_loader(load_digits, spec)


DATASETS = {
    "blobs": gen_blobs,
    "mixed": gen_mixed,
    "mdcgen": gen_mdcgen,
    "moons": gen_moons,
    "circles": gen_circles,
    "anisotropic": gen_anisotropic,
    "text20news": gen_text20news,
    "iris": gen_iris,
    "wine": gen_wine,
    "breast_cancer": gen_breast_cancer,
    "digits": gen_digits,
    "inverse_pca": gen_inverse_pca,
}

# Wire in the shape generators from datasets_shapes.py.
from .datasets_shapes import SHAPE_DATASETS  # noqa: E402
DATASETS.update(SHAPE_DATASETS)

# Wire in the curated real-world datasets from datasets_real.py. These
# include sklearn-bundled (iris/wine/breast_cancer/digits/olivetti_faces)
# and OpenML fetches (glass/vehicle/segment/yeast/ecoli) that fall back
# to a synthetic shape when offline. The metadata dict is exposed for
# the dashboard / analysis.
from .datasets_real import REAL_DATASETS, REAL_METADATA  # noqa: E402
# Only register real-dataset ids that aren't already in DATASETS (so the
# original sklearn loaders here keep their precedence). The OpenML
# additions (glass, vehicle, ...) come along for free.
for _name, _gen in REAL_DATASETS.items():
    DATASETS.setdefault(_name, _gen)
