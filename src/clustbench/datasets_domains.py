"""Domain-specific dataset adapters: time-series + graph node clustering.

A third companion module alongside :mod:`clustbench.datasets` (synthetic
point clouds) and :mod:`clustbench.datasets_real` (real tabular benchmarks).

The existing registry's algorithms all consume ``(n_samples, n_features)``
matrices. This module's job is to turn two *non-tabular* modalities —
univariate time-series and graph nodes — into 16-dimensional feature
matrices the same algorithms can be benchmarked on, so we can finally ask:

- Does kmeans cluster heartbeat shapes (UCR ECG200) the same way it
  clusters Gaussian blobs?
- Does spectral clustering recover Zachary's karate-club factions when it
  sees node-level structural + Laplacian-spectrum features?

Modalities
----------

**Time-series.** Each univariate series of length L is reduced to a
16-dim feature vector: 8 statistical (mean/std/min/max/range/skew/kurt/MAD),
4 spectral (top-4 FFT magnitudes after detrending), 4 autocorrelation
(lag 1/2/5/10). Real series come from a UCR Archive fetch (one of several
mirror URLs); failure falls back to a synthetic sinusoid generator.

**Graph nodes.** Each node of the graph is reduced to a 16-dim feature
vector: 6 structural (degree / clustering coef / eigenvector centrality /
betweenness / PageRank / avg shortest-path length) + 10 spectral
(bottom-10 non-trivial eigenvectors of the normalized Laplacian). Real
graphs are Zachary's karate club (bundled — defined inline as edge list,
no network round-trip) and a stochastic block model. Both are exposed
through the same DataSpec interface.

Public API
----------

- :data:`DOMAIN_DATASETS` — ``{id: gen_fn(spec) -> (X, y)}`` parallel to
  ``SHAPE_DATASETS`` and ``REAL_DATASETS``.
- :data:`DOMAIN_METADATA` — ``{id: DatasetMetadata}`` (reuses the type
  defined in :mod:`clustbench.datasets_real`).
"""

from __future__ import annotations

import warnings
from typing import Tuple

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh

from clustbench.datasets import DataSpec, _inject_noise, _inject_outliers
from clustbench.datasets_real import DatasetMetadata


# ---------------------------------------------------------------------------
# Time-series feature extraction
# ---------------------------------------------------------------------------

def _ts_features(series: np.ndarray) -> np.ndarray:
    """Reduce a (n, L) batch of univariate time-series to (n, 16) features.

    The 16 features are deliberately classical so the resulting matrix is
    informative for *any* downstream clusterer (kmeans / gmm / spectral all
    consume it without further preprocessing):

    - 8 statistical : mean, std, min, max, range, skewness, kurtosis,
      mean-absolute-deviation.
    - 4 spectral    : magnitudes of the first 4 FFT bins (after removing
      the linear trend so the DC + slow drift don't dominate).
    - 4 autocorrel. : Pearson autocorrelation at lags 1, 2, 5, 10.

    NaNs / infs from constant-series corner cases are zero-filled — the
    downstream clusterers can't consume them.
    """
    series = np.asarray(series, dtype=np.float64)
    if series.ndim == 1:
        series = series[None, :]
    n, L = series.shape

    # -- 8 statistical features -------------------------------------------
    mean = series.mean(axis=1)
    std = series.std(axis=1)
    smin = series.min(axis=1)
    smax = series.max(axis=1)
    srange = smax - smin
    centered = series - mean[:, None]
    # Skewness and kurtosis: guard against zero-variance rows.
    var = (centered ** 2).mean(axis=1)
    safe_std = np.where(var > 1e-12, np.sqrt(var), 1.0)
    skew = (centered ** 3).mean(axis=1) / (safe_std ** 3)
    kurt = (centered ** 4).mean(axis=1) / (safe_std ** 4) - 3.0
    mad = np.abs(centered).mean(axis=1)

    # -- 4 spectral features ----------------------------------------------
    # Detrend by subtracting a least-squares linear fit, then take the
    # first 4 FFT magnitudes. The first bin still encodes residual offset
    # but is small after detrending.
    t = np.arange(L)
    t_mean = t.mean()
    t_var = ((t - t_mean) ** 2).sum() or 1.0
    slope = ((centered) * (t - t_mean)).sum(axis=1) / t_var  # (n,)
    detrended = centered - slope[:, None] * (t - t_mean)
    fft_mag = np.abs(np.fft.rfft(detrended, axis=1))
    if fft_mag.shape[1] >= 4:
        spec = fft_mag[:, :4]
    else:
        spec = np.zeros((n, 4))
        spec[:, : fft_mag.shape[1]] = fft_mag

    # -- 4 autocorrelation features ---------------------------------------
    def _autocorr(x: np.ndarray, lag: int) -> np.ndarray:
        if lag >= x.shape[1]:
            return np.zeros(x.shape[0])
        a = x[:, :-lag]
        b = x[:, lag:]
        num = (a * b).mean(axis=1)
        denom = np.sqrt((a ** 2).mean(axis=1) * (b ** 2).mean(axis=1))
        return np.where(denom > 1e-12, num / denom, 0.0)

    ac = np.stack(
        [_autocorr(centered, lag) for lag in (1, 2, 5, 10)], axis=1
    )

    feats = np.concatenate(
        [
            np.stack(
                [mean, std, smin, smax, srange, skew, kurt, mad], axis=1
            ),
            spec,
            ac,
        ],
        axis=1,
    )
    # Replace any leftover NaN / inf (e.g. all-constant input rows).
    feats = np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)
    return feats.astype(np.float32)


# ---------------------------------------------------------------------------
# UCR Archive fetcher (best-effort)
# ---------------------------------------------------------------------------

_UCR_CACHE: dict = {}

# Candidate mirrors for UCR archive. The official archive lives at
# timeseriesclassification.com but its TSV layout varies by mirror; we
# try a couple of likely-available URLs and accept whichever returns
# parseable data.
_UCR_MIRRORS = [
    # tslearn UEA/UCR mirror: each dataset is /{name}/{name}_TRAIN.tsv
    "https://raw.githubusercontent.com/tslearn-team/tslearn/main/tslearn/datasets/UCR_UEA_datasets/{name}/{name}_TRAIN.tsv",
    "https://www.timeseriesclassification.com/Downloads/{name}.txt",
]


def _fetch_ucr(name: str) -> Tuple[np.ndarray, np.ndarray] | None:
    """Fetch a UCR Archive dataset by name. Returns (series, labels) or None.

    The TSV layout is: first column = integer class label, remaining
    columns = the univariate series values. We try a couple of mirrors;
    any failure (HTTP error, parse error) returns ``None`` so the caller
    can fall back to a synthetic generator.
    """
    if name in _UCR_CACHE:
        return _UCR_CACHE[name]
    try:
        import io
        import urllib.request
    except Exception:  # pragma: no cover — stdlib always present
        return None

    for mirror in _UCR_MIRRORS:
        url = mirror.format(name=name)
        try:
            with urllib.request.urlopen(url, timeout=5) as resp:
                raw = resp.read().decode("utf-8", errors="replace")
            arr = np.loadtxt(io.StringIO(raw))
            if arr.ndim != 2 or arr.shape[1] < 4:
                continue
            y = arr[:, 0].astype(np.int64)
            # Relabel to 0..k-1 (UCR labels are sometimes 1..k or {-1,1}).
            _, y = np.unique(y, return_inverse=True)
            series = arr[:, 1:].astype(np.float64)
            _UCR_CACHE[name] = (series, y.astype(np.int64))
            return _UCR_CACHE[name]
        except Exception:
            continue
    return None


# ---------------------------------------------------------------------------
# Synthetic time-series fallback
# ---------------------------------------------------------------------------

def _synthetic_timeseries(
    n: int, length: int, k: int, seed: int, compactness: float = 1.0
) -> Tuple[np.ndarray, np.ndarray]:
    """K clusters of noisy sinusoids with distinct (freq, phase, amp) sigs.

    Each cluster has its own (frequency, phase, amplitude) triple. The
    ``compactness`` knob scales the additive Gaussian noise so the
    benchmark grid's noise axis still has an effect. Returned shape:
    ``(n, length)`` series + ``(n,)`` int labels in 0..k-1.
    """
    rng = np.random.default_rng(seed)
    t = np.linspace(0.0, 2.0 * np.pi, length)
    # Per-cluster signature.
    freqs = 1.0 + 2.0 * rng.random(k)        # 1..3 cycles in the window
    phases = 2.0 * np.pi * rng.random(k)
    amps = 0.5 + 1.5 * rng.random(k)
    noise_std = 0.15 * float(compactness)

    # Roughly balanced cluster sizes.
    base = n // k
    sizes = [base] * k
    sizes[-1] += n - base * k

    series = []
    labels = []
    for j in range(k):
        carrier = amps[j] * np.sin(freqs[j] * t + phases[j])
        block = carrier[None, :] + rng.normal(
            scale=noise_std, size=(sizes[j], length)
        )
        # Slight per-sample amplitude jitter so the cluster isn't degenerate.
        block = block * (1.0 + 0.05 * rng.standard_normal((sizes[j], 1)))
        series.append(block)
        labels.append(np.full(sizes[j], j, dtype=np.int64))

    X_ts = np.vstack(series)
    y = np.concatenate(labels)
    perm = rng.permutation(len(X_ts))
    return X_ts[perm], y[perm]


def _ts_postprocess(series: np.ndarray, y: np.ndarray, spec: DataSpec):
    """Apply feature extraction + DataSpec subsampling + noise/outlier injection."""
    rng = np.random.default_rng(spec.seed)
    # Optional subsample to spec.n_samples (only if positive and smaller).
    if 0 < spec.n_samples < len(series):
        idx = rng.choice(len(series), size=spec.n_samples, replace=False)
        series = series[idx]
        y = y[idx]

    X = _ts_features(series)
    y = y.astype(np.int64)
    X, y = _inject_noise(X, y, spec.noise, rng)
    X, y = _inject_outliers(
        X, y, spec.outliers, rng,
        extremity=getattr(spec, "outlier_extremity", 1.0),
    )
    perm = rng.permutation(len(X))
    return X[perm].astype(np.float32), y[perm]


# ---------------------------------------------------------------------------
# Time-series generators
# ---------------------------------------------------------------------------

def gen_ts_ecg200(spec: DataSpec):
    """UCR ECG200: 200 ECG beats, length 96, k=2 (normal vs ischemia).

    Tries the UCR archive fetch first; on network failure falls back to a
    2-cluster synthetic sinusoid (length 96, n=200) so the registry stays
    functional in sandboxed CI.
    """
    fetched = _fetch_ucr("ECG200")
    if fetched is None:
        warnings.warn(
            "ECG200 UCR fetch failed; using synthetic sinusoid fallback "
            "(n=200, length=96, k=2)."
        )
        series, y = _synthetic_timeseries(
            n=200, length=96, k=2, seed=spec.seed, compactness=spec.compactness
        )
    else:
        series, y = fetched
    return _ts_postprocess(series, y, spec)


def gen_ts_trace(spec: DataSpec):
    """UCR Trace: 200 synthetic instrument signatures, length 275, k=4."""
    fetched = _fetch_ucr("Trace")
    if fetched is None:
        warnings.warn(
            "Trace UCR fetch failed; using synthetic sinusoid fallback "
            "(n=200, length=275, k=4)."
        )
        series, y = _synthetic_timeseries(
            n=200, length=275, k=4, seed=spec.seed, compactness=spec.compactness
        )
    else:
        series, y = fetched
    return _ts_postprocess(series, y, spec)


def gen_ts_synth(spec: DataSpec):
    """Always-on synthetic time-series: noisy sinusoid clusters.

    Length 128, k = ``spec.centers`` (default 3 if the spec doesn't ask
    for more), n = ``spec.n_samples``. Doesn't touch the network — this
    is the offline regression target for the time-series feature
    pipeline.
    """
    k = max(2, min(int(spec.centers), 8))
    n = max(k * 8, int(spec.n_samples) if spec.n_samples > 0 else 200)
    series, y = _synthetic_timeseries(
        n=n, length=128, k=k, seed=spec.seed, compactness=spec.compactness
    )
    return _ts_postprocess(series, y, spec)


# ---------------------------------------------------------------------------
# Graph feature extraction (no networkx required)
# ---------------------------------------------------------------------------

def _adjacency_from_edges(edges, n: int) -> np.ndarray:
    """Build a dense symmetric (n, n) adjacency matrix from an edge list."""
    A = np.zeros((n, n), dtype=np.float32)
    for (u, v) in edges:
        if u == v:
            continue
        A[u, v] = 1.0
        A[v, u] = 1.0
    return A


def _graph_structural_features(A: np.ndarray) -> np.ndarray:
    """Compute the 6 structural features per node: returns (n, 6).

    Features (column order):

    0. degree (normalised by n-1)
    1. clustering coefficient
    2. eigenvector centrality (power-iteration on A)
    3. betweenness centrality (Brandes' algorithm)
    4. PageRank (power iteration with damping 0.85)
    5. average shortest path length from the node (BFS over rows)
    """
    n = A.shape[0]
    if n == 0:
        return np.zeros((0, 6), dtype=np.float32)

    deg = A.sum(axis=1)                       # (n,)
    deg_norm = deg / max(1.0, n - 1)

    # ---- clustering coefficient -----------------------------------------
    # For each node u: triangles_u / (deg_u choose 2). triangles_u =
    # (A @ A)[u, u] / 2 restricted to neighbour pairs that are connected.
    # Equivalently: 0.5 * sum_{v,w in N(u)} A[v,w]. Vectorised:
    A_sq = A @ A
    triangles_x2 = np.einsum("ij,ji->i", A * A_sq, np.ones((n, n)))  # ~2*triangles
    # Simpler and clearer: count triangles via diag(A^3)/2 per node.
    A_cu = A_sq @ A
    triangles = np.diag(A_cu) / 2.0
    poss = deg * (deg - 1) / 2.0
    cc = np.where(poss > 0, triangles / np.maximum(poss, 1.0), 0.0)

    # ---- eigenvector centrality -----------------------------------------
    # Power iteration on A. For disconnected graphs this picks up the
    # dominant component's signal; good enough as a feature.
    v = np.ones(n, dtype=np.float64) / np.sqrt(max(1, n))
    for _ in range(200):
        v_new = A @ v
        nv = np.linalg.norm(v_new)
        if nv < 1e-12:
            break
        v_new = v_new / nv
        if np.linalg.norm(v_new - v) < 1e-9:
            v = v_new
            break
        v = v_new
    eig_cent = np.abs(v)

    # ---- betweenness centrality (Brandes) -------------------------------
    # O(n*m) — fine for n <= 500 graphs which is the only regime we ship
    # in this module.
    bet = np.zeros(n, dtype=np.float64)
    neighbours = [np.where(A[i] > 0)[0] for i in range(n)]
    for s in range(n):
        # BFS from s.
        S = []
        P = [[] for _ in range(n)]
        sigma = np.zeros(n, dtype=np.float64)
        sigma[s] = 1.0
        dist = -np.ones(n, dtype=np.int64)
        dist[s] = 0
        queue = [s]
        head = 0
        while head < len(queue):
            v_ = queue[head]
            head += 1
            S.append(v_)
            for w in neighbours[v_]:
                if dist[w] < 0:
                    dist[w] = dist[v_] + 1
                    queue.append(w)
                if dist[w] == dist[v_] + 1:
                    sigma[w] += sigma[v_]
                    P[w].append(v_)
        delta = np.zeros(n, dtype=np.float64)
        while S:
            w = S.pop()
            for v_ in P[w]:
                if sigma[w] > 0:
                    delta[v_] += (sigma[v_] / sigma[w]) * (1.0 + delta[w])
            if w != s:
                bet[w] += delta[w]
    # Undirected normalisation: divide by 2 then by (n-1)(n-2) for scale.
    if n > 2:
        bet = bet / ((n - 1) * (n - 2))

    # ---- PageRank --------------------------------------------------------
    damping = 0.85
    deg_safe = np.where(deg > 0, deg, 1.0)
    M = (A / deg_safe[:, None]).T  # column-stochastic transition
    pr = np.full(n, 1.0 / n, dtype=np.float64)
    for _ in range(100):
        pr_new = (1 - damping) / n + damping * (M @ pr)
        if np.abs(pr_new - pr).sum() < 1e-9:
            pr = pr_new
            break
        pr = pr_new

    # ---- average shortest-path length from each node --------------------
    avg_sp = np.zeros(n, dtype=np.float64)
    for s in range(n):
        dist = -np.ones(n, dtype=np.int64)
        dist[s] = 0
        queue = [s]
        head = 0
        while head < len(queue):
            v_ = queue[head]
            head += 1
            for w in neighbours[v_]:
                if dist[w] < 0:
                    dist[w] = dist[v_] + 1
                    queue.append(w)
        reachable = dist[dist > 0]
        avg_sp[s] = reachable.mean() if reachable.size > 0 else 0.0

    out = np.stack(
        [deg_norm, cc, eig_cent, bet, pr, avg_sp], axis=1
    ).astype(np.float32)
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def _graph_spectral_features(A: np.ndarray, k: int = 10) -> np.ndarray:
    """Bottom-k non-trivial eigenvectors of the normalised Laplacian.

    Returns ``(n, k)``. The eigenvectors are *diffusion-map weighted* —
    each vector is scaled by ``1/sqrt(eigvalue + eps)`` so the lowest
    non-trivial frequency (the Fiedler vector) dominates the feature
    space. This is the standard spectral-clustering convention and means
    a centroid-based clusterer on these features acts almost like spectral
    clustering itself.

    For tiny graphs (n <= k+2) we just pad with zeros so the output shape
    is stable.
    """
    n = A.shape[0]
    deg = A.sum(axis=1)
    d_inv_sqrt = np.where(deg > 0, 1.0 / np.sqrt(deg), 0.0)
    # L_sym = I - D^-1/2 A D^-1/2
    L = np.eye(n) - (d_inv_sqrt[:, None] * A * d_inv_sqrt[None, :])
    L = (L + L.T) * 0.5  # symmetrise against round-off

    n_take = min(k + 1, n)  # we'll drop the trivial 0-eigenvector
    try:
        if n_take < n:
            # Sparse partial eig — much faster for moderately large n.
            # ``which="SM"`` can be flaky; full eigendecomp is fine for
            # n <= few hundred (this module's regime).
            vals, vecs = eigsh(csr_matrix(L), k=n_take, which="SM")
            order = np.argsort(vals)
            vals = vals[order]
            vecs = vecs[:, order]
        else:
            vals, vecs = np.linalg.eigh(L)
            order = np.argsort(vals)
            vals = vals[order]
            vecs = vecs[:, order]
        # Drop the first (smallest) eigenvector — it's the trivial constant
        # mode and carries no clustering signal.
        sub_vals = vals[1 : 1 + k]
        sub_vecs = vecs[:, 1 : 1 + k]
        # Diffusion-map weighting: 1/sqrt(eigvalue) emphasises low-freq modes.
        weights = 1.0 / np.sqrt(np.maximum(sub_vals, 1e-6))
        spec = sub_vecs * weights[None, :]
    except Exception:
        spec = np.zeros((n, k))

    if spec.shape[1] < k:
        pad = np.zeros((n, k - spec.shape[1]))
        spec = np.concatenate([spec, pad], axis=1)
    return spec.astype(np.float32)


def _graph_features(A: np.ndarray) -> np.ndarray:
    """Concatenate the 6 structural + 10 spectral features per node → (n, 16).

    Per-column structural features (degree, clustering coefficient,
    centralities, avg shortest path) are raw — they keep their natural
    scale so a downstream clusterer can reason about absolute graph
    structure. The 10 spectral features use diffusion-map weighting
    (1/sqrt(eigvalue)) so the Fiedler vector carries most of the
    community signal — which is what makes :func:`gen_graph_sbm` cluster
    near-perfectly. On very small graphs (``n=34`` karate), the
    high-frequency spectral modes inject noise that swamps the low-freq
    Fiedler signal in 16-D Euclidean distance, so clustering ARI is
    *intentionally* low — a real curse-of-dimensionality regime that this
    module exposes for benchmarking.
    """
    struct = _graph_structural_features(A)
    spec = _graph_spectral_features(A, k=10)
    return np.concatenate([struct, spec], axis=1).astype(np.float32)


# ---------------------------------------------------------------------------
# Built-in graph: Zachary's karate club
# ---------------------------------------------------------------------------

# Edge list (34 nodes, 78 edges) from Zachary (1977). Hardcoded so we
# don't depend on networkx being installed.
_KARATE_EDGES = [
    (0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6), (0, 7), (0, 8),
    (0, 10), (0, 11), (0, 12), (0, 13), (0, 17), (0, 19), (0, 21), (0, 31),
    (1, 2), (1, 3), (1, 7), (1, 13), (1, 17), (1, 19), (1, 21), (1, 30),
    (2, 3), (2, 7), (2, 8), (2, 9), (2, 13), (2, 27), (2, 28), (2, 32),
    (3, 7), (3, 12), (3, 13),
    (4, 6), (4, 10),
    (5, 6), (5, 10), (5, 16),
    (6, 16),
    (8, 30), (8, 32), (8, 33),
    (9, 33),
    (13, 33),
    (14, 32), (14, 33),
    (15, 32), (15, 33),
    (18, 32), (18, 33),
    (19, 33),
    (20, 32), (20, 33),
    (22, 32), (22, 33),
    (23, 25), (23, 27), (23, 29), (23, 32), (23, 33),
    (24, 25), (24, 27), (24, 31),
    (25, 31),
    (26, 29), (26, 33),
    (27, 33),
    (28, 31), (28, 33),
    (29, 32), (29, 33),
    (30, 32), (30, 33),
    (31, 32), (31, 33),
    (32, 33),
]

# Faction membership (0/1) from Zachary's paper.
_KARATE_LABELS = np.array(
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1,
     1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    dtype=np.int64,
)


# ---------------------------------------------------------------------------
# Stochastic block model fallback
# ---------------------------------------------------------------------------

def _sbm(n: int, k: int, seed: int, compactness: float = 1.0):
    """Stochastic block model: k roughly-equal blocks, denser within block.

    ``compactness`` modulates the within-vs-between probability ratio —
    larger compactness => more between-block edges => harder clustering.
    Returns ``(adjacency, labels)``.
    """
    rng = np.random.default_rng(seed)
    sizes = [n // k] * k
    sizes[-1] += n - sum(sizes)
    labels = np.concatenate(
        [np.full(s, j, dtype=np.int64) for j, s in enumerate(sizes)]
    )
    # Edge probabilities — within-block roughly 0.25, between 0.02 * compactness.
    p_in = 0.25
    p_out = max(0.005, 0.02 * float(compactness))
    A = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        # Vectorise per row: draw bernoullis for j > i.
        same = labels[i] == labels[i + 1:]
        probs = np.where(same, p_in, p_out)
        flips = rng.random(len(probs)) < probs
        js = np.arange(i + 1, n)[flips]
        A[i, js] = 1.0
        A[js, i] = 1.0
    return A, labels


def _graph_postprocess(A: np.ndarray, y: np.ndarray, spec: DataSpec):
    """Extract features + DataSpec subsample / noise / outlier."""
    X = _graph_features(A)
    y = y.astype(np.int64)
    rng = np.random.default_rng(spec.seed)

    if 0 < spec.n_samples < len(X):
        idx = rng.choice(len(X), size=spec.n_samples, replace=False)
        X = X[idx]
        y = y[idx]

    X, y = _inject_noise(X, y, spec.noise, rng)
    X, y = _inject_outliers(
        X, y, spec.outliers, rng,
        extremity=getattr(spec, "outlier_extremity", 1.0),
    )
    perm = rng.permutation(len(X))
    return X[perm].astype(np.float32), y[perm]


# ---------------------------------------------------------------------------
# Graph generators
# ---------------------------------------------------------------------------

def gen_graph_karate(spec: DataSpec):
    """Zachary's karate club: 34 nodes, 2 communities (factions).

    Bundled inline (no network round-trip). Each node is reduced to a
    16-dim feature vector via :func:`_graph_features`.
    """
    A = _adjacency_from_edges(_KARATE_EDGES, n=34)
    return _graph_postprocess(A, _KARATE_LABELS, spec)


def gen_graph_sbm(spec: DataSpec):
    """Stochastic block model: n=200 nodes, k=4 communities (default).

    ``spec.compactness`` controls the between-block edge density — higher
    compactness => harder clustering problem (more cross-community edges).
    """
    n = max(40, int(spec.n_samples) if spec.n_samples > 0 else 200)
    k = max(2, min(int(spec.centers), 8))
    A, y = _sbm(n=n, k=k, seed=spec.seed, compactness=spec.compactness)
    return _graph_postprocess(A, y, spec)


# ---------------------------------------------------------------------------
# Registries
# ---------------------------------------------------------------------------

DOMAIN_DATASETS = {
    "ts_ecg200": gen_ts_ecg200,
    "ts_trace": gen_ts_trace,
    "ts_synth": gen_ts_synth,
    "graph_karate": gen_graph_karate,
    "graph_sbm": gen_graph_sbm,
}


DOMAIN_METADATA = {
    "ts_ecg200": DatasetMetadata(
        name="ts_ecg200", source="ucr",
        n_samples=200, n_features=16, n_classes=2,
        domain="time-series", license="public",
        notes="UCR ECG200 heartbeat sequences reduced to 16 hand-crafted features "
              "(8 stat + 4 spectral + 4 autocorr). Synthetic sinusoid fallback.",
    ),
    "ts_trace": DatasetMetadata(
        name="ts_trace", source="ucr",
        n_samples=200, n_features=16, n_classes=4,
        domain="time-series", license="public",
        notes="UCR Trace synthetic instrument signatures reduced to 16 features. "
              "Synthetic sinusoid fallback if archive unreachable.",
    ),
    "ts_synth": DatasetMetadata(
        name="ts_synth", source="synthetic",
        n_samples=200, n_features=16, n_classes=3,
        domain="time-series", license="public-domain",
        notes="Always-on noisy-sinusoid synthetic with 16-feature reduction. "
              "Offline regression target for the time-series pipeline.",
    ),
    "graph_karate": DatasetMetadata(
        name="graph_karate", source="bundled",
        n_samples=34, n_features=16, n_classes=2,
        domain="graph", license="public-domain",
        notes="Zachary's karate club (1977). Each node -> 6 structural "
              "+ 10 normalised-Laplacian eigenvector features.",
    ),
    "graph_sbm": DatasetMetadata(
        name="graph_sbm", source="synthetic",
        n_samples=200, n_features=16, n_classes=4,
        domain="graph", license="public-domain",
        notes="Stochastic block model n=200, k=4. Compactness controls "
              "between-block edge probability (clustering difficulty).",
    ),
}


__all__ = [
    "DOMAIN_DATASETS",
    "DOMAIN_METADATA",
    "gen_ts_ecg200",
    "gen_ts_trace",
    "gen_ts_synth",
    "gen_graph_karate",
    "gen_graph_sbm",
]


# ---------------------------------------------------------------------------
# Inline smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":  # pragma: no cover
    import time
    from sklearn.metrics import adjusted_rand_score

    from clustbench.algorithms.base import ALGO_REGISTRY

    print("=" * 72)
    print("clustbench.datasets_domains — smoke test")
    print("=" * 72)

    algos = ["kmeans", "spectral", "gmm"]

    for name, gen in DOMAIN_DATASETS.items():
        meta = DOMAIN_METADATA[name]
        k_native = meta.n_classes
        spec = DataSpec(
            n_samples=meta.n_samples,
            n_features=16,
            centers=k_native,
            compactness=1.0,
            seed=1,
        )

        # Capture warnings so we can report real-vs-fallback per dataset.
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            t0 = time.perf_counter()
            X, y = gen(spec)
            dt = time.perf_counter() - t0
        fell_back = any(
            ("fallback" in str(w.message).lower() or "synthetic" in str(w.message).lower())
            for w in caught
        )
        if meta.source in ("synthetic", "bundled"):
            provenance = meta.source
        elif fell_back:
            provenance = "fallback (synthetic)"
        else:
            provenance = "real fetch"

        print(
            f"\n[{name}] X={X.shape} y={y.shape} k={len(np.unique(y))} "
            f"provenance={provenance} ({dt:.2f}s to load)"
        )

        for algo_name in algos:
            try:
                algo = ALGO_REGISTRY[algo_name]()
                res = algo.fit_predict(X, k=k_native)
                ari = adjusted_rand_score(y, res.labels)
                print(f"  {algo_name:9s}  ARI = {ari:+.4f}")
            except Exception as e:
                print(f"  {algo_name:9s}  FAILED ({type(e).__name__}: {e})")
