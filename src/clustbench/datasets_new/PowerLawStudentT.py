from __future__ import annotations
import numpy as np

from clustbench.datasets import DataSpec


def gen_PowerLawStudentT(spec: DataSpec) -> tuple[np.ndarray, np.ndarray]:
    """Anisotropic multivariate Student-t mixture with Zipf-distributed masses.

    Generates ``K = spec.centers`` components in ``R^{spec.n_features}`` whose
    mass fractions follow a Zipf(alpha=1.2) distribution so the head component
    holds a large share of samples while tail components each hold only a few
    percent (an example schedule at large n is roughly 3000, 900, 300, 100, 30,
    10). A per-cluster floor of one sample is enforced by borrowing from the
    largest bucket, so every label in ``[0, K)`` is present in ``y``.

    Each component is drawn from a multivariate Student-t distribution with
    degrees of freedom sampled uniformly in ``[1.5, 3]`` (so the fourth moment
    does not exist and centroids get dragged by extreme leverage points). The
    covariance is anisotropic: eigenvalues span a condition number sampled
    uniformly in ``[20, 100]`` on a geometric grid, then rotated by a Haar-random
    orthogonal matrix (obtained from the QR decomposition of a Gaussian matrix
    with sign-corrected ``R`` diagonal). Concretely, offsets are formed by the
    normal/chi-square construction: ``z ~ N(0, Sigma)``,
    ``u ~ ChiSquare(df)/df``, ``offset = z / sqrt(u)``, with no clipping of the
    heavy tail.

    Cluster means are drawn uniformly on a sphere of radius
    ``8.0 * spec.compactness * sqrt(kappa_max)`` where ``kappa_max`` is the
    largest sampled condition number, keeping components identifiable in
    expectation despite the long, thin, rotated ellipsoids. Features are cast to
    ``float32`` and points/labels are jointly shuffled with the same RNG.
    """
    rng = np.random.default_rng(spec.seed)

    d = int(spec.n_features)
    n = int(spec.n_samples)
    k = max(1, int(spec.centers))
    compactness = float(spec.compactness)

    # Zipf(alpha=1.2) mass fractions across k components.
    alpha = 1.2
    ranks = np.arange(1, k + 1, dtype=np.float64)
    weights = 1.0 / (ranks ** alpha)
    weights = weights / weights.sum()

    # Convert to integer counts summing to n, absorb drift into the largest.
    raw = weights * n
    counts = np.rint(raw).astype(np.int64)
    drift = n - int(counts.sum())
    largest_idx = int(np.argmax(counts))
    counts[largest_idx] += drift

    # Enforce a per-cluster floor of 1 sample, borrowing from the largest.
    for j in range(k):
        if j == largest_idx:
            continue
        if counts[j] < 1 and int(counts[largest_idx]) > 1:
            need = 1 - int(counts[j])
            available = int(counts[largest_idx]) - 1
            take = max(0, min(need, available))
            counts[j] += take
            counts[largest_idx] -= take
    counts = np.clip(counts, 0, None)
    # Re-absorb any leftover drift from clipping/borrowing.
    largest_idx = int(np.argmax(counts))
    drift = n - int(counts.sum())
    counts[largest_idx] += drift

    # Sample per-component df in U[1.5, 3] and condition number in U[20, 100].
    dfs = rng.uniform(1.5, 3.0, size=k)
    kappas = rng.uniform(20.0, 100.0, size=k)

    # Draw a Haar-random orthogonal matrix via QR of a Gaussian matrix.
    def haar_orthogonal(dim: int) -> np.ndarray:
        if dim <= 0:
            return np.zeros((0, 0), dtype=np.float64)
        A = rng.standard_normal(size=(dim, dim))
        Q, R = np.linalg.qr(A)
        # Sign-correct the diagonal of R so the distribution over Q is Haar.
        sgn = np.sign(np.diag(R))
        sgn = np.where(sgn == 0.0, 1.0, sgn)
        Q = Q * sgn[np.newaxis, :]
        return Q

    # Cluster means on a sphere sized to keep components separable in expectation
    # despite anisotropic covariances with the largest condition number.
    kappa_max = float(kappas.max()) if k > 0 else 1.0
    r_center = 8.0 * compactness * np.sqrt(kappa_max)
    directions = rng.standard_normal(size=(k, d))
    dir_norms = np.linalg.norm(directions, axis=1, keepdims=True)
    dir_norms = np.where(dir_norms == 0.0, 1.0, dir_norms)
    centers = (directions / dir_norms) * r_center

    total = int(counts.sum())
    X = np.empty((total, d), dtype=np.float64)
    y = np.empty(total, dtype=np.int64)
    offset = 0
    for j in range(k):
        cnt = int(counts[j])
        if cnt <= 0:
            continue

        # Anisotropic diagonal eigenvalues with condition number kappa[j].
        # Geometric spacing from 1 to kappa[j], then random shuffle so the
        # largest axis is not systematically aligned with the first coordinate.
        if d == 1:
            eigvals = np.array([1.0])
        else:
            eigvals = np.geomspace(1.0, kappas[j], num=d)
        eigvals = eigvals * (compactness ** 2)
        rng.shuffle(eigvals)

        # Rotate the diagonal covariance by a Haar-random orthogonal matrix.
        Q = haar_orthogonal(d)
        # Sigma = Q diag(eigvals) Q^T; sample z ~ N(0, Sigma) as Q * (sqrt(eig) * n).
        base = rng.standard_normal(size=(cnt, d))
        scaled = base * np.sqrt(eigvals)[np.newaxis, :]
        z = scaled @ Q.T

        # Student-t via z / sqrt(u), u ~ ChiSquare(df)/df. Guard against zeros.
        u = rng.chisquare(dfs[j], size=cnt) / dfs[j]
        u = np.where(u <= 0.0, np.finfo(np.float64).tiny, u)
        t_offsets = z / np.sqrt(u)[:, np.newaxis]

        X[offset : offset + cnt] = centers[j] + t_offsets
        y[offset : offset + cnt] = j
        offset += cnt

    # Joint shuffle of points and labels using the same RNG.
    perm = rng.permutation(total)
    X = X[perm]
    y = y[perm]

    return X.astype(np.float32), y
