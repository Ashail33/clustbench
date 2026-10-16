"""TBMM: tri-basis exponential-family mixture (Fourier + Laplacian + autoencoder).

FMM, LMM and AMM share one EM core and differ only in the basis ``phi``
their component log-densities are linear in:

* FMM: random Fourier features (shape-agnostic, multi-scale),
* LMM: bottom eigenvectors of the normalised kNN-graph Laplacian
  (connectivity / manifold structure),
* AMM: bottleneck activations of a shallow autoencoder (learned, dense).

They win on different regimes, so TBMM concatenates the three into one
feature map and fits the same mixture on top:

    log p_k(x) ∝ alpha_k^F . phi_F(x) + alpha_k^L . phi_L(x) + alpha_k^A . phi_A(x)

Each block is standardised column-wise and then divided by
``sqrt(block_dim)``, so every block contributes the same total variance
regardless of how many columns it has (the Fourier block has ~128 columns,
the Laplacian ~k, the autoencoder ~8). ``block_weights`` re-weights blocks
after that; ``blocks`` selects a subset, which gives the two-basis
ablations (e.g. ``blocks=("laplacian", "fourier")``).

The per-cluster heat kernel is disabled: tau has no shared meaning across
blocks with different spectra. When the Laplacian block is present, EM is
warm-started from spectral clustering (k-means on the row-normalised
eigenvectors) rather than FMM's k-means on X, which cannot start from a
non-convex partition.
"""
from __future__ import annotations

from typing import Any, Dict, Sequence, Tuple

import numpy as np

from .amm import Amm
from .base import register
from .fmm import Fmm
from .lmm import Lmm

_BLOCKS = ("fourier", "laplacian", "autoencoder")


@register
class Tbmm(Fmm):
    """Mixture of exponential families over concatenated Fourier, Laplacian
    and autoencoder bases.

    Parameters
    ----------
    blocks : sequence of {"fourier", "laplacian", "autoencoder"}
        Which bases to concatenate (default: all three).
    block_weights : dict, optional
        Extra multiplicative weight per block after equal-variance scaling.
    n_frequencies, n_scales :
        Passed to the Fourier block (as in :class:`Fmm`).
    n_eigvecs, n_neighbors :
        Passed to the Laplacian block (as in :class:`Lmm`).
    n_components, hidden_sizes, ae_max_iter :
        Passed to the autoencoder block (as in :class:`Amm`).
    Other parameters are inherited from :class:`Fmm`.
    """

    def __init__(
        self,
        blocks: Sequence[str] = _BLOCKS,
        block_weights: Dict[str, float] | None = None,
        n_frequencies: int = 32,
        n_scales: int = 3,
        n_eigvecs: Any = "auto",
        n_neighbors: Any = "auto",
        n_components: int = 8,
        hidden_sizes: tuple[int, ...] = (64,),
        ae_max_iter: int = 200,
        **kwargs: Any,
    ) -> None:
        kwargs.setdefault("learn_bandwidth", False)
        kwargs.setdefault("tau_init", 0.0)
        super().__init__(n_frequencies=n_frequencies, n_scales=n_scales, **kwargs)
        self.name = "tbmm"
        bad = set(blocks) - set(_BLOCKS)
        if bad or not blocks:
            raise ValueError(f"blocks must be a non-empty subset of {_BLOCKS}, got {blocks}")
        self.blocks = tuple(b for b in _BLOCKS if b in blocks)
        self.block_weights = dict(block_weights or {})
        self.fourier_frequencies = int(n_frequencies)
        self.fourier_scales = int(n_scales)
        self.n_eigvecs = n_eigvecs
        self.n_neighbors = n_neighbors
        self.n_components = int(n_components)
        self.hidden_sizes = tuple(hidden_sizes)
        self.ae_max_iter = int(ae_max_iter)
        self._k_hint: int | None = None
        self._block_dims: Dict[str, int] = {}
        self._lap_raw: np.ndarray | None = None

    def _basis_action(self) -> dict:
        return {"type": "tri_basis", "blocks": list(self.blocks), "block_dims": dict(self._block_dims)}

    def _block(self, name: str, X: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        seed = self.random_state
        if name == "fourier":
            sub = Fmm(n_frequencies=self.fourier_frequencies, n_scales=self.fourier_scales,
                      random_state=seed)
        elif name == "laplacian":
            sub = Lmm(n_eigvecs=self.n_eigvecs, n_neighbors=self.n_neighbors, random_state=seed)
            sub._k_hint = self._k_hint
            sub.n_frequencies = sub._resolve_n_eigvecs(self._k_hint)
        else:
            sub = Amm(n_components=self.n_components, hidden_sizes=self.hidden_sizes,
                      ae_max_iter=self.ae_max_iter, random_state=seed)
        _, phi, _ = sub._build_basis(X, rng)
        phi = np.asarray(phi, dtype=np.float64)
        if name == "laplacian":
            self._lap_raw = phi
        return phi

    def _build_basis(
        self, X: np.ndarray, rng: np.random.Generator
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        parts = []
        for name in self.blocks:
            phi = self._block(name, X, rng)
            phi = phi - phi.mean(axis=0, keepdims=True)
            sd = phi.std(axis=0)
            keep = sd > 1e-10  # drop constant columns (e.g. the trivial eigenvector)
            phi = phi[:, keep] / sd[keep]
            phi = phi / np.sqrt(max(phi.shape[1], 1))
            phi = phi * float(self.block_weights.get(name, 1.0))
            self._block_dims[name] = int(phi.shape[1])
            parts.append(phi)
        phi = np.concatenate(parts, axis=1)
        omega_norm_sq = np.zeros(phi.shape[1])  # heat kernel disabled (tau = 0)
        omega = omega_norm_sq.reshape(-1, 1)
        return omega, phi, omega_norm_sq

    def _init_alpha(self, X, k, phi, rng):
        if self._lap_raw is None:
            return super()._init_alpha(X, k, phi, rng)
        from sklearn.cluster import KMeans

        eps = 1e-12
        L = self._lap_raw / np.maximum(np.linalg.norm(self._lap_raw, axis=1, keepdims=True), eps)
        km = KMeans(n_clusters=k, n_init=5, random_state=self.random_state).fit(L)
        labels = km.labels_
        m2 = phi.shape[1]
        global_mean = phi.mean(axis=0)
        phi_std = phi.std(axis=0) + eps
        alpha = np.zeros((k, m2))
        for j in range(k):
            members = phi[labels == j]
            if members.size:
                alpha[j] = (members.mean(axis=0) - global_mean) / phi_std / np.sqrt(m2)
        alpha = alpha + rng.normal(scale=1e-3, size=alpha.shape)
        return (alpha, {"type": "spectral_init", "n_clusters": int(k)},
                {"cost": float(km.inertia_)})

    def fit_predict(self, X, k=None):
        self._lap_raw = None
        self._k_hint = k
        result = super().fit_predict(X, k=k)
        if result.extra is not None:
            result.extra["blocks"] = list(self.blocks)
            result.extra["block_dims"] = dict(self._block_dims)
        return result
