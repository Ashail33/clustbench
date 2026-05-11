"""MRIC: Magnetic-Resonance-Imaging-inspired clustering.

Real MRI distinguishes tissues (fat, muscle, cerebrospinal fluid, ...) by
exploiting the fact that hydrogen nuclei in different chemical environments
relax differently after being excited by a radio-frequency pulse. The
machine

  1. applies a strong static field ``B0`` that aligns spin magnetization
     along one axis;
  2. drives a 90 deg RF pulse that tips the magnetization into the
     transverse plane;
  3. records the free-induction-decay signal as the magnetization both
     precesses about ``B0`` at a position-dependent Larmor frequency and
     relaxes back to equilibrium with tissue-specific time constants
     ``T1`` (longitudinal recovery) and ``T2`` (transverse decay).

The same idea works as a clustering primitive: if we treat every data
point as a spin and let its local neighbourhood play the role of its
chemical environment, then a per-point relaxation signature will be a
nonlinear summary of local geometry. Points that sit in geometrically
similar pockets of the feature space yield similar signatures, just as
real fat vs. muscle voxels yield similar (T1, T2) pairs. A final
k-means on the per-point signatures then recovers the clusters.

Pipeline
--------
1. **B0 alignment.** Choose the first principal axis of the centred
   data as the static field. Equilibrium magnetization ``M0`` points
   along ``B0`` for every spin.

2. **Local environment probing.** For each point compute
   - ``spin_density`` from the inverse mean-distance to its k nearest
     neighbours (dense neighbourhood -> high density), and
   - ``anisotropy`` from the eigenvalues of the local covariance
     (perfectly isotropic = 0, highly elongated = 1).

   These map to per-point relaxation times in the spirit of fat vs.
   tissue:

       T1_i = T1_min + (T1_max - T1_min) * (1 - density_i)
       T2_i = T2_min + (T2_max - T2_min) * (1 - anisotropy_i)
       T2_i = min(T2_i, T1_i)              # physical constraint

3. **Gradient encoding.** Use the top ``n_gradient_axes`` principal
   directions as virtual readout gradients. Each point's projection on
   axis ``a`` sets a per-axis Larmor offset
       omega_i,a = gamma * (Xc @ Vt[a]) / std_a.

4. **90 deg RF pulse.** Tip magnetization into the transverse plane:
   M_xy(0) = M0 * density_i, M_z(0) = 0.

5. **FID acquisition.** At each echo time ``t`` record per-point
       Re_i,a(t) = M_xy(0)_i * exp(-t/T2_i) * cos(omega_i,a * t),
       Im_i,a(t) = M_xy(0)_i * exp(-t/T2_i) * sin(omega_i,a * t),
       Mz_i(t)   = M0      * (1 - exp(-t/T1_i)).

6. **Signature clustering.** Stack (density, anisotropy, T1, T2, and
   Re/Im/Mz for every echo time) into a per-point feature vector,
   standardize, and run k-means.

The trajectory captures one :class:`Step` per phase so the dashboard
can play back the acquisition.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors

from .base import Algorithm, AlgoResult, Step, register


@register
class Mri(Algorithm):
    """Magnetic-Resonance-Imaging-inspired clustering (MRIC).

    Parameters
    ----------
    n_neighbors : int
        Neighbourhood size used to estimate local density, anisotropy,
        and therefore the per-point T1/T2 times.
    n_echoes : int
        Number of echo times sampled between ``te_min`` and ``te_max``.
    te_min, te_max : float
        Echo-time range (in arbitrary units of T2). Defaults are tuned
        so both fast-decay ("fat") and slow-decay ("water") signatures
        are resolved.
    n_gradient_axes : int
        How many principal directions to use as readout gradients. Each
        axis adds a real/imag pair of features per echo, plus its raw
        projection as a position-encoding feature.
    max_phase : float
        Maximum Larmor phase reached at the largest echo time / largest
        gradient projection, in radians. ``omega`` is scaled so that
        ``max_i,a,t |omega_i,a * t| == max_phase``; keeping this below
        ``pi`` avoids aliasing in the sinusoidal encoding (real MRI is
        equally subject to Nyquist).
    t1_range, t2_range : tuple[float, float]
        Min/max relaxation times mapped from density/anisotropy.
    random_state : int
        Seed for the final k-means.
    n_init, max_iter : int
        k-means restarts and per-restart iteration cap.
    """

    def __init__(
        self,
        n_neighbors: int = 15,
        n_echoes: int = 6,
        te_min: float = 0.1,
        te_max: float = 2.5,
        n_gradient_axes: int = 3,
        max_phase: float = 3.0,  # < pi avoids Nyquist aliasing
        t1_range: tuple[float, float] = (0.2, 2.0),
        t2_range: tuple[float, float] = (0.05, 1.0),
        random_state: int = 42,
        n_init: int = 10,
        max_iter: int = 300,
        **kwargs: Any,
    ) -> None:
        self.name = "mri"
        self.n_neighbors = n_neighbors
        self.n_echoes = n_echoes
        self.te_min = te_min
        self.te_max = te_max
        self.n_gradient_axes = n_gradient_axes
        self.max_phase = max_phase
        self.t1_range = t1_range
        self.t2_range = t2_range
        self.random_state = random_state
        self.n_init = n_init
        self.max_iter = max_iter

    def fit_predict(self, X: np.ndarray, k: Optional[int] = None) -> AlgoResult:
        assert k is not None, "k (number of clusters) must be provided"
        X = np.asarray(X, dtype=np.float64)
        n, d = X.shape
        eps = 1e-12
        trajectory: list[Step] = []

        # --- Phase 1: B0 alignment via SVD of the centred data.
        Xc = X - X.mean(axis=0, keepdims=True)
        # Truncated SVD via numpy is fine for the feature dims we hit.
        _, S_sv, Vt = np.linalg.svd(Xc, full_matrices=False)
        B0 = Vt[0]
        explained = float((S_sv[0] ** 2) / (np.sum(S_sv ** 2) + eps))
        trajectory.append(
            Step(
                step_idx=0,
                cost=0.0,
                accepted=True,
                action={"type": "b0_align", "n_dim": int(d)},
                state={"B0_explained_var": explained},
            )
        )

        # --- Phase 2: local environment -> spin density + anisotropy.
        k_nn = max(2, min(self.n_neighbors, n - 1))
        nn = NearestNeighbors(n_neighbors=k_nn + 1).fit(X)
        dists, idx = nn.kneighbors(X)
        # Drop the self-neighbour at column 0.
        dists = dists[:, 1:]
        idx = idx[:, 1:]

        mean_dist = dists.mean(axis=1)
        inv = 1.0 / (mean_dist + eps)
        spin_density = inv / (inv.max() + eps)

        # Neighbourhood covariance eigenvalues per point.
        # Vectorise: gather neighbour offsets in one (n, k_nn, d) tensor.
        nbr_offsets = X[idx] - X[:, None, :]
        # Local covariance: (n, d, d).
        cov = np.einsum("nkd,nke->nde", nbr_offsets, nbr_offsets) / k_nn
        # eigvalsh returns ascending eigenvalues.
        evals = np.linalg.eigvalsh(cov)
        ev_max = np.clip(evals[:, -1], eps, None)
        ev_min = np.clip(evals[:, 0], 0.0, None)
        anisotropy = 1.0 - ev_min / ev_max

        # --- Phase 3: relaxation times from local environment.
        t1_min, t1_max = self.t1_range
        t2_min, t2_max = self.t2_range
        T1 = t1_min + (t1_max - t1_min) * (1.0 - spin_density)
        T2 = t2_min + (t2_max - t2_min) * (1.0 - anisotropy)
        T2 = np.minimum(T2, T1)  # physical: T2 <= T1.
        trajectory.append(
            Step(
                step_idx=1,
                cost=float(mean_dist.mean()),
                accepted=True,
                action={"type": "local_probe", "n_neighbors": int(k_nn)},
                state={
                    "density_mean": float(spin_density.mean()),
                    "anisotropy_mean": float(anisotropy.mean()),
                    "T1_mean": float(T1.mean()),
                    "T2_mean": float(T2.mean()),
                },
            )
        )

        # --- Phase 4: gradient encoding -> per-point Larmor frequencies.
        # Project onto the top principal axes (the "readout gradients"),
        # then scale gamma so the largest phase reached over the FID
        # sequence is exactly ``max_phase`` radians. Real MRI is bound
        # by the same Nyquist constraint when picking gradient strength.
        n_axes = max(1, min(self.n_gradient_axes, d, Vt.shape[0]))
        G = Vt[:n_axes]  # (n_axes, d)
        proj = Xc @ G.T  # (n, n_axes)
        proj_scale = np.abs(proj).max() + eps
        proj_n = proj / proj_scale  # in [-1, 1]
        gamma = self.max_phase / max(self.te_max, eps)
        omega = gamma * proj_n  # (n, n_axes)
        trajectory.append(
            Step(
                step_idx=2,
                cost=0.0,
                accepted=True,
                action={"type": "gradient_encode", "n_axes": int(n_axes)},
                state={
                    "gamma": float(gamma),
                    "omega_abs_max": float(np.max(np.abs(omega))),
                },
            )
        )

        # --- Phase 5: 90 deg RF pulse tips M into transverse plane.
        M_xy0 = spin_density.copy()
        trajectory.append(
            Step(
                step_idx=3,
                cost=0.0,
                accepted=True,
                action={"type": "rf_pulse_90"},
                state={"M_xy_total": float(M_xy0.sum())},
            )
        )

        # --- Phase 6: FID acquisition at each echo time.
        # The free-induction-decay signal naturally factors into two
        # observables that real MRI separates into different pulse
        # sequences:
        #   * k-space: cos(omega*t) and sin(omega*t) at each echo time
        #     -- pure position / Fourier encoding, no tissue dependence.
        #   * relaxation envelopes: exp(-t/T2) (transverse decay) and
        #     1 - exp(-t/T1) (longitudinal recovery) -- pure tissue
        #     contrast, no position dependence.
        # Keeping them in separate blocks prevents tissue noise from
        # corrupting the position signal (and vice versa) when one of
        # the two is uniform in the data.
        echo_times = np.linspace(self.te_min, self.te_max, self.n_echoes)
        kspace_cols: list[np.ndarray] = []
        relax_env_cols: list[np.ndarray] = []
        for ti, t in enumerate(echo_times):
            phase = omega * t                                # (n, n_axes)
            re = np.cos(phase)                               # (n, n_axes)
            im = np.sin(phase)                               # (n, n_axes)
            t2_decay = np.exp(-t / T2).reshape(-1, 1)        # (n, 1)
            mz = (1.0 - np.exp(-t / T1)).reshape(-1, 1)      # (n, 1)
            kspace_cols.append(re)
            kspace_cols.append(im)
            relax_env_cols.append(t2_decay)
            relax_env_cols.append(mz)
            trajectory.append(
                Step(
                    step_idx=4 + ti,
                    cost=float(np.linalg.norm(re) + np.linalg.norm(im)),
                    accepted=True,
                    action={"type": "fid_acquire", "te": float(t)},
                    state={
                        "te": float(t),
                        "t2_decay_mean": float(t2_decay.mean()),
                        "longitudinal_mean": float(mz.mean()),
                    },
                )
            )

        relax_block = np.column_stack(
            [spin_density, anisotropy, T1, T2, np.concatenate(relax_env_cols, axis=1)]
        )
        position_block = proj.copy()
        kspace_block = np.concatenate(kspace_cols, axis=1)

        def _balance(block: np.ndarray) -> np.ndarray:
            centred = block - block.mean(axis=0, keepdims=True)
            scale = np.linalg.norm(centred) / np.sqrt(max(centred.shape[0], 1))
            return centred / (scale + eps)

        F = np.concatenate(
            [_balance(relax_block), _balance(position_block), _balance(kspace_block)],
            axis=1,
        )

        # --- Phase 7: signature k-means.
        km = KMeans(
            n_clusters=k,
            n_init=self.n_init,
            max_iter=self.max_iter,
            random_state=self.random_state,
        )
        labels = km.fit_predict(F).astype(np.int64)
        trajectory.append(
            Step(
                step_idx=4 + len(echo_times),
                cost=float(km.inertia_),
                accepted=True,
                action={"type": "signature_kmeans", "n_features": int(F.shape[1])},
                state={"inertia": float(km.inertia_)},
            )
        )

        return AlgoResult(
            labels=labels,
            extra={
                "inertia": float(km.inertia_),
                "n_signature_features": int(F.shape[1]),
                "n_echoes": int(self.n_echoes),
                "n_gradient_axes": int(n_axes),
                "B0_explained_var": explained,
                "T1_mean": float(T1.mean()),
                "T2_mean": float(T2.mean()),
                "density_mean": float(spin_density.mean()),
                "anisotropy_mean": float(anisotropy.mean()),
            },
            trajectory=trajectory,
        )
