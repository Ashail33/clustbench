"""AMM: Autoencoder-basis exponential-family mixture model.

Same EM core as :class:`~clustbench.algorithms.fmm.Fmm` but the basis
is the bottleneck activations of a shallow autoencoder rather than
random Fourier features or graph Laplacian eigenvectors. Each cluster's
log-density is a linear function of those bottleneck features,
``p_k(x) ∝ exp(α_k · phi(x))``.

Motivation
----------
For high-dimensional sparse data (TF-IDF text vectors, count data,
high-d images), the k-NN graph that drives LMM degrades because
nearest-neighbour distances stop being informative. A tiny autoencoder
trained end-to-end on the data learns a dense, low-dim manifold of its
own, on which mixture-model clustering is well-conditioned. This
mirrors the modern deep-clustering pipeline (encode → cluster the
embedding), but with FMM's principled EM as the clustering layer
instead of plain k-means on the embedding.

Implementation
--------------
The autoencoder is a single ``sklearn.neural_network.MLPRegressor``
shaped as ``X → hidden_sizes → n_components → hidden_sizes reversed → X``,
trained to reconstruct its input. ``phi`` is the activation of the
narrowest hidden layer (the ``n_components``-dim bottleneck) — an
explicit forward pass through the encoder half computes them.

The heat-kernel learning is disabled by default: autoencoder coordinate
axes are not spectrally indexed (unlike Fourier frequencies or graph
eigenvalues), so the per-cluster ``tau`` scaling has no natural
interpretation here.
"""

from __future__ import annotations

from typing import Any, Tuple

import numpy as np

from .base import register
from .fmm import Fmm


@register
class Amm(Fmm):
    """Autoencoder-basis mixture model.

    Parameters
    ----------
    n_components : int
        Bottleneck dimensionality. Becomes ``phi.shape[1]`` and the
        size of each cluster's ``alpha_k``.
    hidden_sizes : tuple[int, ...]
        Encoder hidden layer widths between the input and the
        bottleneck. The decoder mirrors them. ``(128,)`` gives a
        three-hidden-layer net (128 → n_components → 128) which is
        cheap to train and enough capacity for most tabular sizes.
    ae_max_iter : int
        Cap on the LBFGS / Adam iterations inside ``MLPRegressor.fit``.
    ae_alpha : float
        L2 weight decay on the autoencoder.
    ae_activation : {"relu", "tanh", "logistic"}
        Hidden-layer activation. ``"tanh"`` keeps the bottleneck
        bounded which tends to help downstream EM stability.
    ae_solver : {"adam", "lbfgs"}
        ``"lbfgs"`` converges fast on small-to-medium tabular data;
        ``"adam"`` scales better when n is large.
    ae_random_state : int
        RNG seed for the autoencoder.
    Other parameters are inherited from :class:`Fmm`.
    """

    def __init__(
        self,
        n_components: int = 8,
        hidden_sizes: tuple[int, ...] = (128,),
        ae_max_iter: int = 100,
        ae_alpha: float = 1e-4,
        ae_activation: str = "tanh",
        ae_solver: str = "adam",
        ae_random_state: int = 42,
        **kwargs: Any,
    ) -> None:
        # Pass n_frequencies=n_components into Fmm so feature_dim plumbing
        # works. Disable heat kernel + multi-scale, neither of which has
        # a meaningful interpretation on autoencoder coordinates.
        kwargs.setdefault("n_scales", 1)
        kwargs.setdefault("learn_bandwidth", False)
        kwargs.setdefault("tau_init", 0.0)
        super().__init__(n_frequencies=n_components, **kwargs)
        self.name = "amm"
        self.n_components = int(n_components)
        self.hidden_sizes = tuple(int(h) for h in hidden_sizes)
        self.ae_max_iter = int(ae_max_iter)
        self.ae_alpha = float(ae_alpha)
        self.ae_activation = str(ae_activation)
        self.ae_solver = str(ae_solver)
        self.ae_random_state = int(ae_random_state)

    def _basis_action(self) -> dict:
        return {
            "type": "autoencoder_basis",
            "n_components": int(self.n_components),
            "hidden_sizes": list(self.hidden_sizes),
            "activation": self.ae_activation,
            "solver": self.ae_solver,
        }

    def _build_basis(
        self, X: np.ndarray, rng: np.random.Generator
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Train an MLP autoencoder on X, return bottleneck activations as phi.

        The MLP layer layout is ``[hidden... , n_components, ...reversed]``
        so the bottleneck sits at index ``len(hidden_sizes)`` of the
        learned weight stack. We extract activations manually with the
        same forward pass MLPRegressor uses internally.
        """
        from sklearn.neural_network import MLPRegressor

        n, d = X.shape
        encoder_widths = list(self.hidden_sizes) + [self.n_components]
        decoder_widths = list(self.hidden_sizes)[::-1]
        layer_sizes = encoder_widths + decoder_widths

        ae = MLPRegressor(
            hidden_layer_sizes=tuple(layer_sizes),
            activation=self.ae_activation,
            solver=self.ae_solver,
            alpha=self.ae_alpha,
            max_iter=self.ae_max_iter,
            random_state=self.ae_random_state,
            tol=1e-5,
        )
        # Train to reconstruct the input. ``MLPRegressor`` handles input
        # standardisation poorly; we let it use raw X since the
        # autoencoder will adapt to whatever scale.
        ae.fit(X, X)

        # Manual forward pass through the encoder half to recover the
        # bottleneck activations. MLPRegressor exposes ``coefs_`` and
        # ``intercepts_`` indexed by layer.
        bottleneck_idx = len(self.hidden_sizes)  # 0-based index of bottleneck *layer output*
        activation_fn = {
            "relu": lambda z: np.maximum(z, 0.0),
            "tanh": np.tanh,
            "logistic": lambda z: 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30))),
            "identity": lambda z: z,
        }[self.ae_activation]

        a = X
        for layer in range(bottleneck_idx + 1):
            z = a @ ae.coefs_[layer] + ae.intercepts_[layer]
            # All hidden activations use the configured nonlinearity;
            # the final output layer uses linear in MLPRegressor.
            a = activation_fn(z)
        phi = a  # shape (n, n_components)

        # Center so the constant direction doesn't dominate alpha later.
        phi = phi - phi.mean(axis=0, keepdims=True)
        # omega_norm_sq: per-bottleneck-dim variance. If ``learn_bandwidth``
        # is ever flipped on, the heat kernel ``exp(-tau * var / 2)``
        # would damp high-variance directions à la spectral whitening.
        omega_norm_sq = phi.var(axis=0) + 1e-12

        # Record the basis details on the algorithm so ``_basis_action``
        # picks them up via the next trajectory step.
        self._last_recon_loss = float(ae.loss_) if hasattr(ae, "loss_") else float("nan")
        self._last_n_iter = int(ae.n_iter_) if hasattr(ae, "n_iter_") else -1

        # ``omega`` is opaque to the EM core; we pack the spectrum-like
        # quantity (per-dim variance) in for introspection.
        omega = omega_norm_sq.reshape(-1, 1)
        return omega, phi, omega_norm_sq
