"""CfCReg + GruReg: continuous-time and discrete-time recurrent regression.

From-scratch implementation of a Closed-form Continuous-time (CfC) cell
following Hasani et al. (2022, *Closed-form Continuous-time Neural
Networks*, Nature Machine Intelligence). The CfC is the analytical
solution to the Liquid Time-Constant (LTC) ODE that the MIT group
introduced — runs like an RNN cell (no ODE solver needed) but encodes
the LTC's continuous-time inductive bias: each unit's time constant
depends on the input, so the network adapts its own dynamics on the fly.

The minimal CfC update rule used here (see Hasani 2022, eq. 5)::

    f(x, h) = tanh(W_f [x, h] + b_f)
    g(x, h) = tanh(W_g [x, h] + b_g)
    t_int  = sigmoid(W_t [x, h] + b_t)   # input-dependent gating in (0, 1)
    h_new  = f * (1 - t_int) + g * t_int

This is the "minimal CfC" (also called CfC-tanh) — the closed-form
solution to the LTC ODE when the activation is tanh, with a learned
input-dependent gate ``t_int`` playing the role of the inverse time
constant.

Both ``CfCReg`` and ``GruReg`` (the matched-size discrete-time control)
expose the same ``fit``/``predict`` interface as ``FmmReg``.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..algorithms.base import Step
from .base import RegResult, Regressor, register_regressor


def _windowed(y: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray]:
    """Sliding-window framing: input = y[t-w:t], target = y[t]."""
    n = y.shape[0]
    n_seq = n - window
    if n_seq <= 0:
        raise ValueError(f"sequence too short for window={window} (got len={n})")
    X = np.lib.stride_tricks.sliding_window_view(y, window)[:n_seq]
    target = y[window:]
    return X.astype(np.float32), target.astype(np.float32)


class _RecurrentRegBase(Regressor):
    """Shared train/predict logic for CfC and GRU regressors.

    Subclasses define ``_build_module`` returning a torch ``nn.Module``
    that takes ``(batch, seq, 1)`` and returns ``(batch,)`` predictions.
    The base class handles framing, batching, optimisation, and
    trajectory recording.
    """

    cell_kind: str = "rnn"

    def __init__(
        self,
        hidden_size: int = 32,
        window: int = 32,
        max_epochs: int = 60,
        batch_size: int = 64,
        lr: float = 5e-3,
        weight_decay: float = 1e-4,
        random_state: int = 42,
        **kwargs: Any,
    ) -> None:
        self.hidden_size = int(hidden_size)
        self.window = int(window)
        self.max_epochs = int(max_epochs)
        self.batch_size = int(batch_size)
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)
        self.random_state = int(random_state)
        self._module = None
        self._trajectory: list[Step] = []

    def _build_module(self):
        raise NotImplementedError

    def fit(self, X: np.ndarray, y: np.ndarray) -> "_RecurrentRegBase":
        import torch

        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        # The recurrent path treats the 1-D time series as the signal;
        # X is the timestamp (or any 1-D index) and is unused here.
        y = np.asarray(y, dtype=np.float32).ravel()
        Xw, yt = _windowed(y, self.window)
        X_t = torch.from_numpy(Xw).unsqueeze(-1)        # (n_seq, window, 1)
        y_t = torch.from_numpy(yt)                       # (n_seq,)

        self._module = self._build_module()
        opt = torch.optim.Adam(
            self._module.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        loss_fn = torch.nn.MSELoss()
        self._trajectory = []

        n_seq = X_t.shape[0]
        for epoch in range(self.max_epochs):
            perm = torch.randperm(n_seq)
            total = 0.0
            n_batches = 0
            for start in range(0, n_seq, self.batch_size):
                idx = perm[start : start + self.batch_size]
                xb = X_t[idx]
                yb = y_t[idx]
                opt.zero_grad()
                pred = self._module(xb)
                loss = loss_fn(pred, yb)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self._module.parameters(), 1.0)
                opt.step()
                total += float(loss.item())
                n_batches += 1
            mean_loss = total / max(n_batches, 1)
            self._trajectory.append(
                Step(
                    step_idx=epoch,
                    cost=mean_loss,
                    accepted=True,
                    action={"type": f"{self.cell_kind}_epoch", "epoch": int(epoch)},
                    state={"train_mse": float(mean_loss)},
                )
            )
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """X is ignored — the recurrent regressor predicts directly from
        the windowed signal stored at fit-time. For the held-out test
        the caller passes the *test signal* through ``predict_sequence``."""
        raise NotImplementedError(
            "Recurrent regressors predict from a sequence directly; "
            "use ``predict_sequence(y)`` on the held-out signal."
        )

    def predict_sequence(self, y: np.ndarray) -> np.ndarray:
        """One-step-ahead predictions for each ``t >= window`` in ``y``."""
        import torch

        if self._module is None:
            raise RuntimeError(f"{self.cell_kind} regressor predict called before fit.")
        y = np.asarray(y, dtype=np.float32).ravel()
        Xw, _ = _windowed(y, self.window)
        self._module.eval()
        with torch.no_grad():
            preds = self._module(torch.from_numpy(Xw).unsqueeze(-1)).cpu().numpy()
        # Pad with NaN for the first ``window`` timesteps so the output
        # length matches the input length.
        out = np.full_like(y, np.nan)
        out[self.window :] = preds
        return out


@register_regressor
class CfCReg(_RecurrentRegBase):
    """Closed-form Continuous-time recurrent regressor."""

    cell_kind = "cfc"

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.name = "cfc_reg"

    def _build_module(self):
        import torch
        import torch.nn as nn

        class CfCCell(nn.Module):
            def __init__(self, in_features: int, hidden_size: int) -> None:
                super().__init__()
                self.hidden_size = hidden_size
                # Three parallel linear maps over [x, h] for f, g, t_int.
                self.lin_f = nn.Linear(in_features + hidden_size, hidden_size)
                self.lin_g = nn.Linear(in_features + hidden_size, hidden_size)
                self.lin_t = nn.Linear(in_features + hidden_size, hidden_size)

            def forward(self, x: "torch.Tensor", h: "torch.Tensor") -> "torch.Tensor":
                z = torch.cat([x, h], dim=-1)
                f = torch.tanh(self.lin_f(z))
                g = torch.tanh(self.lin_g(z))
                t_int = torch.sigmoid(self.lin_t(z))
                return f * (1.0 - t_int) + g * t_int

        class CfCNet(nn.Module):
            def __init__(self, hidden_size: int) -> None:
                super().__init__()
                self.cell = CfCCell(1, hidden_size)
                self.head = nn.Linear(hidden_size, 1)
                self.hidden_size = hidden_size

            def forward(self, x_seq: "torch.Tensor") -> "torch.Tensor":
                # x_seq: (batch, T, 1)
                batch, T, _ = x_seq.shape
                h = torch.zeros(batch, self.hidden_size, device=x_seq.device,
                                dtype=x_seq.dtype)
                for t in range(T):
                    h = self.cell(x_seq[:, t, :], h)
                return self.head(h).squeeze(-1)

        return CfCNet(self.hidden_size)

    def fit_predict_sequence(self, y_train: np.ndarray, y_test: np.ndarray) -> RegResult:
        self.fit(None, y_train)
        preds = self.predict_sequence(y_test)
        return RegResult(predictions=preds, extra={
            "hidden_size": self.hidden_size,
            "window": self.window,
            "n_epochs": int(len(self._trajectory)),
        }, trajectory=self._trajectory)


@register_regressor
class GruReg(_RecurrentRegBase):
    """GRU regressor — discrete-time recurrent control for CfC."""

    cell_kind = "gru"

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.name = "gru_reg"

    def _build_module(self):
        import torch
        import torch.nn as nn

        class GruNet(nn.Module):
            def __init__(self, hidden_size: int) -> None:
                super().__init__()
                self.gru = nn.GRU(1, hidden_size, num_layers=1, batch_first=True)
                self.head = nn.Linear(hidden_size, 1)

            def forward(self, x_seq: "torch.Tensor") -> "torch.Tensor":
                out, _ = self.gru(x_seq)        # (batch, T, hidden)
                return self.head(out[:, -1, :]).squeeze(-1)

        return GruNet(self.hidden_size)

    def fit_predict_sequence(self, y_train: np.ndarray, y_test: np.ndarray) -> RegResult:
        self.fit(None, y_train)
        preds = self.predict_sequence(y_test)
        return RegResult(predictions=preds, extra={
            "hidden_size": self.hidden_size,
            "window": self.window,
            "n_epochs": int(len(self._trajectory)),
        }, trajectory=self._trajectory)
