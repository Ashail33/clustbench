"""CfC (Liquid Neural Network) vs GRU vs feedforward baselines on
sequence regression.

Tests the hypothesis that the continuous-time inductive bias of
Closed-form Continuous-time (CfC) networks (Hasani et al. 2022)
provides a measurable advantage on non-stationary regression tasks.

Result on commit 86171ea (2 seeds, 24-step lagged input window,
70/30 time-ordered train/test split, R² on holdout):

  dataset           n     K   ridge  mlp    rf     fmm    gru    cfc
  sinusoid_drift   1200   4   0.973  0.969  0.967  0.932  0.972  0.968
  sinusoid_drift   1200   8   0.957  0.948  0.961  0.916  0.959  0.956
  modulated_ar     1500   3   0.550  0.033  0.456 -4.446  0.489  0.525
  multi_sinusoid   1200   4   0.993  0.991  0.977  0.980  0.988  0.990
  random_walk      1500   1   0.977  0.963  0.898  0.928  0.970  0.884

Mean fit time: ridge 1-32 ms, mlp 80-600 ms, rf 750-1200 ms,
fmm 330-410 ms, **gru 3.9-5.2 s, cfc 4.9-6.6 s**.

Honest verdict:

1. **Ridge regression wins 4 of 5 configs**, 100-1000x faster than
   the recurrent baselines. Once you give any model a 24-step lagged
   window of the signal, the regression becomes nearly linear in the
   lags and ridge handles it.

2. **CfC and GRU are statistically indistinguishable** (0.968 vs
   0.972, 0.525 vs 0.489, etc.). The continuous-time inductive bias
   that LNNs are advertised for produces zero visible advantage at
   our scale.

3. **CfC actually loses to GRU on the random-walk control** (0.884
   vs 0.970) — the continuous-time machinery hurts on the very task
   where "just copy the last value" is the optimal predictor.

4. **FmmReg collapses on modulated_ar** (R² = -4.4, worse than
   predicting the mean) — the MoE-Fourier features are useless for
   time-varying autoregressive coefficients.

5. **Control worked**: no algorithm "wins" on random_walk by being
   fancy; ridge edges it because the optimal prediction is y_{t-1},
   which ridge picks up trivially.

Caveats: LNNs are advertised for (a) long effective horizons, (b)
genuinely non-linear lag structure that ridge can't absorb, and (c)
larger networks. None of these apply in our setup. The result is
specific to "synthetic regression with 24-step lagged window, ~1500
samples, 1-D signal" — it doesn't refute LNNs on robotics control,
real time-series, or autoregressive language modelling. It does
refute the broad claim that they offer free wins on small synthetic
non-stationary tasks.

Run from repo root::

    python scripts/bench_cfc.py
"""

from __future__ import annotations

import time

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.neural_network import MLPRegressor

from clustbench.regression.cfc_reg import CfCReg, GruReg, _windowed
from clustbench.regression.datasets import REG_DATASETS, RegSpec
from clustbench.regression.fmm_reg import FmmReg


def run_ff(model_factory, y_tr, y_te, window: int) -> float:
    Xtr, ytr = _windowed(y_tr, window)
    Xte, yte = _windowed(y_te, window)
    m = model_factory()
    m.fit(Xtr, ytr)
    return float(r2_score(yte, m.predict(Xte)))


def run_rec(cls, y_tr, y_te, window: int) -> float:
    m = cls(hidden_size=32, window=window, max_epochs=40, batch_size=64, lr=5e-3)
    res = m.fit_predict_sequence(y_tr, y_te)
    preds = res.predictions
    mask = ~np.isnan(preds)
    return float(r2_score(y_te[mask], preds[mask]))


CONFIGS = [
    ("sinusoid_drift", 1200, 4, 0.10),
    ("sinusoid_drift", 1200, 8, 0.10),
    ("modulated_ar",   1500, 3, 0.20),
    ("multi_sinusoid", 1200, 4, 0.10),
    ("random_walk",    1500, 1, 0.10),
]

ALGOS = [
    ("ridge", "ff", lambda: Ridge(alpha=1.0)),
    ("mlp",   "ff", lambda: MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=400,
                                          random_state=1)),
    ("rf",    "ff", lambda: RandomForestRegressor(n_estimators=100, random_state=1)),
    ("fmm",   "ff", lambda: FmmReg(n_components=4, n_frequencies=32, max_iter=40)),
    ("gru",   "rec", GruReg),
    ("cfc",   "rec", CfCReg),
]
WINDOW = 24


def main() -> None:
    print(
        f"{'dataset':18s}{'n':>5s}{'K':>3s}  "
        + "  ".join(f"{a[0]:>11s}" for a in ALGOS),
        flush=True,
    )
    print("-" * (28 + 13 * len(ALGOS)), flush=True)
    for ds, n, K, noise in CONFIGS:
        spec = RegSpec(n_samples=n, n_features=1, n_components=K, noise=noise, seed=1)
        _, y = REG_DATASETS[ds](spec)
        split = int(0.7 * n)
        y_tr, y_te = y[:split], y[split:]
        row = []
        for name, kind, factory in ALGOS:
            r2s, ts = [], []
            for seed in (1, 2):
                np.random.seed(seed)
                t0 = time.time()
                try:
                    r2 = (run_rec(factory, y_tr, y_te, WINDOW) if kind == "rec"
                          else run_ff(factory, y_tr, y_te, WINDOW))
                except Exception:
                    r2 = float("nan")
                ts.append(time.time() - t0)
                r2s.append(r2)
            row.append(f"{np.nanmean(r2s):>6.3f}/{1000 * np.mean(ts):>4.0f}ms")
        print(
            f"{ds:18s}{n:>5d}{K:>3d}  "
            + "  ".join(f"{c:>12s}" for c in row),
            flush=True,
        )


if __name__ == "__main__":
    main()
