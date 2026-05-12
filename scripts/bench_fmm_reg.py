"""FmmReg head-to-head benchmark vs sklearn regression baselines.

Result on commit a05b06d (3 train/test splits, R² on held-out 30%)::

  dataset            n    d    ridge  kernel_ridge   mlp   rand_forest  fmm_reg(K=4)  fmm_reg(K=8)
  piecewise_linear  1500   5    0.46     0.79         0.94    0.69          0.83          0.82
  piecewise_linear  1500  10    0.21     0.33         0.71    0.36          0.39          0.01
  friedman1         1500  10    0.73     0.95         0.84    0.88          0.77          0.71
  friedman1         2000  10*   0.71     0.93         0.83    0.86          0.77          0.72
  sinusoidal        1000   5    0.14     0.85         0.88    0.90          0.68          0.46
  (* noise=1.0)

Mean fit time across configs: ridge 1-60 ms, kernel_ridge 19-169 ms,
mlp 700-1500 ms, random_forest 300-900 ms, fmm_reg(K=4) 300-500 ms,
fmm_reg(K=8) 500-700 ms.

Honest verdict: **FmmReg wins zero of five configs**. Even on
piecewise_linear (the regime it was designed for), MLP beats it
0.94 vs 0.83. More experts (K=8) usually hurts and catastrophically
collapses on the 10-dim piecewise data.

Why: for clustering, the mixture architecture decomposes data
density and the features ARE the representation. For regression, you
also need to learn how y depends on x — MLPs / kernel methods do
this jointly, while MoE-Fourier separates gating and experts and
the separation is harder to optimise without shared representation
between experts.

The architecture pattern transferred cleanly (same Newton M-step,
same EM loop, same gating-via-softmax), but it's not competitive as
a regressor. A real negative result.
"""

from __future__ import annotations

import time

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor

from clustbench.regression.datasets import REG_DATASETS, RegSpec
from clustbench.regression.fmm_reg import FmmReg


BASELINES = {
    "ridge":         lambda: Ridge(alpha=1.0),
    "kernel_ridge":  lambda: KernelRidge(alpha=0.1, kernel="rbf", gamma=0.5),
    "mlp":           lambda: MLPRegressor(
        hidden_layer_sizes=(64, 32), max_iter=300, random_state=1
    ),
    "random_forest": lambda: RandomForestRegressor(n_estimators=100, random_state=1),
    "fmm_reg(K=4)":  lambda: FmmReg(n_components=4, n_frequencies=32, max_iter=40),
    "fmm_reg(K=8)":  lambda: FmmReg(n_components=8, n_frequencies=32, max_iter=40),
}


def main() -> None:
    configs = [
        ("piecewise_linear", 1500, 5, 3, 0.3),
        ("piecewise_linear", 1500, 10, 5, 0.3),
        ("friedman1",        1500, 10, None, 0.3),
        ("friedman1",        2000, 10, None, 1.0),
        ("sinusoidal",       1000, 5, None, 0.2),
    ]
    print(
        f'{"dataset":18s}{"n":>5s}{"d":>4s}  '
        + "  ".join(f"{m:>15s}" for m in BASELINES),
        flush=True,
    )
    for ds, n, d, K, noise in configs:
        spec = RegSpec(n_samples=n, n_features=d, n_components=K or 3, noise=noise, seed=1)
        X, y = REG_DATASETS[ds](spec)
        row = []
        for name, make in BASELINES.items():
            r2s, ts = [], []
            for seed in (1, 2, 3):
                X_tr, X_te, y_tr, y_te = train_test_split(
                    X, y, test_size=0.3, random_state=seed
                )
                t0 = time.time()
                m = make()
                m.fit(X_tr, y_tr)
                pred = m.predict(X_te)
                ts.append(time.time() - t0)
                r2s.append(r2_score(y_te, pred))
            row.append(f"{np.mean(r2s):.3f}/{1000 * np.mean(ts):.0f}ms")
        print(
            f"{ds:18s}{n:>5d}{d:>4d}  "
            + "  ".join(f"{c:>15s}" for c in row),
            flush=True,
        )


if __name__ == "__main__":
    main()
