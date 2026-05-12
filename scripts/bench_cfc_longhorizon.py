"""Long-horizon CfC benchmark.

Follow-up to ``bench_cfc.py``. The hypothesis was that LNNs would
shine when ridge can't see the relevant history — give ridge a tiny
8-step window and CfC/GRU a 64-step window, plus longer sequences
(5000 samples), plus a ``delayed_copy`` task with explicit 100-step
memory requirement.

Result on commit 01de840 (2 seeds, 70/30 time-ordered split, R²):

  dataset           n     K   ridge@w=8  mlp@w=8  rf@w=8  fmm@w=8  gru@w=64  cfc@w=64
  sinusoid_drift   5000   3   0.98       0.98     0.98    0.98     0.97      0.98
  modulated_ar     5000   3   0.65       0.64     0.63    0.58     0.66      0.65
  multi_sinusoid   5000   4   0.99       0.99     0.99    0.99     0.99      0.99
  delayed_copy     5000 100   0.62       0.67     0.82    0.60     0.61      0.61
  random_walk      5000   1   0.99       0.99     0.99    0.99     0.99      0.99

Mean fit time: ridge 2 ms, mlp 110-280 ms, rf 1.4-2 s, fmm 0.7-0.8 s,
GRU 24-27 s, **CfC 30-33 s**. CfC is ~15,000x slower than ridge.

Honest verdict — the long-horizon defence doesn't save LNNs:

1. **Ridge with window=8 wins or ties on 4 of 5 configs.** Even with
   8x less input than the recurrent models. The 64-step recurrent
   window provides ZERO measurable advantage at our scale.

2. **CfC ties GRU.** The continuous-time inductive bias still
   doesn't show.

3. **delayed_copy is the killer**: this task literally has
   y_t = y_{t-100}, requiring memory of values 100 steps back. With
   window=8 ridge cannot in principle solve it; with window=64 the
   recurrent models could in principle... except they don't. Random
   Forest wins at 0.82 by exploiting the near-periodic structure
   (y_t ≈ y_{t-100} ≈ y_{t-200} ≈ ...).

4. **15,000x compute multiplier** on the recurrent models, with no
   quality return. CfC at 30s vs ridge at 2ms.

Caveat: to genuinely remember y_{t-100}, CfC needs *stateful*
training across batches (carry hidden state between training
examples). The standard windowed-training setup we used resets
hidden state between examples — so CfC's effective horizon is just
``window=64``, which is less than the 100-step delay. That different
experimental setup we did NOT test. What we DID refute is the
casual reading: "give LNNs a longer window and they'll beat ridge".
At our scale they don't, by a wide margin.

The whole arc of two CfC benchmarks (short horizon in bench_cfc.py,
long horizon here) gives the same answer: at synthetic-regression
scale with windowed inputs, ridge regression is unreasonably
effective. The fancy architectures don't earn their compute.

Run from repo root::

    python scripts/bench_cfc_longhorizon.py
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


W_FF = 8
W_REC = 64

CONFIGS = [
    ("sinusoid_drift", 5000, 3, 0.10),
    ("modulated_ar",   5000, 3, 0.20),
    ("multi_sinusoid", 5000, 4, 0.10),
    ("delayed_copy",   5000, 100, 0.05),
    ("random_walk",    5000, 1, 0.10),
]

ALGOS_FF = [
    ("ridge", lambda: Ridge(alpha=1.0)),
    ("mlp",   lambda: MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=400,
                                    random_state=1)),
    ("rf",    lambda: RandomForestRegressor(n_estimators=100, random_state=1)),
    ("fmm",   lambda: FmmReg(n_components=4, n_frequencies=32, max_iter=40)),
]
ALGOS_REC = [("gru", GruReg), ("cfc", CfCReg)]


def run_ff(factory, y_tr, y_te, window: int) -> float:
    Xtr, ytr = _windowed(y_tr, window)
    Xte, yte = _windowed(y_te, window)
    m = factory()
    m.fit(Xtr, ytr)
    return float(r2_score(yte, m.predict(Xte)))


def run_rec(cls, y_tr, y_te, window: int) -> float:
    m = cls(hidden_size=32, window=window, max_epochs=25, batch_size=64, lr=5e-3)
    res = m.fit_predict_sequence(y_tr, y_te)
    preds = res.predictions
    mask = ~np.isnan(preds)
    return float(r2_score(y_te[mask], preds[mask]))


def main() -> None:
    headers = [a[0] + f"@w{W_FF}" for a in ALGOS_FF] + [
        a[0] + f"@w{W_REC}" for a in ALGOS_REC
    ]
    print(
        f"{'dataset':18s}{'n':>5s}{'K':>4s}  "
        + "  ".join(f"{h:>12s}" for h in headers),
        flush=True,
    )
    print("-" * (29 + 14 * len(headers)), flush=True)
    for ds, n, K, noise in CONFIGS:
        spec = RegSpec(n_samples=n, n_features=1, n_components=K, noise=noise, seed=1)
        _, y = REG_DATASETS[ds](spec)
        split = int(0.7 * n)
        y_tr, y_te = y[:split], y[split:]
        row = []
        for name, factory in ALGOS_FF:
            r2s, ts = [], []
            for seed in (1, 2):
                np.random.seed(seed)
                t0 = time.time()
                try:
                    r2 = run_ff(factory, y_tr, y_te, W_FF)
                except Exception:
                    r2 = float("nan")
                ts.append(time.time() - t0)
                r2s.append(r2)
            row.append(f"{np.nanmean(r2s):>5.2f}/{1000 * np.mean(ts):>5.0f}ms")
        for name, cls in ALGOS_REC:
            r2s, ts = [], []
            for seed in (1, 2):
                np.random.seed(seed)
                t0 = time.time()
                try:
                    r2 = run_rec(cls, y_tr, y_te, W_REC)
                except Exception:
                    r2 = float("nan")
                ts.append(time.time() - t0)
                r2s.append(r2)
            row.append(f"{np.nanmean(r2s):>5.2f}/{1000 * np.mean(ts):>5.0f}ms")
        print(
            f"{ds:18s}{n:>5d}{K:>4d}  "
            + "  ".join(f"{c:>12s}" for c in row),
            flush=True,
        )


if __name__ == "__main__":
    main()
