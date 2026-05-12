"""Regression module: mixture-of-experts variants of the FMM/LMM/SMM family.

Same architectural pattern as the clustering side — features ``phi(x)``
plus a per-cluster (here per-*expert*) mixture model — but each
"component" is a regression on ``y`` rather than a density over ``x``.

Parallels:
  * clustering side: ``p_k(x) ∝ exp(alpha_k . phi(x))``, fit by EM to
    recover cluster labels.
  * regression side (this module): same gating
    ``pi_k(x) ∝ exp(alpha_k . phi(x))`` plus per-expert prediction
    ``ŷ_k(x) = beta_k . phi(x)``, fit by EM to minimise MSE.

Public surface lives in ``src/clustbench/regression/``:
  * ``base.py``: ``Regressor`` ABC, ``RegResult`` dataclass, registry.
  * ``fmm_reg.py``: ``FmmReg`` (MoE-Fourier).
  * ``datasets.py``: synthetic regression generators.
"""
