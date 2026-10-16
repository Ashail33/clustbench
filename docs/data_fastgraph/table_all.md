# Graph clustering at n = 20k and 100k (d = 20, 11 families, 2 seeds)

`*_lobpcg` rows are the fast variants before the AMG eigensolver (round 19).

| algorithm | ARI 20k | time 20k (s) | ARI 100k | time 100k (s) |
|---|---|---|---|---|
| `lmm` | 0.846 | 2.20 | 0.837 | 40.10 |
| `lmm_fast` | 0.853 | 1.19 | 0.844 | 8.53 |
| `spectral` | 0.853 | 7.17 | — | — |
| `spectral_fast` | 0.855 | 2.55 | 0.832 | 8.92 |
| `mcg` | 0.850 | 1.30 | 0.853 | 1.50 |
| `mcg_cf` | 0.855 | 0.54 | 0.859 | 0.74 |
| `lmm_mcg` | 0.856 | 0.90 | 0.858 | 1.96 |
| `lmm_mcg_dual` | 0.847 | 1.36 | 0.858 | 3.29 |
| `gmm` | 0.537 | 0.35 | 0.557 | 1.32 |
| `trident` | 0.556 | 1.66 | 0.570 | 2.50 |
| `minibatch_kmeans` | 0.554 | 0.13 | 0.555 | 0.67 |
| `lmm_fast_lobpcg` | 0.842 | 1.31 | 0.836 | 10.31 |
| `spectral_fast_lobpcg` | 0.844 | 2.81 | 0.776 | 9.71 |

Exact `spectral` is capped at 20k.
