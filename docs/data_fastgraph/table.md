| n | algorithm | mean ARI | median wall time (s) | runs |
|---|---|---|---|---|
| 20,000 | `lmm` | 0.846 | 2.2 | 22 |
| 20,000 | `lmm_fast` | 0.842 | 1.3 | 22 |
| 20,000 | `spectral` | 0.853 | 7.2 | 21 |
| 20,000 | `spectral_fast` | 0.844 | 2.8 | 22 |
| 20,000 | `gmm` | 0.537 | 0.4 | 22 |
| 20,000 | `trident` | 0.556 | 1.7 | 22 |
| 20,000 | `minibatch_kmeans` | 0.554 | 0.1 | 22 |
| 100,000 | `lmm` | 0.837 | 40.1 | 22 |
| 100,000 | `lmm_fast` | 0.836 | 10.3 | 22 |
| 100,000 | `spectral` | capped (not run above 20k) | | 0 |
| 100,000 | `spectral_fast` | 0.776 | 9.7 | 22 |
| 100,000 | `gmm` | 0.557 | 1.3 | 22 |
| 100,000 | `trident` | 0.570 | 2.5 | 22 |
| 100,000 | `minibatch_kmeans` | 0.555 | 0.7 | 22 |

Per family, mean ARI over two seeds:

| family | lmm 20k | lmm_fast 20k | spectral 20k | spectral_fast 20k | lmm 100k | lmm_fast 100k | spectral_fast 100k |
|---|---|---|---|---|---|---|---|
| VariableDensityBridges | 0.89 | 0.89 | 0.89 | 0.89 | 0.89 | 0.89 | 0.89 |
| anisotropic | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| circles | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| heavy_tailed_mixture | 0.88 | 0.88 | 0.88 | 0.88 | 0.88 | 0.88 | 0.88 |
| hierarchical_nested | 0.63 | 0.63 | 0.62 | 0.62 | 0.63 | 0.63 | 0.62 |
| inverse_pca | 0.98 | 0.98 | 0.98 | 0.98 | 0.98 | 0.98 | 0.98 |
| mdcgen | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| mixed_shapes | 0.86 | 0.86 | 0.86 | 0.86 | 0.86 | 0.86 | 0.71 |
| moons | 0.99 | 0.99 | 0.99 | 0.99 | 0.99 | 0.99 | 1.00 |
| rings | 0.89 | 0.85 | 1.00 | 0.88 | 0.79 | 0.79 | 0.29 |
| spiral | 0.17 | 0.17 | 0.17 | 0.17 | 0.17 | 0.17 | 0.17 |
