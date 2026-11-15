# o-means check (round 25)

o-means (owlgebra.ai/blog/o-means) is a GPU mini-batch k-means for large k: e-means' moving-average centroid
updates with centroid revival, plus a two-level assignment that compares each point with cell centres first and then
only with the centroids in its nearest cells. The published code is a CUDA package; `src/clustbench/algorithms/o_means.py`
re-implements the described method on CPU so it can run in the harness.

## 1. Vector quantisation at large k (`scripts/omeans_quantization.py`)

n = 200,000, d = 64, a Gaussian mixture with 65,536 modes; error relative to 25-iteration Lloyd k-means; "passes" =
distance computations / (n k).

| k | method | error vs Lloyd | passes |
|---|---|---|---|
| 4,096 | MiniBatchKMeans, 3 epochs | +6.72% | |
| 4,096 | e-means (exact assignment), 3 / 6 epochs | +1.57% / +0.89% | 3.00 / 6.00 |
| 4,096 | o-means, 3 / 6 epochs | +1.59% / +0.86% | 2.38 / 5.22 |
| 16,384 | MiniBatchKMeans, 3 epochs | +9.25% | |
| 16,384 | e-means (exact assignment), 3 / 6 epochs | +6.06% / +5.29% | 3.00 / 6.00 |
| 16,384 | o-means, 1 / 3 / 6 epochs | +11.05% / +6.93% / +5.83% | 0.14 / 0.73 / 1.69 |

* **Reproduced:** the two-level assignment needs 14% of exact comparisons per epoch at k = 16,384 (27% at 8,192,
  54% at 4,096; with the post's defaults it does nothing at k <= 2,048), for +0.2 to +1.3 points of extra error.
* **Not reproduced:** "Lloyd's 25-iteration error in 0.15-0.45 passes". Here o-means is still 0.9-5.8% above Lloyd
  after 6 epochs. A sweep of batch size (2,048-32,768) and peak step size (0.05-0.9) at k = 4,096 gives at best
  +5.1% after 1 epoch and +1.6% after 3. The post does not publish e-means' fitted batch/step-size laws, so the gap
  may be the settings, the data (the post uses real embedding sets) or unstated details; it cannot be settled
  without the authors' code.

## 2. Shapes benchmark (`configs/benchmark.omeans.yaml`, 60 tasks x 3 seeds, 1,620 runs, none hit the limits)

| algorithm | mean ARI | median time (s) |
|---|---|---|
| `mcg_omeans` (MCG with o-means micro-clusters) | **0.684** | **0.02** |
| `lmm_mcg` | 0.680 | 0.19 |
| `mcg_cf` | 0.674 | 0.18 |
| `mcg` | 0.667 | 0.16 |
| `lmm_mcg_omeans` | 0.666 | 0.17 |
| `mcg_cf_omeans` | 0.661 | 0.16 |
| `kmeans` | 0.508 | 0.03 |
| `minibatch_kmeans` | 0.493 | 0.03 |
| `o_means` | 0.478 | 0.01 |

Paired over the 180 task-seed runs (Wilcoxon, uncorrected):

* `o_means` vs `kmeans`: -0.030 (27 better / 63 worse, p = 1e-5). As a clusterer it is a slightly worse k-means.
* `mcg_omeans` vs `mcg`: +0.017 (62 / 49, p = 0.14), 8x faster. Mixtures 0.673 -> 0.786, non-convex shapes
  0.863 -> 0.808. The overall gain is not significant; the speed-up is.
* `mcg_cf_omeans` vs `mcg_cf`: -0.013 (p = 0.08); `lmm_mcg_omeans` vs `lmm_mcg`: -0.014 (p = 0.02). In the
  tiered variants o-means only builds the ~60 coarse centres, and replacing MiniBatchKMeans there hurts slightly.

At the harness's scale (at most 1,000 micro-clusters) the two-level assignment never activates, so these runs test
o-means' update rule (moving averages and revival), not its cell index.
