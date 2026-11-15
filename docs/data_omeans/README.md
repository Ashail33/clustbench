# o-means check (round 25)

o-means (owlgebra.ai/blog/o-means) is a GPU mini-batch k-means for large k. It combines:

* e-means' moving-average centroid updates, with centroid revival;
* a two-level assignment: each point is compared with cell centres first, then only with the centroids in its
  nearest cells.

The published code is a CUDA package. `src/clustbench/algorithms/o_means.py` re-implements the described method on
CPU so it can run in the harness. The post does not publish e-means' fitted batch-size and step-size laws, so the
defaults here come from the sweeps below. In those sweeps:

* The number of updates per epoch matters most. The default batch is about n / 200, clipped to 256-4,096.
* The step size must shrink with the batch, because every centroid decays every batch, hit or not:
  `alpha_max = min(0.5, 0.5 b / k)`.
* Generous revival helps. A centroid is split from a busy one when its count falls below 1% of the mean, and
  splitting keeps refining dense regions.

A first version used a fixed step size of 0.5 with a 0.1% revival threshold. At batch 256 and k = 4,096 that version
flagged 41% of healthy centroids as dead at every check, ending +10.5% above Lloyd after 3 epochs. Reviving only
centroids with no hits over a long window was also worse: +3.1% at batch 1,024.

## 1. Vector quantisation at large k (`scripts/omeans_quantization.py`)

The data are a Gaussian mixture in d = 64. Error is relative to 25-iteration Lloyd k-means. "Passes" means distance
computations divided by n k.

| k (n) | method, batch | 1 epoch: error / passes | 3 epochs: error / passes |
|---|---|---|---|
| 4,096 (200k) | MiniBatchKMeans | | +7.46% |
| 4,096 (200k) | e-means, batch 256 | +7.49% / 1.00 | **+0.19%** / 3.00 |
| 4,096 (200k) | o-means, batch 256 | +7.76% / 0.52 | **+0.21%** / 2.53 |
| 4,096 (200k) | e-means, batch 1,024 | +6.84% / 1.00 | +0.22% / 3.00 |
| 4,096 (200k) | e-means, batch 4,096 | +5.58% / 1.00 | +0.78% / 3.00 |
| 8,192 (500k) | MiniBatchKMeans | | +7.40% |
| 8,192 (500k) | e-means, batch 1,024 | +4.85% / 1.00 | **+0.20%** / 3.00 |
| 8,192 (500k) | o-means, batch 1,024 | +6.05% / 0.27 | **+0.81%** / 1.42 |
| 8,192 (500k) | e-means, batch 8,192 | +4.72% / 1.00 | +1.01% / 3.00 |

The 500k-point run has 61 points per centroid, the ratio of the post's datasets.

**Reproduced:**

* e-means reaches 25-iteration Lloyd error to within 0.2% in 3 passes, about 8x fewer distance computations. This is
  in line with the "~10x faster than Lloyd" of the e-means paper.
* o-means gets within 0.8% in 1.4 passes, about 18x fewer.
* The two-level assignment needs 27% of exact comparisons at k = 8,192 and 14% at k = 16,384 (earlier run). With the
  post's defaults it is off at k <= 2,048.

**Not reproduced:** the post's "Lloyd's 25-iteration error in 0.15-0.45 passes". After one epoch (1 pass for e-means,
0.27-0.52 for o-means) both are still 4.7-7.8% above Lloyd.

## 2. The toy chart (`scripts/omeans_toy_demo.py`)

This reproduces the LinkedIn chart: 12,000 2-D points, k = 64, mean of 3 runs.

| method | error by passes used |
|---|---|
| Lloyd | 0.938 (1), 0.798 (3), 0.753 (6), 0.724 (25) |
| e-means | 0.770 (1), 0.726 (2), **0.719 (3)** |
| o-means, 16 cells, probe 1 (~20 checks/point) | 0.830 (0.32), 0.768 (0.65), 0.781 (0.98) |

* e-means matches the chart. After 1 pass it is where Lloyd is after about 4 passes. After 3 passes it equals Lloyd
  after 25.
* o-means is not ahead of e-means at equal compute at k = 64. With 4 centroids per cell, boundary points often miss
  their true nearest centroid. The post's own defaults switch the cells off at this k.

## 3. Shapes benchmark (`configs/benchmark.omeans.yaml`, 60 tasks x 3 seeds, 1,620 runs, none hit the limits)

| algorithm | mean ARI | median time (s) |
|---|---|---|
| `mcg_omeans` (MCG with o-means micro-clusters) | **0.688** | **0.03** |
| `lmm_mcg_omeans` | 0.681 | 0.18 |
| `lmm_mcg` | 0.680 | 0.19 |
| `mcg_cf_omeans` | 0.676 | 0.17 |
| `mcg_cf` | 0.674 | 0.18 |
| `mcg` | 0.667 | 0.17 |
| `kmeans` | 0.508 | 0.03 |
| `o_means` | 0.497 | 0.01 |
| `minibatch_kmeans` | 0.493 | 0.03 |

Paired comparisons over the 180 task-seed runs (Wilcoxon, uncorrected, 5 comparisons):

* **`o_means` vs `kmeans`:** -0.011 (28 better / 37 worse, p = 0.19). It ties MiniBatchKMeans (+0.005).
* **`mcg_omeans` vs `mcg`:** +0.021 (66 / 43, p = 0.046; not significant after Holm over the 5 comparisons), and
  6x faster.
  * Mixtures improve: 0.673 -> 0.773.
  * Non-convex shapes are level: 0.844 vs 0.842, family means.
* **`mcg_cf_omeans` and `lmm_mcg_omeans`:** tie their originals (+0.002, +0.001). In the tiered variants, o-means
  only builds the ~60 coarse centres.

At this scale (at most 1,000 micro-clusters), the two-level assignment never activates. These runs therefore test
o-means' update rule, not its cell index.
