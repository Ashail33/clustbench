# An empirical-iteration methodology for clustering algorithm synthesis

This document distills the pattern that emerged from building clustbench
into a transferable methodology. The pattern was *not* designed up
front — it was discovered iteratively by running every experiment, and
the empirical evidence for each step is in
[`docs/ALGORITHM_ANALYSIS.md`](ALGORITHM_ANALYSIS.md).

## The core observation

Most clustering algorithms have one or two specific bottlenecks that
dominate their failure modes — outlier sensitivity, non-convex-shape
failure, quadratic scaling, hyperparameter brittleness, local-minimum
search, or wrong data assumption. Across the 32 algorithms in the
registry, the bottlenecks recur far more than they vary; every
"family" of failure mode has a small menu of known fixes.

This means **algorithm synthesis can proceed by mechanically composing
known fixes**, and the result can be benchmarked, the failure modes
identified, and another iteration spawned. The methodology below is
the explicit four-stage loop that emerges.

## The loop

```
       ┌──────────────────────────────────────────────────┐
       │                                                  │
   v1: synthesise from primitives that won on each axis   │
       │                                                  │
       ▼                                                  │
   benchmark v1 ──► identify the specific new failure mode │
       │                                                  │
       ▼                                                  │
   v2: targeted fix for that failure mode                 │
       │                                                  │
       ▼                                                  │
   benchmark v2 ──► observe what v2 lost vs v1            │
       │                                                  │
       ▼                                                  │
   v3: meta-of-meta — dispatch between v1 and v2 by       │
       data signature (the regimes where each won)        │
       │                                                  │
       ▼                                                  │
   benchmark v3 ──► all three versions in the table       │
       │                                                  │
       ▼                                                  │
   v4 (learned): replace the hand-coded dispatch with     │
       a kNN over fingerprints, trained on past results   │
       │                                                  │
       └──────────────────────────────────────────────────┘
```

## Stage 1 — synthesis from empirical primitives

The first version of any new algorithm should be a *deliberate
composition of the mechanisms that have already won* on the axes that
matter. The empirical lookup table:

| dimension | winning mechanism | reference algorithm |
|---|---|---|
| Convex blobs | k-means EM | `kmeans` |
| Non-convex shape | graph Laplacian embedding + k-means | `spectral`, `lmm` |
| Outlier robustness | posterior-weighted updates | `gmm` |
| Sub-quadratic scaling at quality | single-pass tree + global step | `birch_algo` |
| Hyperparameter robustness | adaptive estimation from data | `dbscan_auto` (Ester k-distance knee) |
| Cross-regime robustness | diverse ensembles | `pwcc_diverse` |

A v1 synthesis combines one mechanism per axis the algorithm intends to
cover. In clustbench, three v1s were built:

- `aura`: Nyström-Laplacian embedding + posterior-weighted GMM.
- `meta_clusterer`: four data fingerprints + hand-coded dispatch rules.
- `rapid`: density partition (Stage 1) + per-region routing (Stage 2)
  + 1-NN noise reassignment (Stage 3).

## Stage 2 — benchmark, then identify the *specific* new failure mode

The benchmark sweep produces:

- Per-algorithm ARI / NMI / silhouette / Davies-Bouldin / Dunn
- Per-algorithm wall-time / RSS / CPU
- Per-shape (mdcgen / anisotropic / moons / circles) ARI
- Per-outlier-injection ARI
- Friedman χ² across all algorithms

The critical step is not "v1 works" or "v1 doesn't work" but
**identifying the specific narrow failure mode**, e.g.:

- "AURA v1: ARI 0.00 on circles. Diagnosis: GMM's full covariance
  absorbs both rings into one elongated Gaussian."
- "META v1: ARI 0.16 on circles. Diagnosis: the eigengap rule never
  fires because the normalised Laplacian gap is *large* at k for
  clean circles."
- "RAPID v1: ARI delta -0.145 with outliers. Diagnosis: the k-distance
  knee in Stage 1 is contaminated by the same outliers it should
  ignore."

Each diagnosis points at exactly one mechanism to swap out in v2.

## Stage 3 — v2 with a targeted fix, then observe what it lost

v2 changes *only* the mechanism named in the v1 diagnosis. In
clustbench:

- AURA v2: z-score the embedding before k-means (the eigenvectors had
  inconsistent scale and full-covariance GMM mis-shaped them).
- META v2: replace eigengap with `convexity_ratio = clip(1.2 - 2·CV, 0, 1)`
  of intra-cluster distances + a knn_modularity check + a
  silhouette-probe fallback.
- RAPID v2: add a Stage 0 LOF outlier prefilter before Stage 1's
  density partition.

The pattern that *every* v2 in clustbench exhibited: **the targeted
fix worked, and v2 traded the v1 failure mode for a new one.**

| algorithm | what v2 fixed | what v2 broke |
|---|---|---|
| AURA | circles 0.00 → 1.00 | moons 0.37 → 0.01 (z-score amplified noise in low-variance eigen-directions) |
| META | circles 0.16 → 1.00 | nothing — clean win |
| RAPID | outlier delta -0.145 → -0.050 | shape regressed (LOF removed real cluster points when density ≈ outlier density) |

META v2 was the exception — when the fix is a strict replacement of a
brittle rule, no regression occurs. AURA v2 and RAPID v2 are the
common case: every "fix the bottleneck" intervention opens a new
failure mode at roughly the same rate.

## Stage 4 — v3 meta-of-meta dispatch between v1 and v2

v3 acknowledges that v1 and v2 each won on a different regime, and
*dispatches between them by a cheap data signature*.

- AURA v3: dispatch by `effective_rank` of the embedding's column
  std-devs. `>=2` → z-scored k-means (v2 path); `<2` → raw GMM (v1
  path). Works because moons has effective rank 1 (one informative
  eigenvector) and circles has effective rank 2 (two informative
  eigenvectors).
- META v3: run both v1's and v2's routers; if they agree, route once;
  if they disagree, fit both candidates on a 20% subsample and pick
  the silhouette winner.
- RAPID v3: gate stage 0 on an outlier-fraction estimate. If LOF
  outlier_frac > 5% → v2 path (with prefilter); else → v1 path.

The pattern observed:

| algorithm | v1 vs v2 | v3 outcome |
|---|---|---|
| AURA | complementary (different shape regimes) | **+0.14 ARI over v1, rank 9 → 2** |
| META | v2 strict-better than v1 everywhere | v3 slightly worse than v2 (probe overrides correct v2 routes) |
| RAPID | v1 strict-better than v2 everywhere | v3 ≈ v1 (dispatch always picks v1) |

**The pattern is: v3 wins when v1 and v2 are complementary on
identifiable regimes, is neutral when one dominates everywhere, and
slightly loses when the dispatch criterion is noisier than v2.**

## Stage 5 — v4 learned router

Replace the hand-coded dispatch with a small classifier trained on the
benchmark data. clustbench's `learned_router` is a 7-feature kNN over
data fingerprints (log_n, d, k, effective_dim, conv_cv, outlier_frac,
density_skew), trained on every `(task, ARI)` pair in
`docs/data/results.json`.

Honest evaluation: `exclude_self=True` drops the nearest training
point if its fingerprint matches the inference fingerprint exactly
(leave-one-out at the seed level). Near-neighbour leakage from
adjacent seeds is not eliminated — for a held-out generalisation
number, use [`configs/benchmark.holdout.yaml`](../configs/benchmark.holdout.yaml).

**Empirical result on clustbench**: rank 3/32 with the highest
absolute mean ARI in the registry (0.884), narrowly beating both
`lmm` and `aura_v3`. The dispatch distribution across 16 tasks was:

- `parallel_kmeans` (n=8, ARI 0.998) — convex blobs.
- `spectral` (n=4, ARI 0.812) — non-convex.
- `lmm` (n=3, ARI 0.742) — borderline non-convex.
- `gmm` (n=1, ARI 0.689) — outlier-heavy.

The learned router rediscovered the "What to try first" decision tree
from `ALGORITHM_ANALYSIS.md` without being shown it.

## What the pattern *isn't*

Three honest caveats:

1. **v2 doesn't always work.** AURA v2 and RAPID v2 both regressed
   overall before v3 fixed them. The cycle requires the discipline of
   reading the benchmark output carefully and naming the new failure
   mode precisely.
2. **v3 doesn't always help.** META v3 made META slightly worse. The
   meta-dispatch only pays off when v1 and v2 are complementary in a
   way the cheap dispatch signature can detect.
3. **The learned router has leakage.** Its top spot is partly
   "memorising the dataset family" rather than "discovering the right
   algorithm for unseen data." The held-out benchmark
   (`benchmark.holdout.yaml`) is the only honest evaluation.

## When to stop the loop

The loop terminates when one of three conditions holds:

- **Quality plateau.** The latest version's mean ARI is within noise
  of the previous version *and* no clean failure mode is identifiable
  in the per-shape / per-outlier breakdown.
- **Routing equilibrium.** The learned router (v4) selects the same
  algorithm > 90% of the time, indicating that the regime
  partitioning has converged and adding more algorithms doesn't
  enlarge the achievable frontier.
- **The benchmark hits its limits.** When the dataset grid no longer
  has a regime where the loop's iteration can demonstrate a lift,
  the next move is to *enlarge the benchmark*, not the synthesis chain.

## Transferring the pattern

The methodology generalises beyond clustering. The structural
requirements are:

1. **A registry of algorithms** that share a common interface
   (`fit_predict` here).
2. **A benchmark harness** that runs the registry across a parametric
   grid and writes a tidy `(task, algorithm, metric)` table.
3. **An "analysis document"** that names the failure mode of each
   algorithm in plain language, mapping it to a known fix.
4. **A trajectory layer** (or any per-step state log) so the next
   iteration can capture data the algorithm interacts with, not just
   the algorithm's final output.

Given those four ingredients, the v1 → v4 loop is mechanical. The
discipline is in writing the diagnoses honestly enough that v2 is
actually targeted.
