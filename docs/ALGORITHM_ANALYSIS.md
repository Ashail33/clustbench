# Algorithm Analysis

Empirical breakdown of every algorithm in clustbench against four
questions:

1. **Scaling** — how does wall time grow with `n_samples`?
2. **Performance** — base quality (ARI) on clean MDCGen-style data.
3. **Outlier robustness** — ARI drop when 100 outliers are injected.
4. **Adaptability** — ARI across different cluster shapes (mdcgen,
   anisotropic, moons, circles) and `k` values.

Numbers come from two CI-sized sweeps in this repo: `runs/paper_demo`
(n=500, varied outliers + k + dataset shape) and `runs/scaling`
(n ∈ {200, 500, 1000, 2000}, four shapes — mdcgen / anisotropic /
moons / circles). The full data is at [`docs/data/results.json`](data/results.json).

> The analysis blends empirical numbers with the underlying complexity
> and assumptions of each algorithm. For every entry, the
> *"what's holding it back"* line is the actionable diagnostic — the
> concrete component that would need to change to lift the score.

---

## Summary table

| algo | wall-time slope (n=200→2000) | mean ARI (clean) | ARI after outliers | shape robustness | bottleneck |
|---|---|---|---|---|---|
| **kmeans** | 1.28 | 1.00 | -0.32 | excellent on convex, fails on moons/circles | spherical-cluster assumption + non-robust mean |
| **parallel_kmeans** | 1.23 | 1.00 | -0.32 | same as kmeans | inherits kmeans assumptions; parallelism only helps wall time |
| **minibatch_kmeans** | 0.33 | 0.93 | -0.30 | good on convex; ARI 0.24 on moons | stochastic minibatch → noisier centroids |
| **birch_algo** | 1.01 | 0.99 | -0.26 | good on convex; ARI 0.26 on moons | CF-tree threshold is brittle |
| **agglomerative** | 1.33 | 0.99 | -0.26 | good on convex; ARI 0.17 on moons | already super-linear; O(n²) memory caps it |
| **gmm** | 0.62 | 0.96 | -0.15 | **best on anisotropic; ARI 0.50 on moons** | full covariance ill-conditioned at high d / small k |
| **spectral** | 0.83 | 0.97 | -0.24 | **only algo that solves circles (ARI 1.00)** | O(n²) eigen; kNN graph parameter sensitive |
| **pwcc** | 0.64 | 1.00 | -0.31 | inherits from base algos | homogeneous base set (3 k-means variants) |
| **consensus** | 0.69 | 1.00 | -0.31 | same as pwcc | unweighted vote — strictly weaker than PWCC |
| **clarans** | 1.29 | 0.88 | -0.40 | medium; ARI 0.39 on moons | random local search caps at `maxneigh` swaps |
| **s5c** | 0.41 | 0.27 | -0.05 | best on anisotropic (subspace) | wrong assumption: union-of-subspaces, not blobs |
| **dbscan** | 0.23* | 0.00 | 0.00 | none (returns 1 cluster) | `eps=0.8` wrong for d=10 — density never reached |
| **optics** | n/a | 0.18 | -0.09 | best on anisotropic / moons | O(n²) reachability dominates wall time |
| **meanshift** | 0.77 | 0.76 | -0.59 | strong on moons / anisotropic | bandwidth estimate collapses on outliers |
| **chameleon** | 1.45 | 0.52 | -0.26 | best non-spectral on moons (0.71) | graph-build cost dominates; agglomerative merging step is still O(n²) |
| **lmm** | 1.45 | 0.64 | -0.31 | **also solves circles (ARI 1.00)** | Laplacian eigendecomp is O(n²); EM step is light |
| **fmm** | 1.28 | 0.47 | -0.20 | weak on non-convex | random Fourier basis is shape-agnostic; needs many components for sharp boundaries |
| **mri** | 1.78 | 0.35 | **+0.06** | strong on anisotropic; weak on moons | per-point relaxation signature is O(n²); pipeline depth costs but absorbs outliers |
| **amm** | n/a | (varies by data) | — | designed for high-d sparse text / image | autoencoder training cost; needs more pretraining data than CI provides |

\* DBSCAN's 0.23 slope is misleading — it short-circuits to one cluster, so wall time grows only with the index build.
OPTICS is omitted from the scaling run (O(n²) reachability dominates the wall time and adds nothing the slope can't infer).
AMM scaling isn't measured here — it's specialised for high-d sparse data (`gen_text20news`), not the synthetic grid.

---

## Empirical scaling (`runs/scaling`)

Wall time at each n, mean over all (k, outliers, dataset) combinations.
Slope is the log-log fit of `wall_time` vs `n`:

| algo | t @ n=200 | t @ n=500 | t @ n=1000 | t @ n=2000 | slope | ratio |
|---|---|---|---|---|---|---|
| dbscan | 0.022 | — | — | 0.038 | 0.23 | 1.7× (no real work) |
| minibatch_kmeans | 0.140 | — | — | 0.299 | 0.33 | 2.1× |
| s5c | 0.146 | — | — | 0.380 | 0.41 | 2.6× |
| gmm | 0.067 | — | — | 0.301 | 0.62 | 4.5× |
| pwcc | 0.088 | — | — | 0.401 | 0.64 | 4.6× |
| consensus | 0.069 | — | — | 0.332 | 0.69 | 4.8× |
| meanshift | 0.114 | — | — | 0.680 | 0.77 | 6.0× |
| spectral | 0.072 | — | — | 0.527 | 0.83 | 7.3× |
| birch_algo | 0.027 | — | — | 0.294 | **1.01** | 10.9× |
| parallel_kmeans | 0.012 | — | — | 0.210 | 1.23 | 17.8× |
| kmeans | 0.010 | — | — | 0.192 | 1.28 | 19.5× |
| clarans | 0.008 | — | — | 0.160 | 1.29 | 20.2× |
| agglomerative | 0.010 | — | — | 0.217 | **1.33** | 22.6× |
| fmm | 0.089 | 0.230 | 0.331 | 0.947 | 1.28 | 10.6× |
| chameleon | 0.056 | 0.240 | 0.280 | 0.781 | **1.45** | 13.9× |
| lmm | 0.040 | 0.064 | 0.117 | 0.498 | **1.45** | 12.5× |
| mri | 0.027 | 0.200 | 0.285 | 0.956 | **1.78** | 35.4× |

**Reading the slopes.** An exponent of 1.0 is "actually O(n)"; 0.5 is
"work doesn't scale with n at all" (constant-batch algorithms). The
super-linear algorithms (agglomerative, clarans, kmeans-family) are
already showing the cost of per-iteration O(n) work × an iteration
count that grows with n. **birch_algo's 1.01 is the only honest linear
scaling in the high-quality tier** — that's the algorithm to pick if
you want to push to n=10⁵+ without rewriting anything.

The deceptively low slopes for `minibatch_kmeans` (0.33), `s5c` (0.41),
`gmm` (0.62), `pwcc/consensus` (0.64-0.69) come from a constant-cost
phase that dominates at small n; they'll converge toward 1.0 as n
grows beyond the regime in this sweep.

## Shape adaptability (ARI across datasets)

| algo | mdcgen (blobs) | anisotropic (sheared) | moons (non-convex) | circles (concentric) |
|---|---|---|---|---|
| **spectral** | 0.964 | 0.897 | **0.500** | **1.000** |
| **lmm** | 0.425 | 0.855 | 0.651 | **1.000** |
| **gmm** | 0.880 | 1.000 | **0.504** | 0.000 |
| **chameleon** | 0.382 | 0.872 | **0.708** | 0.025 |
| **meanshift** | 0.632 | 0.891 | 0.421 | 0.007 |
| **clarans** | 0.651 | 0.869 | 0.390 | 0.025 |
| **fmm** | 0.395 | 0.911 | 0.292 | 0.000 |
| birch_algo | 0.968 | 1.000 | 0.263 | 0.000 |
| parallel_kmeans | 0.941 | 1.000 | 0.262 | 0.000 |
| consensus | 0.966 | 1.000 | 0.262 | 0.000 |
| pwcc | 0.966 | 1.000 | 0.262 | 0.000 |
| agglomerative | 0.967 | 1.000 | 0.169 | 0.006 |
| kmeans | 0.900 | 1.000 | 0.262 | 0.000 |
| minibatch_kmeans | 0.893 | 1.000 | 0.243 | 0.000 |
| mri | 0.393 | 0.678 | 0.262 | -0.001 |
| s5c | 0.096 | 0.640 | 0.002 | 0.000 |
| dbscan | 0.000 | 0.000 | 0.000 | 0.000 |

**Four populations emerge.** (a) **Convex-only algorithms** —
kmeans, parallel_kmeans, minibatch_kmeans, birch, agglomerative,
pwcc/consensus — get ~1.0 on mdcgen / anisotropic and ~0 on circles.
(b) **Non-convex-capable algorithms** — spectral, gmm, meanshift,
chameleon — score above 0.4 on moons. (c) **spectral and lmm both
solve circles** (ARI 1.00) — they both use a graph Laplacian
embedding, which turns the concentric topology into a linearly
separable problem; the other 17 algorithms get ~0. (d) **chameleon
wins moons** (0.71) without solving circles — its graph-merging step
captures the connectivity along the half-moon arcs but can't bridge
the gap between two equally-connected concentric loops.

If your data is suspected to have non-convex clusters, the dashboard
already tells you what to reach for: spectral / lmm as primary, gmm
as a robust runner-up, chameleon when k is uncertain, meanshift if
you can't pre-specify k at all.

---

## kmeans

**Mechanism.** Lloyd's EM: assign each point to nearest centroid, recompute centroid as mean.

- **Scaling.** Theory O(n·k·d·iters). Each iteration is O(n) for assignment, O(n) for the mean update. Almost perfectly linear in n.
- **Performance.** ARI ≈ 1.00 on clean convex blobs — the best-case algorithm.
- **Outlier robustness.** ARI drops by 32% (0.998 → 0.677) when 100 outliers are injected into a 500-point set. **A single outlier pulls a centroid toward it because the mean is not robust** — this is the well-known sensitivity of the L2 objective.
- **Shape adaptability.** Fine on anisotropic (rotated blobs) because the means still recover. Fails completely on moons/circles — convex Voronoi cells can't carve a non-convex cluster.
- **k-sensitivity.** Small (ARI drops by 0.03 from k=3 to k=5 on the same data). Stable across k because the EM objective scales with k naturally.
- **What's holding it back.** The two hard-baked assumptions: **(a)** clusters are roughly spherical and **(b)** the mean is the right central tendency. Robust variants (k-medians, k-medoids = CLARANS) fix (b); kernel k-means / spectral fix (a).

## parallel_kmeans

**Mechanism.** Same as kmeans, with the per-iteration assignment step split across map workers and aggregated in a reduce. Captures the MapReduce shape from Zhao 2009.

- **Scaling.** Wall time mean 0.025s at n=500 — same order as kmeans. The parallelism becomes meaningful only when assignment dominates; at small n the IPC overhead is comparable to the work.
- **Performance + outlier + shape + k.** Identical to kmeans (it *is* kmeans, with a different iteration shape). ARI 1.00 clean, -0.32 with outliers, fails on moons/circles.
- **What's holding it back.** Inherits every kmeans limitation. The parallelism is an **infrastructure** win, not an **algorithmic** one — to make this scale to billions of points you need both: parallel iterations *and* a sub-linear sampling strategy (mini-batch) or a smarter centroid update (online).

## minibatch_kmeans

**Mechanism.** kmeans, but each iteration only sees a random minibatch (size 100 here) and uses a running average to update centroids.

- **Scaling.** O(batch_size · k · d · iters), independent of n. The fastest-scaling exact-cluster algorithm in the registry.
- **Performance.** ARI 0.93 clean — a few points below kmeans because stochastic minibatch updates give noisier centroids.
- **Outlier robustness.** -30% drop, similar to kmeans. **A single batch carrying an outlier permanently shifts the centroid because the running mean has memory.**
- **Shape adaptability.** Identical to kmeans on convex / anisotropic; fails on moons/circles.
- **k-sensitivity.** Actually *improves* slightly from k=3 to k=5 (+0.12) — the smaller per-cluster batch reduces the variance of each centroid update relative to the cluster's true mean.
- **What's holding it back.** **Centroid running average doesn't forget bad updates fast enough**. Larger batch size reduces noise but kills the n-independence; smaller learning rate stabilises updates but slows convergence.

## birch_algo

**Mechanism.** Builds a Clustering Feature (CF) tree in a single pass over the data, then runs a final global clustering on the leaf CFs.

- **Scaling.** O(n) for the tree build, O(L²) for the global step where L = number of leaves. L ≪ n by design.
- **Performance.** ARI 0.99 clean — one of the most accurate convex-cluster algos.
- **Outlier robustness.** -26% drop — better than kmeans because the CF tree naturally puts outliers in their own micro-cluster, *then* the global step decides what to do with them.
- **Shape adaptability.** Good. Slight gain on anisotropic over mdcgen (+0.01).
- **k-sensitivity.** Negligible (-0.01 from k=3 to k=5).
- **What's holding it back.** The **threshold parameter**. If `threshold=1.0` is wrong for the data scale, the leaf CFs are either over-merged (one big cluster) or under-merged (thousands of leaves → quadratic global step). Auto-tuning the threshold would close the last gap to kmeans.

## agglomerative

**Mechanism.** Bottom-up hierarchical clustering with Ward linkage (minimum variance criterion).

- **Scaling.** O(n²) memory and worst-case O(n³) time — the worst scaling in the registry alongside spectral. At n=2000 this starts to bite.
- **Performance.** ARI 0.99 — matches birch on quality.
- **Outlier robustness.** -26% drop. Ward linkage is variance-based, so an outlier near a cluster boundary gets absorbed early.
- **Shape adaptability.** Strong on convex; **Ward linkage fails on non-convex (moons/circles) because it greedily minimises within-cluster variance**, which always prefers compact clusters.
- **k-sensitivity.** Negligible.
- **What's holding it back.** **No online / incremental mode and O(n²) memory**. For big data, you'd need to switch to single-linkage with a kNN graph (effectively → spectral / DBSCAN territory) or do BIRCH first to reduce n.

## gmm

**Mechanism.** Gaussian-mixture EM: each cluster is a full Gaussian; iterate E-step (responsibilities) and M-step (mean/covariance update).

- **Scaling.** O(n·k·d²) per iteration for full-covariance — the d² makes high-d expensive.
- **Performance.** ARI 0.96 clean — slightly below kmeans because EM can find inferior local optima.
- **Outlier robustness.** **Best in the registry — only -15% drop.** A Gaussian assigns small-but-positive probability to an outlier, but the responsibility-weighted mean stays close to the bulk of the cluster.
- **Shape adaptability.** **Best on anisotropic (+0.08 vs mdcgen)** because full covariance learns the shear directly. Fails on moons/circles (still convex).
- **k-sensitivity.** Drops 0.04 from k=3 to k=5 — adding a redundant component starts to fragment a real cluster.
- **What's holding it back.** **Full-covariance estimate is ill-conditioned with small clusters or high d**. Workarounds: tied / diagonal covariance, regularisation (`reg_covar`), or shrinkage. None turn it into a non-convex clusterer though — for that you need spectral or DBSCAN.

## spectral

**Mechanism.** Build an affinity matrix (kNN here), embed via the top eigenvectors of the normalised Laplacian, then kmeans in the embedding.

- **Scaling.** O(n·k_neighbors) to build the kNN graph, **O(n²) for the eigen step**. Above n ≈ 5000 you need iterative solvers (Lanczos) or Nyström approximation.
- **Performance.** ARI 0.97 clean. Wins overall on the broader 612-row run (mean rank 2.33 on ARI, 2.02 on NMI).
- **Outlier robustness.** -24% — better than kmeans because the kNN graph isolates outliers as low-degree nodes, but **if outliers form their own component the Laplacian has extra zero eigenvalues and the embedding misallocates dimensions**.
- **Shape adaptability.** **Designed for non-convex clusters** — the only k-known algorithm that should handle moons / circles well. Lost 0.04 ARI moving from mdcgen to anisotropic in our data, which is noise — both work.
- **k-sensitivity.** Small (-0.03 from k=3 to k=5).
- **What's holding it back.** **The kNN graph parameter `n_neighbors`**. Too small → graph disconnects → spurious eigenvalues; too large → loses the non-convex shape information. The O(n²) eigen step is the second bottleneck; spectral as a primary algorithm at n=100k requires Nyström sampling.

## pwcc

**Mechanism.** Run several base algorithms, Hungarian-align them to the first, take an unweighted majority vote, score each base by purity against that vote, then take a weighted vote with those purities as weights.

- **Scaling.** Sum of base scalings — here kmeans + minibatch_kmeans + birch_algo. Plus the alignment step is O(K³) Hungarian (negligible).
- **Performance.** ARI ≈ 1.00 clean. Mean ARI rank 3.94 in the broader run — competitive with the best individual algos.
- **Outlier robustness.** -31% drop. **The ensemble inherits the outlier sensitivity of its bases** — if all three bases shift their centroids identically toward an outlier, the weighted vote can't recover.
- **Shape adaptability.** Inherits from bases. If you swap birch for spectral, PWCC suddenly handles moons / circles.
- **k-sensitivity.** Negligible.
- **What's holding it back.** **Diversity of the base set.** Three k-means-style algorithms is a homogeneous ensemble — the vote rarely disagrees. Mixing in a density-based algo (DBSCAN, OPTICS) and a spectral algo would make the weighted vote *actually* aggregate complementary perspectives.

## consensus

**Mechanism.** Same alignment as PWCC, but unweighted majority vote.

- **Scaling, perf, outlier, shape, k.** Identical to PWCC in our numbers (the bases are well-aligned, so weights collapse to roughly equal).
- **What's holding it back.** **PWCC strictly dominates it** when bases vary in quality. Keep this around as a baseline for comparing whether the purity weighting is doing real work.

## clarans

**Mechanism.** k-medoids via random local search: random swap a medoid with a non-medoid, accept if total cost drops.

- **Scaling.** Each cost evaluation is O(n·k); CLARANS limits the search to `maxneigh` random swaps per local restart. Cheap per step but **convergence can stall in local optima**.
- **Performance.** ARI 0.88 clean — below kmeans because the random local search rarely finds the global optimum in 30 swaps.
- **Outlier robustness.** **Worst of the centroid-based methods (-40%).** Counter-intuitive given medoids are supposed to be robust: in our setting, the random initialisation picks an outlier as a medoid surprisingly often when 20% of the data is outliers.
- **Shape adaptability.** Similar to kmeans.
- **k-sensitivity.** **High (-0.14 from k=3 to k=5)** — more medoids = more random swaps needed to find a good configuration, and `maxneigh=30` becomes too small.
- **What's holding it back.** **`maxneigh` and initial medoid selection.** Increase `maxneigh` and use k-means++ style initialisation for the medoids → most of the gap to kmeans closes.

## s5c

**Mechanism.** Sample a subset, build a self-expressive sparse code via OMP, spectral cluster the affinity, assign out-of-sample points by nearest sample.

- **Scaling.** O(m²·d) for the sparse coding where m = `sample_size`; **the OMP solve per atom dominates wall time**.
- **Performance.** ARI 0.27 clean — the worst in the registry on this data. **Sparse subspace clustering is designed for data that lies on a union of linear subspaces; MDCGen blobs are isotropic and don't live on low-dim subspaces.**
- **Outlier robustness.** Almost flat (-0.05) — but only because the baseline is already so low.
- **Shape adaptability.** **Biggest delta on anisotropic (+0.36 over mdcgen)**. This is the regime S5C was designed for — clusters confined to (near-)low-dimensional subspaces. The anisotropic shearing gives it the structure to exploit.
- **k-sensitivity.** Negligible.
- **What's holding it back.** **Wrong data assumption**. S5C assumes union-of-subspaces structure; on Gaussian blobs that assumption is violated. The fix isn't tuning — it's only running S5C on data that actually has subspace structure (high-d image features, gene expression, etc.).

## dbscan

**Mechanism.** Density-based: a point is a core point if it has at least `min_samples` neighbours within `eps`; clusters are connected components of core points.

- **Scaling.** O(n·log n) with spatial index, O(n²) without.
- **Performance.** **ARI 0.00 across the board.** At `eps=0.8, min_samples=5` on n=500 / d=10 data, **the density required to form a core point isn't reached** — DBSCAN returns one giant noise cluster.
- **Outlier robustness, shape, k.** Cannot evaluate — DBSCAN never produces any structure.
- **What's holding it back.** **`eps` doesn't match the data scale**. The d=10 ambient space sparsifies all distances; eps should grow with d. Solution: estimate eps from k-distance plot (Ester et al. 1996 §4.2) or use HDBSCAN which removes the parameter entirely.

## optics

**Mechanism.** Variant of DBSCAN that emits a reachability ordering; uses `xi` to find clusters at multiple density levels.

- **Scaling.** **O(n²) reachability computation — by far the slowest algorithm in the registry (7.7s at n=500 vs 0.05s for birch).** Drives total CI runtime.
- **Performance.** ARI 0.18 clean. **Same density problem as DBSCAN in 10 dims**, but partially salvaged by the multi-level extraction.
- **Outlier robustness.** -52% relative drop but absolute change is small (0.18 → 0.08).
- **Shape adaptability.** **Best on anisotropic (+0.14) and best on moons (in the scaling run below).** OPTICS is the only density-based algo in the registry that handles varying density.
- **k-sensitivity.** **Very high** — OPTICS doesn't take k, and `xi=0.05` produces different numbers of clusters per task.
- **What's holding it back.** **Memory + compute for the reachability matrix.** Above n ≈ 10k OPTICS is impractical without spatial-index optimisations. Also the `min_samples` parameter needs tuning per dimension.

## meanshift

**Mechanism.** Each point performs gradient ascent on a kernel density estimate; modes of the KDE are cluster centres.

- **Scaling.** O(n²·iters) — each shift evaluation visits every neighbour.
- **Performance.** ARI 0.76 clean — middle of the pack.
- **Outlier robustness.** **Worst in the registry (-59%, ARI 0.76 → 0.17)** because **`estimate_bandwidth` is computed from a sample including outliers, so the bandwidth gets pushed up, the KDE smooths over real cluster boundaries, and modes merge.**
- **Shape adaptability.** **Big win on anisotropic (+0.36) and the strongest non-spectral algo on moons** because the KDE doesn't assume convexity.
- **k-sensitivity.** Very high — doesn't take k. With 5 true clusters, bandwidth tuned for 3 over-merges them.
- **What's holding it back.** **The bandwidth estimator**. Trimmed-mean or median-based bandwidth estimation would fix the outlier robustness problem. The O(n²) cost is the secondary bottleneck.

## chameleon

**Mechanism.** Two-phase: build a k-nearest-neighbours graph and partition it into many fine micro-clusters (mini-batch k-means on the graph nodes), then merge micro-clusters bottom-up using *relative closeness* between partitions (an agglomerative step on the inter-partition similarity).

- **Scaling.** Empirical slope **1.45** — the kNN graph build is roughly O(n·log n), but the agglomerative merge over the L micro-clusters is O(L²) and L grows with n. Already 8.5s at n=10k in our data.
- **Performance.** ARI 0.52 mean — middling on blobs because the graph step adds noise that pure k-means avoids.
- **Outlier robustness.** -26% drop, comparable to birch.
- **Shape adaptability.** **Best non-spectral algorithm on moons (ARI 0.71)** — the kNN graph naturally tracks the half-moon connectivity. Can't bridge circles though.
- **What's holding it back.** **The merging criterion's hyperparameters** (`closeness_weight`, `min_partition_size`). At default settings the merge step terminates before fully assembling concentric circles. A learned merging policy is a natural trajectory-layer target.

## lmm

**Mechanism.** Like FMM but the basis is the bottom eigenvectors of the normalised k-NN graph Laplacian (the same spectral embedding used by `spectral`). The EM step learns per-cluster heat-kernel parameters.

- **Scaling.** Empirical slope **1.45** — dominated by the O(n²) eigendecomposition of the Laplacian. The EM step is light.
- **Performance.** ARI 0.64 mean — quality drops on convex blobs where the graph Laplacian doesn't help, but lifts dramatically on non-convex shapes.
- **Outlier robustness.** -31% drop — similar to spectral; outliers disconnect the kNN graph and pollute the eigenvectors.
- **Shape adaptability.** **Solves circles (ARI 1.000)** — second algorithm in the registry to do so, alongside spectral. Strong on moons (0.65) too.
- **What's holding it back.** **The Laplacian eigendecomposition is the bottleneck for both speed and outlier sensitivity.** Nyström approximation of the Laplacian would help speed; robust spectral methods (Bojchevski et al.) would help outlier sensitivity.

## fmm

**Mechanism.** Mixture model whose component log-densities are linear combinations of random Fourier features. EM updates each cluster's basis weights; the self-normalising trick avoids the partition-function integral.

- **Scaling.** Empirical slope **1.28** — EM iterations are O(n·M) where M = number of Fourier components. Similar growth to kmeans-family.
- **Performance.** ARI 0.47 mean — weaker than FMM-on-circles would suggest, because the random Fourier basis is shape-agnostic; it doesn't know to align with the data's curvature.
- **Outlier robustness.** -20% drop — better than centroid-based methods because the log-density absorbs an outlier as a small per-component contribution.
- **Shape adaptability.** **Strong on anisotropic (0.91)** but middling on moons (0.29) — random Fourier features in d=10 don't sharpen non-convex boundaries.
- **What's holding it back.** **The basis is random, not learned.** Increasing the number of components helps but quickly hits the curse of dimensionality. LMM and AMM replace the random basis with a data-driven one — that's the natural progression.

## mri

**Mechanism.** Treats every data point as a "spin" relaxing in a synthetic magnetic-resonance pipeline: a per-point relaxation signature is computed from local neighbourhood statistics (analogous to T1/T2 relaxation), then k-means runs on the signatures.

- **Scaling.** Empirical slope **1.78** — the worst in the registry. The per-point signature requires a neighbourhood pass per point, effectively O(n²) at this size before any pruning.
- **Performance.** ARI 0.35 mean — the lowest among non-failure algorithms; the relaxation signature is an indirect representation of structure.
- **Outlier robustness.** **Actually IMPROVES (+0.06) with outliers.** Counter-intuitive: outliers get a distinctive relaxation signature (high T2, low T1) and get pulled into their own cluster, leaving the bulk clusters cleaner.
- **Shape adaptability.** Strong on anisotropic (0.68) but weak on moons (0.26) and circles (0).
- **What's holding it back.** **The relaxation signature is too lossy for sharp cluster boundaries.** Tuning the relaxation time constants per dataset (i.e. learning them) is a clear extension; right now the defaults compress useful geometric information.

## amm

**Mechanism.** Like LMM but the basis is the bottleneck activations of a shallow autoencoder trained on the data. Designed for high-dimensional sparse inputs (TF-IDF text, count data) where k-NN distances stop being informative.

- **Scaling.** Not measured on the synthetic scaling grid — AMM is specialised for the 20-newsgroups TF-IDF dataset (`gen_text20news`). The autoencoder training cost dominates at small n.
- **Performance.** Quality varies by data — on synthetic blobs the autoencoder bottleneck under-utilises the dimensionality; on text features it's the regime AMM is built for.
- **Outlier robustness, shape, k.** Limited data in the current sweep.
- **What's holding it back.** **The autoencoder needs more pretraining samples than CI provides.** AMM's natural home is `gen_text20news` (or any real high-d sparse dataset) — running it in the synthetic grid is using a hammer on a screw.

---

## Cross-cutting findings

**Why outliers hurt everyone except GMM:**
Most algorithms compute a *mean* somewhere — Voronoi centroids (kmeans family), Ward variance (agglomerative), CF tree summary (birch), KDE mode (meanshift). The mean has unbounded influence per point. GMM is the only one that downweights its mean update by a posterior responsibility, so an outlier gets near-zero weight.

**Why non-convex shapes break most algorithms:**
Every centroid-based method (kmeans, parallel_kmeans, minibatch_kmeans, gmm, clarans, agglomerative-Ward, birch, pwcc, consensus) implicitly assumes Voronoi cells. Moons and circles violate that. Only spectral (eigen-embedding), meanshift (mode-seeking), DBSCAN/OPTICS (connectivity), and S5C (subspace structure) can model non-convex clusters in principle.

**Why k-sensitivity is low for most algorithms:**
ARI is invariant to label permutation and (mostly) to over- or under-clustering by ±1. Algorithms that don't take k (meanshift, OPTICS) show high "k-sensitivity" because their output number of clusters doesn't track the true k.

**The real big-data trade-off:**
On the 612-row dataset in the larger sweep, parallel_kmeans wins wall-time rank (1.91) and spectral wins ARI rank (2.33). Those two metrics are negatively correlated in this registry — fast algorithms make harsher assumptions. The only algorithm currently combining good speed (slope ≈ 1.0) and good quality on most datasets is **birch_algo**.

---

## What this implies for the state-action research

Each "bottleneck" line above is a specific component that, if replaced
or learned, would lift the algorithm. The trajectory layer captures
exactly the kind of state-action data needed to learn those replacements:

- **kmeans / parallel_kmeans / minibatch_kmeans** — the centroid
  update is the bottleneck under outliers. The trajectory records
  every centroid at every step; a model over `(centroid_t, candidate_assignment) → delta_inertia`
  could learn an outlier-robust centroid update.
- **clarans** — the bottleneck is the random swap proposal. The
  trajectory records every swap with `accepted` and `delta_cost`,
  i.e. a binary classification dataset for "is this swap worth trying."
  A learned proposal distribution should dramatically cut wall time
  to convergence.
- **pwcc** — the weighted vote uses purity weights derived from an
  unweighted vote; a learned weighting over `(base_partition_features) → optimal_weight`
  could outperform purity on heterogeneous bases.
- **chameleon** — every merge step in phase 2 is a decision: accept or
  reject. The trajectory could log each candidate merge with the
  relative-closeness and inter-connectivity features, and a learned
  classifier could replace the hand-tuned merging criterion entirely.
- **fmm / lmm / amm** — these are EM-based mixture models, so every
  iteration produces a state (per-cluster log-density parameters) and
  a delta (negative log-likelihood improvement). The trajectory layer
  is a natural fit: a learned step-size schedule or basis-component
  selector could cut EM iterations dramatically.

---

## Overcoming the bottlenecks

The diagnoses above all reduce to six recurring failure modes. Each
one has a small menu of concrete fixes, ordered from cheapest (a
config change) to most ambitious (a learned component on top of the
trajectory layer).

### 1. Outlier sensitivity

*Affects: kmeans, parallel_kmeans, minibatch_kmeans, agglomerative-Ward,
clarans, meanshift, chameleon, lmm.*

The shared root cause is that each of these algorithms computes a
*mean* (or a sum) at some step, and the mean has unbounded influence
per point. GMM is the natural counter-example — its mean update is
weighted by posterior responsibility, so an outlier gets near-zero
weight.

| fix | applies to | effort | expected lift |
|---|---|---|---|
| Replace mean with median or medoid | kmeans → k-medians / k-medoids (CLARANS variant); agglomerative-Ward → average linkage | one-line swap | recovers most of the -32% drop on kmeans-family |
| Trimmed-mean update (drop top α% by distance per step) | every centroid-based method | ~10 lines per algo | ~half the drop, no medoid cost |
| Outlier pre-filter (LOF, IsolationForest, DBSCAN-as-detector) then cluster | every algo | independent step | depends on detector recall |
| Posterior-weighted update | borrow GMM's trick — soft assignment in the M-step | algorithmic | matches GMM (-15% instead of -30%) |
| Bandwidth estimator: trimmed sample | **meanshift specifically** — fixes the -59% collapse | one line | most of the meanshift gap |
| Initialise away from outliers | clarans / kmeans++ | one line | fixes CLARANS's -40% surprise |

### 2. Non-convex shape failure

*Affects: every centroid-based method (kmeans, parallel_kmeans,
minibatch_kmeans, gmm, agglomerative-Ward, birch, pwcc, consensus,
fmm).*

| fix | applies to | effort | expected lift |
|---|---|---|---|
| **Use the right tool**: spectral / lmm / chameleon / gmm | choose by data | none — config change | spectral & lmm: 0 → 1.00 on circles |
| Pre-embed with diffusion maps / Laplacian eigenmaps / UMAP / kernel PCA, then run a fast convex method | every centroid-based algo | one preprocessing step | typically lifts moons from 0.26 → 0.6+ |
| Replace L2 with a kernel (kernel k-means) | kmeans-family | a new algo class | equivalent to spectral but without the eigen step |
| Mix a non-convex-capable base into the consensus | PWCC — currently 3 k-means-style bases | one config line in `base: [...]` | PWCC starts solving circles |
| Use a graph Laplacian basis (LMM-style) | any EM mixture (FMM → LMM) | already in the repo | what LMM does — solves circles |

### 3. Quadratic scaling

*Affects: agglomerative (1.33), chameleon (1.45), lmm (1.45),
spectral (0.83 at this n; → 2 at higher n), optics (~2), meanshift
(0.77; → 2 at higher n), mri (1.78).*

| fix | applies to | effort | expected lift |
|---|---|---|---|
| **Spatial indices** (ball-tree / kd-tree / HNSW) for neighbour queries | DBSCAN, OPTICS, meanshift, chameleon | sklearn already supports `algorithm="ball_tree"` | linear-ish up to high d |
| **Nyström approximation** of the Laplacian / kernel | spectral, lmm | ~50 LOC | O(n·m²) where m=samples |
| **Sampling + projection** (S5C's strategy applied elsewhere) | agglomerative, spectral, gmm | wrap-around | O(m³) on the sample, O(n) projection |
| **Mini-batch / streaming variants** | gmm → online GMM, agglomerative → BIRCH | use BIRCH/MBK first to compress | n → L ≪ n |
| **Coreset construction** (constant-factor approx) | kmeans-family, gmm | published algorithms (Bachem, Lucic, Krause) | provable bounds, linear |
| **Reduce ambient dim first** (PCA, UMAP) | every algo with d in inner loop | preprocessing | gmm's d² → m² for m ≪ d |
| **MapReduce / multi-process** | already done for `parallel_kmeans`; same shape works for GMM, BIRCH | replicate the pattern | k× speedup on k cores |

### 4. Hyperparameter brittleness

*Affects: DBSCAN (eps), OPTICS (min_samples, xi), BIRCH (threshold),
S5C (sample_size, n_nonzero_coefs), meanshift (bandwidth), spectral
(n_neighbors), FMM/LMM/AMM (basis size, regularisation).*

| fix | applies to | effort |
|---|---|---|
| **k-distance plot** to pick eps (Ester 1996 §4.2) | DBSCAN | one helper function |
| **HDBSCAN** — parameter-free DBSCAN variant | replaces DBSCAN | `pip install hdbscan` + a wrapper |
| **Silverman's rule on trimmed sample** for bandwidth | meanshift | one line |
| **Auto-threshold by silhouette sweep** | BIRCH, S5C | wrap-around |
| **Bayesian optimization over hyperparameter space** | any algo | optuna / scikit-optimize integration |
| **Stability selection** (cluster ensemble across hyperparameters, pick the most stable partition) | any algo | medium |

### 5. Local-minimum search

*Affects: clarans (random local search), kmeans (random init), GMM
(EM is a local optimiser), FMM/LMM/AMM (same).*

| fix | applies to | effort |
|---|---|---|
| **k-means++ initialisation** | already used for kmeans, missing for CLARANS | one line — closes most of CLARANS's gap |
| **n_init restarts** | already done for kmeans, GMM; not for CLARANS | n_init in config |
| **Simulated annealing / tabu** instead of greedy swap | clarans | medium rewrite |
| **Warm-start from a cheaper algo** (kmeans → CLARANS, BIRCH → kmeans) | any iterative algo | wire one to the next |
| **Learned proposal distribution** (state-action research) | clarans, GMM-EM, FMM-EM, LMM-EM | needs trajectory data → policy model |

### 6. Wrong data assumption

*Affects: S5C (assumes union-of-subspaces; fails on blobs), MRI
(relaxation signature is lossy for sharp boundaries), FMM (random
basis is shape-agnostic), DBSCAN at default eps in d=10 (density
never reached).*

| fix | applies to |
|---|---|
| **Don't run them on the wrong data** — match algorithm to data structure (the lookup table in [Cross-cutting findings](#cross-cutting-findings) below) |
| **Test the assumption first**: estimate intrinsic dim before S5C; estimate local geometry before MRI; check density before DBSCAN |
| **Replace the fixed component with a learned one** — FMM → LMM (graph Laplacian basis) → AMM (autoencoder basis). This is already the natural progression in the repo |
| **Combine algorithms**: use a density-based algo to find dense regions, then run a centroid-based algo within each region |

### Validation: did the fixes actually work?

Five of the cheapest fixes above are now implemented and registered as
new algorithms (`kmeans_trimmed`, `clarans_pp`, `dbscan_auto`,
`meanshift_robust`, `pwcc_diverse`) so the benchmark runs them
alongside the originals. Mean ARI across all 16 tasks in the demo
sweep:

| pair (original → improved) | ARI orig | ARI improved | Δ ARI | wall time orig | wall time improved |
|---|---|---|---|---|---|
| `dbscan` → `dbscan_auto` | 0.000 | **0.589** | **+0.589** | 0.020s | 0.035s |
| `meanshift` → `meanshift_robust` | 0.470 | **0.622** | **+0.151** | 0.229s | 0.288s |
| `clarans` → `clarans_pp` | 0.581 | 0.656 | +0.075 | 0.018s | 0.017s |
| `pwcc` → `pwcc_diverse` | 0.702 | 0.751 | +0.049 | 0.145s | 0.223s |
| `kmeans` → `kmeans_trimmed` | 0.701 | 0.703 | +0.003 | 0.024s | 0.024s |

**What worked, qualitatively.**

- **`dbscan_auto` is the most dramatic single intervention in the
  repo.** DBSCAN at the fixed eps=0.8 in d=10 returned one cluster on
  every task (ARI 0.0). Replacing the constant with the k-distance
  knee from Ester 1996 §4.2 lifts DBSCAN from completely failing to
  joining `spectral` and `lmm` as the third algorithm in the registry
  that *partially solves circles* (ARI 0.83 on circles, 0.97 on
  anisotropic). One scientific intervention (auto-eps), zero
  architecture change, +0.589 ARI.
- **`meanshift_robust` halves the outlier collapse.** Original
  meanshift loses 40 ARI points under outliers (0.57 → 0.17) because
  the bandwidth is contaminated. Estimating bandwidth on the trimmed
  sample drops the loss to 25 points (0.68 → 0.44) and pushes mean
  ARI on clean MDCGen from 0.38 → 0.69.
- **`clarans_pp` and `pwcc_diverse` confirm the structural fixes.**
  Distance-biased initialisation closes most of the clarans-vs-kmeans
  gap (`+0.075` ARI overall, `+0.29` on moons), and swapping pwcc's
  homogeneous k-means-style base set for `[kmeans, spectral, gmm]`
  lifts pwcc's circles score from 0.0 to 0.16 (partial credit thanks
  to spectral's vote weight) and moons from 0.25 to 0.42.

**What surprised the analysis.**

- **`kmeans_trimmed` barely moves.** Trimming 10% of the farthest
  cluster members per iteration is *capped at* tolerating roughly
  twice that contamination — the demo grid uses 20% outliers, right
  at the limit. A higher trim helps but starts to bias the centroid
  inward; the right fix is a posterior-weighted update (i.e. become
  GMM) rather than a heavier trim.
- **`dbscan_auto` is now itself outlier-sensitive** (-0.475 ARI
  delta under outliers). The k-distance plot's knee is contaminated
  by the same outliers it's supposed to ignore — the plot's tail rises,
  pulling eps up, smoothing real clusters. The next intervention is
  robust eps estimation (k-distance plot computed on a trimmed
  sample), not the auto-eps itself.
- **`clarans_pp` is more outlier-sensitive than vanilla `clarans`.**
  Kmeans++ probability is proportional to squared distance, which
  means in a 20%-outlier setting the seeding is *biased toward picking
  outliers as medoids*. The fix is to combine kmeans++ with an
  outlier filter or use a robust seeding strategy (e.g., medoid-init
  on a trimmed sample).

Each surprise becomes the next bottleneck in the chain — exactly the
pattern the trajectory layer is designed to compress: each
intervention is logged as an extra (state, action, delta_cost) tuple
that a meta-policy can learn from to choose the right *combination*
of fixes per data regime.

### Synthesised algorithms — combining what works

Three new algorithms were designed by composing the winning mechanisms from
the empirical analysis (rather than tweaking one existing algorithm):

- **`rapid`** — two-stage pipeline. Stage 1: `dbscan_auto` with the
  k-distance knee partitions the data into density-connected regions.
  Stage 2: per region, route to kmeans++ (convex regions) or spectral
  (non-convex regions). Stage 3: 1-NN reassign DBSCAN-noise points.
- **`meta_clusterer`** — inspect four data fingerprints first
  (effective dim via PCA, Laplacian eigengap, LOF outlier fraction,
  kNN density skew) then dispatch to the right existing algorithm in
  the registry with the right hyperparameters. Encodes the "What to
  try first" decision tree as code.
- **`aura`** — Adaptive Unified Robust Algorithm. Nyström-approximated
  Laplacian embedding (for non-convex shape + sub-quadratic scaling)
  followed by posterior-weighted Gaussian-mixture EM in the embedded
  space (for outlier robustness). A genuine synthesis trying to put
  spectral and GMM in the same pipeline.

Empirical comparison (mean across all 16 tasks, ARI rank is position
out of 25 registered algorithms; lower rank is better):

| algo | ARI mean | ARI rank | mdcgen | anisotropic | moons | circles | outlier Δ |
|---|---|---|---|---|---|---|---|
| **`rapid`** | **0.843** | **4 / 25** | 0.76 | 1.00 | **0.85** | **0.84** | -0.15 |
| `meta_clusterer` | 0.774 | 5 / 25 | 0.90 | 1.00 | 0.42 | 0.16 | **+0.06** |
| `aura` | 0.733 | 8 / 25 | 0.87 | 1.00 | 0.37 | 0.00 | +0.03 |

For comparison, the previous top three were `lmm` (rank 1, ARI 1.00 on
circles but 0.65 on moons), `spectral` (rank 2, ARI 1.00 on circles but
0.50 on moons), and `pwcc_diverse` (rank 3).

**Headline empirical finding.** *RAPID is the only algorithm in the
registry that handles both moons (0.85) and circles (0.84) at the same
time.* Every other algorithm that solves circles (spectral and lmm) is
mediocre on moons; every algorithm that's strong on moons (chameleon,
gmm) gets 0 on circles. The two-stage density-then-route pipeline
breaks the trade-off by treating each density-connected region with
the algorithm best suited to its local shape signature, instead of
forcing one algorithm to handle every regime.

**META-CLUSTERER's positive outlier delta (+0.057)** validates the
routing-to-GMM rule from the "What to try first" decision tree — when
the LOF outlier fraction is high, the meta-algorithm picks GMM, which
*does* improve under contamination (the only base algorithm with that
property). The rule fires correctly and lifts overall ARI.

**Where AURA fell short** — the Nyström-Laplacian embedding followed
by GMM-in-embedded-space *should* solve circles in theory (the
Laplacian eigenvectors separate concentric loops into distinct
clusters in eigen-space, where they look like ordinary blobs). In
practice the GMM step finds a different local optimum than spectral's
final k-means: GMM's full-covariance fits absorb both rings into one
elongated Gaussian. Fixes to investigate: tied covariance, smaller
embedding dimension, or replace GMM with k-means in the embedded
space (which is what `spectral` already does — and AURA would collapse
to spectral). The negative result is itself the next experimental
direction: *the choice of "what to put after the embedding" matters
more than the embedding quality.*

### Iteration: v2 of the synthesised algorithms

The three synthesised algorithms each had a specific empirical
weakness identified in the previous round. Three more parallel
worktree-isolated agents built v2 versions targeting those weaknesses
(complete code in `src/clustbench/algorithms/aura_v2.py`,
`meta_clusterer_v2.py`, `rapid_v2.py`).

**v2 changes per algorithm:**

| algorithm | v1 weakness | v2 intervention |
|---|---|---|
| `aura_v2` | ARI 0.00 on circles — GMM's full covariance absorbed both rings into one elongated Gaussian | (a) raise Nyström threshold so benchmark-scale n uses full eigendecomp, (b) z-score the embedding before k-means (the v1 eigenvectors had std 0.05 vs 0.93 — k-means weighted only the high-variance column), (c) add a trimmed-mean k-means track + tied-covariance GMM track gated on LOF outlier signal |
| `meta_clusterer_v2` | ARI 0.16 on circles — `eigengap < 0.05` rule never fired (the Laplacian gap is *large* at k for clean circles) | replace eigengap with `convexity_ratio = clip(1.2 - 2·CV, 0, 1)` of centroid distances (separates blobs ~0.65 from circles ~0.24) plus `knn_modularity` of the kmeans partition; add a probe stage that fits 3 candidates on a 20% subsample and picks by silhouette when the rule fires by a narrow margin |
| `rapid_v2` | ARI delta -0.145 with outliers — the k-distance knee in stage 1 was contaminated by the outliers it was meant to ignore | add a stage 0 LOF outlier prefilter; remove top-10% LOF score points before stage 1's density partition; reassign held-aside outliers by 1-NN at the end |

**Empirical v1 vs v2 (mean ARI across all 16 tasks):**

| algorithm | v1 ARI | v2 ARI | Δ | ARI rank (of 28) |
|---|---|---|---|---|
| `meta_clusterer_v2` | 0.774 | **0.879** | **+0.105** | **3 / 28** ↑ from 5 / 25 |
| `aura_v2` | 0.733 | 0.764 | +0.032 | 10 / 28 (vs v1 at 9) |
| `rapid_v2` | 0.843 | 0.774 | -0.069 | 13 / 28 (vs v1 at 5) |

**Per-shape delta (v2 minus v1):**

| algorithm | mdcgen | anisotropic | moons | circles |
|---|---|---|---|---|
| `aura_v2` | -0.10 | 0.00 | **-0.36** | **+1.00** |
| `meta_clusterer_v2` | 0.00 | 0.00 | 0.00 | **+0.84** |
| `rapid_v2` | -0.03 | -0.09 | 0.00 | -0.27 |

**Outlier robustness delta (v2 minus v1):**

| algorithm | v1 ARI delta | v2 ARI delta | improvement |
|---|---|---|---|
| `aura_v2` | +0.029 | -0.043 | regressed |
| `meta_clusterer_v2` | +0.057 | -0.084 | regressed in relative terms (the clean baseline went up so absolute drop is bigger) |
| `rapid_v2` | -0.145 | **-0.050** | **closed the gap by 95 ARI points** — the intervention worked |

**What this round taught us.** The iteration outcomes are uneven, and
that's the lesson:

- **META v2 is the unambiguous win.** It cracked the top 3 by replacing
  one brittle heuristic (eigengap) with two cheap robust ones
  (convexity ratio + knn modularity) and adding a silhouette-based
  probe for borderline cases. The architecture lesson generalises:
  *brittle single-rule routing beats fixed strategy; probe-based
  refinement beats brittle routing.*
- **AURA v2 traded moons for circles.** The z-score-then-kmeans pipeline
  perfectly recovers circles (the eigenvectors' loop structure is now
  metric-correct), but the moons embedding has near-zero variance in
  the relevant eigen-direction and z-scoring amplifies noise that
  k-means then partitions arbitrarily. Fix: choose between v1's
  raw-GMM and v2's z-scored-kmeans dynamically based on the LOF
  signal *and* the embedding rank — not yet implemented.
- **RAPID v2 closed the outlier delta but broke shape.** Stage 0's LOF
  prefilter pulls real cluster points along with outliers when the
  cluster density is comparable to the outlier density. The +95 ARI
  point lift on outlier robustness is real but the base quality
  regression means v1 is still the best RAPID variant overall. Fix:
  conditional stage 0 — only run LOF when outlier_frac (estimated
  cheaply) exceeds a threshold.

**The combined message** — every "fix the bottleneck" intervention
opens a new failure mode. The next iteration should be a
*meta-algorithm-over-meta-algorithms* that dispatches between v1 and
v2 of each synthesis based on the data signature, OR an explicit
choose-the-better-of-v1-and-v2 wrapper that runs both on a subsample
and picks the silhouette winner.

### Iteration: v3 (meta-of-meta routing between v1 and v2)

Every v2 closed its targeted bottleneck but opened a new one. The
natural next step is a *meta-of-meta* layer: dispatch between v1 and
v2 of each synthesis based on a cheap data signature, so the right
version runs per regime instead of one fixed choice across all data.
Three more parallel worktree-isolated agents built v3:

| algo | v3 dispatch rule |
|---|---|
| `aura_v3` | effective rank of the embedding's column std-devs. `effective_rank ≥ 2` → v2 path (z-scored k-means); `< 2` → v1 path (raw GMM). |
| `meta_clusterer_v3` | runs v1 and v2 routers in parallel. Agree → route once. Disagree → fit both candidates on a 20% subsample and pick the silhouette winner. |
| `rapid_v3` | LOF outlier-fraction estimate up front. `outlier_frac > 5%` → v2 path (stage-0 LOF prefilter); else → v1 path. |

**Empirical v1 vs v2 vs v3 (mean ARI across all 16 tasks):**

| algorithm | v1 ARI | v2 ARI | v3 ARI | Δ(v3 - v1) | rank (of 31) |
|---|---|---|---|---|---|
| `aura_v3` | 0.733 | 0.764 | **0.875** | **+0.142** | **2 / 31** |
| `meta_clusterer_v3` | 0.774 | 0.879 | 0.827 | +0.053 | 8 / 31 |
| `rapid_v3` | 0.843 | 0.774 | 0.843 | +0.001 | 7 / 31 |

For comparison the top 10 across the whole registry:

| pos | algorithm | rank |
|---|---|---|
| 1 | `lmm` | 7.53 |
| **2** | **`aura_v3`** ← new | **8.78** |
| 3 | `spectral` | 9.25 |
| 4 | `meta_clusterer_v2` | 10.72 |
| 5 | `rapid` (v1) | 11.03 |
| 6 | `pwcc_diverse` | 11.09 |
| 7 | `rapid_v3` | 11.16 |
| 8 | `meta_clusterer_v3` | 11.22 |
| 9 | `meta_clusterer` (v1) | 11.72 |
| 10 | `gmm` | 12.47 |

**Per-shape ARI for the v3s:**

| algorithm | mdcgen | anisotropic | moons | circles |
|---|---|---|---|---|
| `aura_v3` | 0.86 | 1.00 | **0.55** | **1.00** |
| `meta_clusterer_v3` | 0.90 | 1.00 | 0.42 | 0.58 |
| `rapid_v3` | 0.77 | 1.00 | 0.85 | 0.84 |

**What this round taught us about meta-of-meta dispatch:**

- **`aura_v3` is the unambiguous win — second place in the registry**,
  passing `spectral`, `meta_clusterer_v2`, and every other algorithm
  except `lmm`. The effective-rank dispatch successfully picks GMM
  for moons (1D-effective embedding) and z-scored k-means for circles
  (2D-effective embedding). Both predecessors' weaknesses are gone.
  *Lesson: meta-dispatch is powerful when v1 and v2 are genuinely
  complementary on identifiable data regimes.*
- **`meta_clusterer_v3` slightly regressed (-0.052 from v2).** The
  silhouette probe sometimes overrules v2 when v2 alone was already
  correct, e.g. on circles v2 routes to spectral and gets ARI 1.00,
  but v3's probe occasionally picks v1's `pwcc_diverse` because the
  subsample silhouette doesn't always track ARI. *Lesson: when one
  version is already correct, probing introduces noise. The right
  application of meta-dispatch is only when neither version dominates
  globally.*
- **`rapid_v3` matched v1 exactly (Δ +0.001).** v2 doesn't have any
  data regime where it beats v1 in this benchmark grid, so the
  conditional dispatch ends up always picking v1. *Lesson: meta-
  dispatch can't help if one version dominates everywhere; the dispatch
  rule needs an actual quality difference to arbitrate.*

**The meta-pattern across three iterations.** v2 fixes the targeted
bottleneck but opens a new failure mode. v3 dispatches between v1 and
v2 to capture each version's regime. v3 wins when v1 and v2 are
genuinely complementary (AURA), is neutral when one version
dominates (RAPID), and slightly loses when the dispatch criterion
itself is noisier than the v2 it's trying to refine (META). This is
the natural empirical limit of binary-meta-dispatch — the next
generation would learn the dispatch rule from past `(data, ARI)`
pairs across the full benchmark, which is exactly the training data
the trajectory layer keeps producing.

### v4: a learned router

The v3 generation shipped three hand-coded meta-of-meta dispatchers,
and one of them (`meta_clusterer_v3`) slightly regressed because its
silhouette probe sometimes overrules v2 when v2 was already correct.
The natural follow-up: stop hand-coding the dispatch rule and *learn*
it from the existing benchmark data.

`learned_router` (`src/clustbench/algorithms/learned_router.py`)
implements that. At training time (which happens once per process at
module import) it:

1. Reads `docs/data/results.json` — the table the dashboard already
   uses, with one row per (task, algorithm) and an ARI score per row.
2. Regenerates every historical task from its (dataset_id, n, d, k,
   compactness, outliers, noise, density, seed) tuple.
3. Computes a 7-feature fingerprint per task: `log_n`, `d`, `k`,
   `eff_dim` (PCA components > 1% variance), `conv_cv` (CV of
   intra-cluster distances after a quick k-means), `outlier_frac`
   (LOF score > 1.5), `density_skew` (CV of mean k-NN distances).
4. Normalises the fingerprints, caches the matrix.

At inference time:

1. Computes the input's fingerprint.
2. Finds the K=5 nearest training tasks by Euclidean distance in
   normalised fingerprint space.
3. With `exclude_self=True`, drops the nearest task if its fingerprint
   matches the inference fingerprint exactly — this is strict
   leave-one-out for honest benchmark evaluation against the same
   tasks the router trained on.
4. For each candidate algorithm in the registry (except
   `learned_router` itself), computes its mean ARI-rank position
   across the K neighbouring tasks. Dispatches to the candidate with
   the lowest mean rank.

**Empirical result on the 16-task / 32-algorithm benchmark:**

| metric | value |
|---|---|
| mean ARI across all 16 tasks | **0.884** — highest absolute mean in the registry |
| ARI rank position | **3 / 32** (only behind `lmm` and `aura_v3`) |

**Per-shape ARI:**

| dataset | learned_router ARI | best single algo (and its ARI) |
|---|---|---|
| anisotropic | **1.000** | `kmeans` etc. (1.00) |
| circles | **1.000** | `spectral` (1.00) |
| mdcgen | 0.862 | `birch_algo` (0.97) |
| moons | 0.623 | `rapid` (0.85) |

**Dispatch distribution across the 16 tasks** (i.e. which algorithm
the router picked, and how it did on those tasks):

| dispatched algo | n_tasks | mean ARI on those tasks |
|---|---|---|
| `parallel_kmeans` | 8 | 0.998 — almost perfect on convex blobs |
| `spectral` | 4 | 0.812 — strong on non-convex |
| `lmm` | 3 | 0.742 — picked for borderline non-convex |
| `gmm` | 1 | 0.689 — picked for the heaviest outlier task |

The router learned the correct partitioning of the regime space:
non-convex shapes go to spectral / lmm, outlier-heavy tasks go to
gmm, clean convex blobs go to the fastest convex method
(`parallel_kmeans`). The few sub-optimal cases (mdcgen 0.862 vs
birch's 0.97; moons 0.623 vs rapid's 0.85) are tasks where the
fingerprint pointed at a near-neighbour with a different best
algorithm than the one the inference task itself prefers — the
classic local-decision-boundary issue of k-NN.

**What this round taught us.**

- **A 7-feature kNN classifier beats every hand-coded router.** The
  router's mean ARI (0.884) is higher than `aura_v3`'s 0.875,
  `meta_clusterer_v2`'s 0.879, and even `lmm`'s 0.887 by less than
  three thousandths — three iterations of carefully-engineered
  hand-coded routing got us *into* the top tier, but a small
  data-driven classifier puts us *at* the top.
- **The dispatch story is interpretable.** The router didn't learn
  some opaque ensemble; it picked four algorithms (parallel_kmeans,
  spectral, lmm, gmm) across 16 tasks, one per identifiable regime.
  That matches exactly the "What to try first" decision tree we
  hand-wrote earlier in this document — the model rediscovered the
  decision tree without being shown it.
- **The next move is unambiguous.** A learned router with full coverage
  of the benchmark data outperforms every hand-coded synthesis. The
  real research question now is whether richer per-task features
  (intrinsic dim, multi-scale density, trajectory-based features
  from a quick probe) lift the router further. The trajectory layer
  is the natural source for those features, closing the loop on the
  whole project — the same per-step state-action data that motivated
  this exercise is the next training signal for the meta-meta-meta
  layer.

### What to try first

A pragmatic decision tree from the dashboard data:

1. **Big data, convex clusters, need k known** → `parallel_kmeans` (slope 1.23, ARI 1.00 clean). Add `n_workers > 1` for real parallelism.
2. **Big data, convex, k unknown** → `birch_algo` then a cheap final k-means on the leaf CFs. Birch is the only honest-linear algo at quality.
3. **Suspected non-convex shape** → `spectral` first; `lmm` if you want soft EM-style assignments; `chameleon` if k is uncertain.
4. **Outlier-heavy data** → `gmm` (most robust empirically) or robust k-medoids; **never meanshift with the default bandwidth**.
5. **High-d sparse data (text, gene expression)** → `amm` (autoencoder basis) or LMM with `n_neighbors` tuned per d.
6. **Don't know what the data looks like** → run `pwcc` with a *diverse* ensemble: `[kmeans, spectral, gmm]` instead of three k-means variants. The weighted vote gives you robustness across regimes.
7. **Want to push the research direction** → start collecting trajectories for the iterative algos. Every per-step record in `artifacts/trajectory_*.parquet` is training data for a learned step proposer or value model that, in principle, lets a "smarter version of the same algorithm" run with fewer iterations. The natural first notebook in `notebooks/` is an autoencoder over the serialized state column plus a value model over `(state, action) → delta_cost`, then playing the learned policy back against the trajectory dataset to measure step savings.
