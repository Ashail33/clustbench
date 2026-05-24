---
title: "Where Each Clustering Algorithm Actually Breaks"
subtitle: "Not 'k-means struggles with non-convex shapes' — which line of the algorithm causes it, and what you'd change to fix it."
series: "Benchmarking 44 Clustering Algorithms — Part 2"
tags: [machine-learning, clustering, algorithms, data-science]
---

# Where Each Clustering Algorithm Actually Breaks

Everyone can recite the clustering folklore: k-means assumes round
clusters, DBSCAN needs the right `eps`, GMM is sensitive to
initialization. The folklore is true but useless — it tells you *that*
an algorithm fails without telling you *why*, and the "why" is where
the fix lives.

After running 43 algorithms across 15 dataset regimes (Part 1), I went
back and asked a sharper question for each method: **which specific
computation is the bottleneck, and what would you have to replace to
lift it?** Three failure axes kept recurring — outliers, non-convex
shape, and scaling — and they trace back to a surprisingly small number
of root causes.

## Outliers hurt almost everyone — and there's one reason why

Look at the mean-ARI drop when you inject outliers into otherwise clean
data, and a pattern jumps out: nearly every algorithm degrades, and
they degrade for the *same* reason.

**They all compute a mean somewhere, and the mean has unbounded
influence per point.**

- k-means, mini-batch k-means, parallel k-means: Voronoi centroids are
  means of their assigned points.
- Agglomerative (Ward): the merge criterion is a variance, which is
  built on a mean.
- BIRCH: the CF-tree summary statistics are running means.
- Mean-shift: the mode estimate is a kernel-weighted mean.

One outlier, placed far enough away, drags the mean toward it and
corrupts every downstream assignment. The one consistent exception is
the **Gaussian Mixture Model**, and the exception is instructive: GMM's
mean update is *weighted by posterior responsibility*. An outlier that
fits no component gets near-zero weight in the M-step, so it barely
moves the means. In the benchmark, GMM is the most outlier-robust of
the centroid-style methods by a wide margin (mean ARI 0.745) — not
because it's "better," but because it downweights instead of averaging.

**The fix menu**, cheapest first:

| fix | how | cost |
|---|---|---|
| Replace mean with medoid/median | k-means → k-medoids; Ward → average linkage | one-line swap |
| Trimmed-mean update | drop the top α% farthest points each step | ~10 lines |
| Outlier pre-filter | LOF / IsolationForest / DBSCAN-as-detector, then cluster | separate step |
| Posterior-weighted update | borrow GMM's trick — soft assignment in the M-step | algorithmic |

I later built `kmeans_trimmed` and `meanshift_robust` from this exact
menu. The trimmed variants recover most of the outlier-induced drop for
a handful of lines of code. (More on synthesized variants in Part 3.)

## Non-convex shapes break every centroid method — by construction

Moons and concentric circles are the classic traps. Spectral clustering
solves them perfectly (ARI ≈ 1.0); k-means scores ARI ≈ 0 on circles —
literally no better than random labels.

This isn't a tuning problem. **Every centroid-based method implicitly
assumes Voronoi cells** — each point belongs to the nearest center, so
cluster boundaries are straight lines (hyperplanes). A ring inside a
ring cannot be separated by any hyperplane through the centers. The
assumption is baked into the assignment step, so no amount of
re-initialization or extra iterations helps.

Only four mechanisms can represent non-convex clusters at all:

- **Spectral** — embeds the data via the graph Laplacian's
  eigenvectors, where the rings become linearly separable.
- **Mean-shift** — follows density gradients to modes, no convexity
  assumption.
- **DBSCAN / OPTICS** — connectivity, not centroids.
- **Subspace methods (S5C)** — model local structure.

**The fix is usually "use the right tool,"** but there's a cheaper
trick: pre-embed with Laplacian eigenmaps / diffusion maps / kernel
PCA, *then* run a fast convex method on the embedding. That's exactly
what spectral clustering is under the hood, and it typically lifts
moons from ARI 0.26 to 0.6+. Hold that thought — pre-embedding as a
*primitive action* becomes the heart of the RL agent in Part 6.

## Scaling: fast and accurate are negatively correlated

Here's the trade-off nobody likes to state plainly. In this registry,
the algorithms that win on wall-time rank *lose* on quality rank, and
vice versa:

- **parallel k-means** wins wall-time (near-linear scaling) but makes
  the harshest assumptions (convex, known k).
- **spectral** wins quality but pays an eigendecomposition that scales
  badly.
- **agglomerative, chameleon, LMM** are quadratic-ish — great quality,
  punishing at scale.

The only algorithm that combines roughly linear scaling *and* decent
quality across most datasets is **BIRCH** — it summarizes the data into
a tree of cluster features and clusters those, paying a small accuracy
tax for honest linearity. If I had to pick one default for "big,
mostly-convex, k roughly known," it'd be BIRCH followed by a cheap
final k-means on the leaf summaries.

## Why this matters for what comes next

Every bottleneck above is a *specific component* — the centroid update,
the assignment step, the merge criterion, the swap proposal. And every
one of those components is exactly what the trajectory layer (Part 1)
records, step by step.

That reframes the whole project. Instead of "which algorithm is best,"
the question becomes: *can we learn a better version of the bottleneck
component from the trajectory data?* A learned outlier-robust centroid
update. A learned swap proposal for CLARANS. A learned merge classifier
for chameleon.

In **Part 3**, I take the first step toward that idea — not by learning
components yet, but by *composing* what works. If GMM has the robust
update and spectral has the non-convex embedding, what happens if you
build new algorithms that combine the winning mechanisms?

The full per-algorithm bottleneck analysis (with the complexity tables)
lives in `docs/ALGORITHM_ANALYSIS.md`:
https://github.com/Ashail33/clustbench

---

*Part 1: "I Rebuilt My Master's Thesis as a Benchmark Anyone Can Run."
Next — Part 3: "Letting the Analysis Design the Algorithm."*
