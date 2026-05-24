---
title: "I Rebuilt My Master's Thesis as a Benchmark Anyone Can Run"
subtitle: "Clustering papers are almost impossible to compare. So I built the boring infrastructure that fixes the lower half of that problem."
series: "Benchmarking 44 Clustering Algorithms — Part 1"
tags: [machine-learning, clustering, benchmarking, data-science, reproducibility]
---

# I Rebuilt My Master's Thesis as a Benchmark Anyone Can Run

In 2024 I submitted a master's review titled *Review of Big Data
Clustering Methods*. Like most reviews, it ended with a table: this
method is fast, that one handles outliers, this other one needs you to
know the number of clusters in advance. And like most reviews, the
table was a snapshot — true on the day I wrote it, frozen forever after.

The problem with that table is the problem with most of the clustering
literature: **the results are almost impossible to compare across
papers.** Different authors report different metrics, on different
datasets, at different sample sizes, with different hyperparameters —
and almost nobody reports runtime or memory in a way you can reproduce.
If you want to know "should I use spectral clustering or a Gaussian
mixture for my data," the literature gives you a hundred partial
answers and no way to line them up.

So I built **clustbench**: a benchmark harness that takes the thing my
thesis *described* and makes it something you can actually *run*.

> Live dashboard: https://ashail33.github.io/clustbench/
> Code: https://github.com/Ashail33/clustbench

## What "reproducible" actually means here

Clustbench is deliberately boring infrastructure. It does four things:

1. **Generates synthetic datasets with known ground truth** at sample
   sizes you control — convex blobs, moons, concentric circles,
   anisotropic clusters, high-outlier regimes, low-rank-in-high-d
   structure, and more. Known ground truth means we can compute the
   Adjusted Rand Index (ARI) honestly instead of squinting at a
   scatter plot.

2. **Wraps real datasets** (iris, wine, breast cancer, digits) plus
   time-series feature stacks and graph node-feature matrices — so the
   benchmark spans modality boundaries, not just Gaussian toys.

3. **Runs every registered algorithm** through the same harness, the
   same metrics (ARI, NMI, silhouette, Davies–Bouldin), and the same
   resource accounting (wall time, RSS), then writes everything to a
   single results table.

4. **Publishes a live dashboard.** Every push to the repo re-runs the
   benchmark in CI and redeploys the charts to GitHub Pages. The
   numbers you see are never hand-curated — they're whatever the code
   produced on the last commit.

That last point matters more than it sounds. A benchmark you can't
re-run is a benchmark you have to trust on faith. A benchmark that
re-runs itself on every commit is one you can *audit*.

## The shape of the current run

The canonical config runs **43 algorithms across 15 dataset
configurations at 3 seeds each** — roughly 2,900 individual
(algorithm, dataset, seed) results. That includes the classic
baselines (k-means, DBSCAN, GMM, spectral, agglomerative, BIRCH,
mean-shift, OPTICS), several big-data methods from the review, and a
growing set of algorithms I synthesized later in the series.

Here's the headline you'd expect — and the one you wouldn't:

| algorithm | mean ARI |
|---|---|
| spectral | 0.765 |
| gmm | 0.745 |
| agglomerative | 0.717 |
| k-means | 0.681 |
| birch | 0.633 |
| mean-shift | 0.387 |
| optics | 0.145 |
| dbscan | 0.078 |

DBSCAN and OPTICS at the bottom is not a bug — it's the cost of a
fixed `eps` across 15 very different dataset geometries. Density
methods are exquisitely sensitive to their bandwidth, and a benchmark
that uses one config across all regimes punishes them for it. That's a
finding too: **the algorithm that needs the most tuning looks worst in
an untuned benchmark.** We come back to that tension repeatedly.

## The part that isn't boring: the trajectory layer

Most benchmarks record one row per run: which algorithm, which
dataset, final score. Clustbench records that — but it also records,
for every *iterative* algorithm, a per-step trajectory:

```
(state, action, cost, delta_cost)
```

Every centroid update in k-means, every swap proposal in CLARANS,
every merge decision in agglomerative clustering becomes a row in a
trajectory table. The optimization *process* itself becomes data.

I didn't fully appreciate why this mattered when I built it. The point
of this series is to find out. If every step of every algorithm is
captured as a state-action record, then in principle you can *learn* a
better version of that step — an outlier-robust centroid update, a
smarter swap proposal, a learned merge criterion. By the final part of
this series I'll have tried exactly that: a reinforcement-learning
agent that clusters by chaining primitive actions, trained on these
trajectories.

It works on the toy cases and hits a hard wall on the real ones. But
that's Part 6. We have a lot of ground to cover first.

## What's next

In **Part 2**, I go algorithm by algorithm and ask a single question:
*what, specifically, is holding this method back?* Not "k-means is bad
at non-convex shapes" — everyone knows that — but *which line of the
algorithm* causes it, and what you'd have to change to fix it.

The whole thing is open. Clone it, run it, break it:
https://github.com/Ashail33/clustbench

---

*This is Part 1 of a series on building a living clustering benchmark.
Part 2: "Where Each Clustering Algorithm Actually Breaks."*
