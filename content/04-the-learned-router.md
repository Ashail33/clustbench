---
title: "Training a Model to Pick the Right Clustering Algorithm"
subtitle: "Instead of one best algorithm, a model that reads your data and dispatches to the one most likely to win. It's meta-learning, and it climbs straight to the top of the leaderboard."
series: "Benchmarking 44 Clustering Algorithms, Part 4"
tags: [machine-learning, clustering, meta-learning, automl, data-science]
---

# Training a Model to Pick the Right Clustering Algorithm

Part 3 ended with a reframe: if no single algorithm wins everywhere,
and composing them tops out at the component frontier, then the real
leverage is **selection**: reading a dataset and dispatching to the
algorithm most likely to win on it.

Hand-coded routing (AURA) already did a version of this with a few
cheap probes. The natural next step is to stop hand-coding the rules and
*learn* them from the benchmark's own accumulated history.

That's the **learned router**, and it sits at the top of the
leaderboard. It's also where the project's biggest cautionary tale
lives, but that's Part 5. First, what it actually is.

## It's not a magic forest: it's nearest neighbours over fingerprints

When people hear "learned model that picks an algorithm," they imagine
some deep ensemble. The reality is humbler and more interpretable.

For every dataset the benchmark has ever run, I compute a
**fingerprint**, a cheap feature vector describing the data's
geometry: number of samples, dimensionality, target k, effective rank,
outlier fraction, a silhouette probe, a few summary statistics. Paired
with each fingerprint is the known answer: *which algorithm got the
best ARI on this dataset*.

At inference time, the router:

1. Computes the fingerprint of the new dataset.
2. Finds the **k nearest neighbours** among all past fingerprints.
3. Looks at which algorithm won for those neighbours.
4. Runs the algorithm that won most often.

That's the core of `learned_router`: a k-NN classifier over data
fingerprints, where the labels are "the algorithm that won here." This
is a known idea in the AutoML literature (meta-learning for algorithm
selection, landmarking in the Pfahringer sense), and naming the prior
art honestly matters: I'm operationalizing an established direction on
a fresh benchmark, not inventing the genre.

To evaluate it honestly, the router uses **leave-one-out at
exact-fingerprint match**: when scoring a dataset that's already in
the history, it excludes the exact match so it can't just memorize its
own answer.

## The versions, and what each one added

Like the synthesized algorithms, I versioned the router and kept every
version in the registry:

- **v1**: k-NN over fingerprints, majority vote.
- **v2**: distance-weighted voting + a richer fingerprint.
- **v3**: meta-of-meta routing: pick between v1 and v2 based on
  fingerprint disagreement and a silhouette probe.
- **v4**: a learned dispatch rule trained on the full results table.
- **v5–v7**: probe-based refinements; v6 adds a *landmarking* probe
  (run a couple of cheap algorithms, use their results as features),
  and v7 stacks the best of the lot.

The leaderboard, mean ARI across 15 dataset configs:

| algorithm | mean ARI |
|---|---|
| **learned_router_v7** | **0.877** |
| learned_router_v5 | 0.867 |
| learned_router_v3 | 0.867 |
| learned_router_v4 | 0.860 |
| learned_router_v1 | 0.816 |
| spectral (best classical) | 0.765 |

The top of the board is a clean sweep of routers. v7 beats the best
single classical algorithm (spectral) by **+0.11 ARI**, a large gap
in clustering terms. On the training distribution, learning *which*
algorithm to run genuinely beats running any *one* algorithm.

## The methodology that makes it self-extending

Here's the part I'm proudest of. The routers don't need new code to
improve; they need new *data*. When I later added a graph-native
algorithm (`louvain_knn`) to close a gap on graph datasets, I didn't
touch the router code at all. I just re-ran the benchmark. The new
results entered the history, the routers re-trained on the next run,
and they *automatically* started dispatching to `louvain_knn` on graph
data. Mean ARI lifted +0.03 with zero router changes.

That's a self-extending system: add a capability to the registry, and
the selection layer learns to use it for free.

## The asterisk

Everything above is true *on the training distribution*. The numbers
are real, the leave-one-out is honest, the methodology is sound.

But a router is a memory. It wins by recognizing that your new dataset
looks like something it has seen before. So the question that should
make you nervous is this: **what happens when the data looks like nothing
it has seen?**

I built a config specifically to answer that: new sample sizes, new
dimensionalities, new k values, new outlier extremities, and random
seeds far outside the training range. Then I ran all 43 algorithms on
it.

The router that ranked #1 on training data dropped 13 ARI points. A
plain mixture model from the 2000s beat it. That's Part 5, and it's
the most useful result in the whole series.

Code and live leaderboard: https://github.com/Ashail33/clustbench

---

*Part 3: "Letting the Analysis Design the Algorithm." Next, Part 5:
"My Best Model Overfit. A Simpler One Won."*
