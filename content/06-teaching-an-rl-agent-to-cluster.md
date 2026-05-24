---
title: "Teaching a Reinforcement-Learning Agent to Cluster From Primitives"
subtitle: "I broke clustering into 8 primitive moves and tried to teach an agent to chain them. It learned the spectral trick from imitation alone, then hit a wall on bigger data. Here's exactly where, and why."
series: "Benchmarking 44 Clustering Algorithms: Part 6"
tags: [machine-learning, reinforcement-learning, clustering, behavior-cloning, data-science]
---

# Teaching a Reinforcement-Learning Agent to Cluster From Primitives

Way back in Part 1 I mentioned the trajectory layer: clustbench records
every step of every iterative algorithm as a `(state, action, cost,
delta_cost)` record. Every centroid update, every swap, every merge
becomes a row. I said the point of the series was to find out why that
mattered.

This is the payoff. If clustering algorithms are just *sequences of
primitive operations*, then maybe we don't need to pick an algorithm at
all. Maybe an agent can learn to **chain the primitives itself**,
choosing the right move at each step based on the state of the
clustering so far. A learned algorithm, composed on the fly.

That's the framing: clustering as a sequential decision problem. Here's
how far I got, and exactly where it broke.

## The action ontology

I decomposed the registry's algorithms into eight primitive actions,
the recurring "moves" that show up across k-means, k-medoids,
agglomerative, spectral, and density methods:

| action | what it does | borrowed from |
|---|---|---|
| `kpp_init` | k-means++ seeding | k-means |
| `assign_to_centers` | Voronoi assignment | k-means |
| `update_centers` | recompute centroids | k-means |
| `medoid_swap` | swap a center for a data point | CLARANS / k-medoids |
| `ward_merge` | merge two clusters by min variance | agglomerative |
| `density_partition` | density-based split | DBSCAN |
| `eigen_embed` | re-embed via the graph Laplacian | spectral |
| `outlier_trim` | drop the farthest points | robust variants |

A `ClusteringState` summarizes "where we are" as a length-20 vector:
n, d, target k, current cost, silhouette, plateau signals, outlier
fraction, centroid statistics. The environment's reward is the
normalized drop in clustering cost at each step. An episode ends on a
step cap, a silhouette threshold, or a plateau.

Sanity check first: I verified that *hand-written* trajectories through
this environment reproduce the source algorithms. A k-means-equivalent
sequence reaches ARI 1.0 on blobs, and a spectral-equivalent sequence
(`kpp → eigen_embed → assign → update`) reaches ARI 1.0 on concentric
circles. The primitives are expressive enough. Now: can an agent
*learn* to chain them?

## Behavior cloning: learn the moves by imitation

Full reinforcement learning from scratch needs millions of rollouts.
The cheaper, sane first step is **behavior cloning (BC)**: supervised
learning on demonstrations.

I drove five existing algorithms (k-means, CLARANS, agglomerative,
spectral, DBSCAN) through the environment and recorded what they did:
**50 episodes, 419 (state, action, reward) rows.** Then I trained two
small PyTorch networks:

- a **policy** net: state → probability over the 8 actions,
- a **value** net: state → expected return-to-go.

The policy reached **59% validation accuracy** at predicting the next
action (chance is 12.5%; the majority-class baseline is 41%, since
`assign_to_centers` dominates the traces). I had to mildly up-weight
rare actions so the 10 `eigen_embed` examples didn't drown under the
170 `assign_to_centers` ones.

At inference, the agent runs several stochastic rollouts and a tiered
selector keeps the best clustering: a convex-looking rollout if one
clears a silhouette bar, otherwise the best spectral-embedded rollout.

## The win: it learned the spectral trick from imitation

Here's the result that genuinely surprised me. On concentric circles
(the canonical non-convex trap that k-means scores ~0 on) the agent
reached **ARI 1.0**.

And it did it *the right way*. The chosen trajectory was:

```
kpp_init → eigen_embed → ward_merge → kpp_init → assign → update → ...
```

The agent had no hand-coded knowledge that spectral methods solve
rings. It learned, from 419 imitation traces, that **when the cost
surface refuses to drop, the move that helps is `eigen_embed`**: switch
to the Laplacian embedding where the rings become separable. Of its 16
rollouts, one fired `eigen_embed` early and landed at silhouette 0.76
in the embedded space; the selector recognized that no non-spectral
rollout could match it and picked it.

A learned policy rediscovered the core idea of spectral clustering as a
*move* to deploy conditionally. That's the trajectory layer paying off.

## The wall: it doesn't generalize to bigger data

Now the honest half. Across the full benchmark, `rl_pipeline` ranked
**36th of 43**, mean ARI **0.584**, well below the routers (0.82–0.88)
and below plain spectral (0.765).

It's bimodal. On its Gaussian-mixture wheelhouse it's excellent:
anisotropic blobs 0.996, a graph SBM 0.982. But on the same circles it
*aced* at 300 points, it scored **−0.001 at 500 points**.

I dug into why, and the failure is precise and instructive. At n=500
the policy *still fires* `eigen_embed` at step 0. The spectral instinct
survived. But the trajectory then terminates after a few
`update_centers` no-ops. The agent **never learned the full follow-up
sequence** needed to turn a 500-point spectral embedding into a clean
partition. At n=200–300 a stochastic rollout stumbles onto a good
completion; at n=500 every rollout collapses into the same dead end.

This is the **behavior-cloning ceiling**, made visible. 419 traces are
enough to learn *individual moves* ("try eigen_embed when stuck") but
not enough to learn the *coordinated multi-step sequences* that make a
move pay off. BC imitates per-step decisions; it has no mechanism to
value a move by where the *whole episode* ends up.

## What fixes it (and why I stopped here)

The textbook fix is **PPO on top of the BC initialization**: let the
agent explore its own rollouts and re-weight actions by the *terminal*
silhouette rather than per-step imitation loss. BC teaches the
vocabulary; reinforcement learning would teach the grammar. That's the
natural next stage, and it needs hours of rollout compute I haven't
spent yet.

I'm deliberately reporting the BC result on its own, because it stands
as a finding: **a learned policy over primitive clustering operators
can recover the spectral pipeline from imitation alone, at low overall
accuracy, but unmistakably non-random, and not by memorization.**

`rl_pipeline` is not a clustering algorithm I'd recommend you deploy. It
is a working proof that the trajectory layer is trainable: that
"clustering as a sequence of learned moves" is a real, tractable
research direction and not just a slide.

The agent, the environment, and the 419-row trace dataset are all in
the repo: https://github.com/Ashail33/clustbench

---

*Part 5: "My Best Model Overfit. A Simpler One Won." Next, Part 7:
"What I Learned Building a Self-Extending ML Benchmark."*
