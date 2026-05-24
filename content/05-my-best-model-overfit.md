---
title: "My Best Model Overfit. A Simpler One Won."
subtitle: "I built seven versions of a learned router for clustering. On data it had never seen, the simplest version (and a plain mixture model) beat the sophisticated one. Here's why that's the most useful result I got."
series: "Benchmarking 44 Clustering Algorithms, Part 5"
tags: [machine-learning, overfitting, generalization, meta-learning, data-science]
---

# My Best Model Overfit. A Simpler One Won.

This is the post I most wanted to write, because it's the result most
people leave out.

By Part 4 I had a learned router sitting at the top of my clustering
leaderboard: `learned_router_v7`, mean ARI **0.877**, beating the best
classical algorithm by a comfortable margin. Seven versions of
increasing sophistication: distance-weighted voting, meta-of-meta
routing, landmarking probes, a final stacker. The numbers were honest
(leave-one-out, no memorizing the test answer). It really was the best
dispatcher *on the benchmark's data*.

Then I tested it on data it had never seen. And it fell off a cliff.

## The setup: a deliberate out-of-distribution test

A router is a memory. It wins by recognizing that your new dataset
looks like something it has seen before. So I built a config,
`benchmark.unseen.yaml`, designed to look like *nothing* in the
training history:

- Sample sizes, dimensionalities, and k values not in the training grid
  (e.g. d=25, d=150, k=7).
- An outlier extremity (10×) the training set never used.
- Random seeds in the 100s, far outside the training range of 1–3.

Then I ran all 43 algorithms on it and compared each one's
out-of-distribution (OOD) ARI to its training ARI.

## The result

| algorithm | trained ARI | unseen ARI | Δ |
|---|---|---|---|
| `learned_router_v7` (rank 1 on training) | 0.879 | 0.753 | **−0.126** |
| `learned_router_v3` | 0.868 | 0.753 | **−0.115** |
| `learned_router_v5` | 0.868 | 0.753 | **−0.115** |
| `learned_router_v4` | 0.862 | 0.734 | **−0.128** |
| `learned_router_v2` | 0.826 | 0.812 | −0.014 |
| **`learned_router` (v1)** | 0.817 | **0.840** | **+0.023** |
| **`lmm`** (a Laplacian mixture model) | 0.78 | **0.845** | **+0.065** |
| `louvain_knn` | 0.769 | 0.825 | +0.056 |

Read that top-to-bottom and the story is brutal and clean:

**The more sophisticated the router, the harder it fell.** v7, v3, v4,
and v5, the four most elaborate models, each lost 11–13 ARI points on
unseen data. The simplest router, v1, *gained* a little. And the
overall winner on unseen data wasn't a router at all. It was `lmm`, a
straightforward mixture model, at 0.845.

The rank-1-on-training model finished mid-pack on data that mattered.

## Why the fancy ones overfit

The sophistication that won on training data was precisely the
sophistication that failed to generalize. The advanced routers
learned fine-grained rules keyed to specific fingerprint regions: "for
data that looks *exactly* like this cluster of training points, dispatch
to that algorithm." Those rules are sharp and accurate inside the
training distribution and meaningless outside it. When the new data's
fingerprint landed in a region the router had no neighbours for, it
extrapolated confidently and wrongly.

The simple v1 router, with its coarse k-NN majority vote, had less to
overfit *with*. And `lmm` doesn't memorize anything; it just fits a
flexible model to whatever data it's handed, so a new regime is just
another fit.

This is the bias-variance tradeoff wearing a clustering costume. I
built seven models climbing the variance axis and was rewarded with a
better training score and a worse real one, every single step.

## Three things I now believe

1. **The "best algorithm: learned_router_v7" headline carries an
   asterisk.** It's the best dispatcher *if your data resembles the
   benchmark's training distribution*. For genuinely unseen data, the
   honest recommendation is the simple v1 router, or just `lmm`
   directly.

2. **The remedy is more coverage, not a cleverer model.** Every router
   I built saturates at the achievable frontier *given its training
   data*. When you hit a new data domain, the move isn't to add a
   smarter routing rule; it's to add that domain to the benchmark and
   let the simple router re-train. (That's exactly what fixed the graph
   gap in Part 4.)

3. **My theoretical performance predictions were systematically
   pessimistic about classical methods.** I'd built "algorithm cards"
   that predict an upper-bound ARI from inductive-bias matching. On
   unseen data, spectral clustering scored 0.775 against a predicted
   ceiling of 0.614. The heuristic underestimated how gracefully old,
   simple, assumption-light algorithms generalize.

## The uncomfortable, useful takeaway

If you're doing AutoML or algorithm selection, your leaderboard is
lying to you by a predictable amount, and the amount grows with model
sophistication. The only honest number is the one from data your
selector has never touched, and you have to go out of your way to
generate it, because nothing in your normal workflow will.

For deployment outside a known distribution, the boring stack wins:
**simple router + lmm + agglomerative + a graph method**, picked
conservatively. I spent weeks building something more elaborate and the
most valuable thing it produced was proof that I didn't need it.

You can reproduce all of this; the unseen config ships in the repo:
https://github.com/Ashail33/clustbench

---

*Part 4: "Training a Model to Pick the Right Clustering Algorithm."
Next, Part 6: "Teaching a Reinforcement-Learning Agent to Cluster
From Primitives."*
