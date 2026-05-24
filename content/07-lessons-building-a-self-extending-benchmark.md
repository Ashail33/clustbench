---
title: "What I Learned Building a Self-Extending ML Benchmark"
subtitle: "Six months of turning a thesis into a living system: the methodology that worked, the results that humbled me, and what infrastructure actually buys you."
series: "Benchmarking 44 Clustering Algorithms — Part 7 (Finale)"
tags: [machine-learning, benchmarking, research, mlops, data-science]
---

# What I Learned Building a Self-Extending ML Benchmark

This series started with a frozen table at the end of a master's thesis
and ended with a reinforcement-learning agent that learned to cluster
from primitives. Forty-three algorithms, fifteen dataset regimes, a
live dashboard, and a stack of results — some of which made me look
clever and some of which made me look foolish, which is exactly the
right ratio.

Here's what the whole thing taught me.

## 1. Infrastructure changes what questions you can ask

The single highest-leverage decision was making the benchmark
**re-run itself on every commit** and publish a live dashboard. That
sounds like ops hygiene. It's actually epistemics.

When re-running the full comparison costs nothing, you stop asking "is
this algorithm good?" and start asking "good *compared to what,
where, and does that hold up when I change the data?*" Every idea in
Parts 3–6 — synthesizing algorithms, learned routing, the RL agent —
was only tractable because adding a candidate and seeing it ranked
against 42 others was a one-line config change, not a research project.

Cheap evaluation is a force multiplier on the *number of ideas you're
willing to test*, including the bad ones. Most of mine were bad. The
benchmark told me which, fast.

## 2. The self-extending loop is the best part

The pattern I'm proudest of: the learned routers (Part 4) improve
without new code. Add a capability to the registry — like the
graph-native `louvain_knn` — re-run the benchmark, and the selection
layer *automatically* learns to dispatch to it where it wins. Mean ARI
lifted +0.03 with zero changes to the router.

That's a system that gets better as you feed it, not as you re-engineer
it. The lesson generalizes: **build the layer that learns from results,
then make producing results cheap.** The intelligence accrues in the
data, not the code.

## 3. Your leaderboard is lying, by an amount that grows with cleverness

The hardest lesson (Part 5). I built seven increasingly sophisticated
routers. Each one scored better on the training distribution than the
last. Each one *also* generalized worse — the top model dropped 13 ARI
points on unseen data and lost to a plain mixture model from the 2000s.

Sophistication bought training-set accuracy and paid for it in
generalization, monotonically, every step. If you do algorithm
selection, AutoML, or meta-learning, the only number you can trust is
the one from data your system has never touched — and you have to build
that test on purpose, because your normal workflow will never produce
it for you.

## 4. Honest negative results are the most valuable output

Three of my best findings were failures:

- Composed algorithms tie the best classical methods but don't beat
  them — the component frontier is real (Part 3).
- The fanciest router overfits; simple + classical wins out of
  distribution (Part 5).
- The RL agent learns individual moves but not multi-step sequences —
  the behavior-cloning ceiling (Part 6).

None of these are what I hoped for. All of them are more useful than a
win would have been, because they tell you *where the frontier is* and
*which direction has slack left in it.* A benchmark that only produces
wins is a benchmark you've stopped believing.

## 5. Name your prior art, then find the one new thing

When I did a literature search, most of what I'd built turned out to
have prior art — meta-learning for algorithm selection, landmarking,
stacked generalization, all decades old. That stung for about a day,
and then it clarified the project: clustbench *operationalizes* an
established direction (the centralized, reproducible clustering DB my
thesis argued for) on fresh, broad coverage.

The one piece that appears genuinely novel is the **trajectory layer** —
capturing every algorithm's optimization as state-action data, and
training a policy on it directly. The RL agent (Part 6) is the first
evidence that layer is trainable. That's where I'd put more time.

## Where this goes next

- **PPO on top of the behavior-cloned policy** — teach the agent the
  multi-step grammar it's currently missing, scored by terminal
  cluster quality instead of per-step imitation.
- **Bigger trajectory datasets** — every benchmark run now produces
  them automatically, so the training set for a step-proposer grows for
  free.
- **Learned bottleneck components** — replace specific weak steps (the
  CLARANS swap proposal, the chameleon merge criterion) with models
  trained on the trajectory data, as Part 2 mapped out.

## If you take one thing

Build the boring infrastructure first. A benchmark that re-runs itself,
records everything, and ranks every candidate against every other made
the difference between "I have opinions about clustering algorithms"
and "I have evidence, including the evidence that I was wrong." The
results came and went. The infrastructure is what compounded.

Everything — code, dashboard, the 419-row trajectory dataset, every
algorithm version, and the original thesis — is open:
https://github.com/Ashail33/clustbench
https://ashail33.github.io/clustbench/

Thanks for reading the series.

---

*Part 6: "Teaching a Reinforcement-Learning Agent to Cluster From
Primitives." This is the finale.*
