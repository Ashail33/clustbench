# clustbench content strategy

A 7-part series for Medium (long-form canon) syndicated to LinkedIn
(hooks + distribution). Each Medium article spawns 2–4 LinkedIn posts —
write once, atomize once.

## Core thesis
"I turned my master's thesis on clustering into a living benchmark that
builds, breaks, and routes between 44 algorithms — then tried to teach
an RL agent to cluster from scratch."

The credibility comes from the honest negative results, not the wins.

## The series
1. `01-the-setup.md` — paper → reproducible benchmark
2. `02-where-algorithms-break.md` — per-family bottleneck reference
3. `03-synthesizing-algorithms.md` — composing new methods from analysis
4. `04-the-learned-router.md` — meta-learning algorithm selection
5. `05-my-best-model-overfit.md` — the OOD humility post (viral candidate)
6. `06-teaching-an-rl-agent-to-cluster.md` — Framing C / rl_pipeline
7. `07-lessons-building-a-self-extending-benchmark.md` — capstone

`linkedin-posts.md` holds the atomized hooks for each article.

## Cadence
- Medium: 1 deep post / 1–2 weeks, in series order.
- LinkedIn: 2–3 posts / week, drawn from the current Medium piece.
- Publish Medium then syndicate the link same morning (LinkedIn +
  r/MachineLearning `[P]` tag). Tue–Thu, 8–10am.

## Assets to attach every time
- Live dashboard: https://ashail33.github.io/clustbench/
- Repo: https://github.com/Ashail33/clustbench
- A screenshot of the interactive chart beats any static plot.

## Numbers of record (from runs/paper_demo_r13, 43 algos × 15 configs × 3 seeds)
- Top mean ARI: learned_router_v7 0.877, v5/v3 0.867, v4 0.860, v1 0.816
- Classical: spectral 0.765, gmm 0.745, agglomerative 0.717, kmeans 0.681,
  birch 0.633, meanshift 0.387, optics 0.145, dbscan 0.078
- rl_pipeline: rank 36/43, mean ARI 0.584
- OOD (benchmark.unseen.yaml): v7 0.879→0.753 (−0.126), v1 0.817→0.840
  (+0.023), lmm 0.78→0.845 (+0.065), louvain_knn 0.769→0.825 (+0.056)
