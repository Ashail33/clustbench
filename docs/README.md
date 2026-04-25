# Clustbench dashboard

A small static dashboard for exploring a clustbench run. Rendered by
GitHub Pages from this `docs/` folder at
<https://ashail33.github.io/clustbench/>.

## What's shown

- **Overview** — counts of algorithms, tasks, and runs; mean ARI and wall time.
- **Algorithm comparison** — bar charts of mean ARI, NMI, silhouette, and wall time per algorithm.
- **Per-task leaderboard** — filterable table of every (algo, task) run.
- **Trajectory viewer** — cost vs. step for iterative algorithms (kmeans, clarans).
  This is the raw state-action data: each point has `step_idx` (action index),
  `cost` (scalar objective), and `delta_cost` (reward signal) that a downstream
  latent-space model or RL policy can train on.

## How to refresh with a new run

```bash
# 1. Run a benchmark
clustbench --config configs/benchmark.paper.demo.yaml --out runs/dashboard

# 2. Convert parquet -> browser JSON
python scripts/build_site.py --run runs/dashboard --out docs/data

# 3. Commit + push; GitHub Pages serves from docs/ on master
git add docs/data configs/benchmark.paper.demo.yaml
git commit -m "Refresh dashboard data"
git push
```

## Enabling GitHub Pages (one-time)

Repo **Settings → Pages → Source**: `Deploy from a branch` / branch `master`
/ folder `/docs`. Save. The site lives at `https://<owner>.github.io/<repo>/`.

## Local preview

```bash
python -m http.server 8000 --directory docs
# open http://localhost:8000/
```
