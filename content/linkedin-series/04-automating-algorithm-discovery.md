# How I automated clustering algorithm discovery: the pipeline

By the time the schema was stable, clustbench had four moving parts. This post walks through them in order, because the sequence is the whole point.

**One: the dataset generators.** Every dataset in the benchmark is a Python function of one argument (a `DataSpec` with sample size, dimensionality, target k, compactness, seed) returning a matrix and its ground-truth labels. That single signature covers synthetic regimes (MDCGen-style Gaussians, moons, circles, anisotropic blobs), sklearn-bundled real data (iris, wine, breast cancer, digits), time-series feature stacks, graph node-feature matrices, and my more recent additions: hierarchical nested Gaussians, imbalanced blobs at 10-to-1 ratio, heavy-tailed mixtures, and two deliberately-adversarial datasets designed to break specific algorithm classes. Nineteen dataset configurations at the current count. Adding a new one is one file plus one config entry.

**Two: the algorithm registry.** Each algorithm is a class inheriting from a single `Algorithm` base, decorated with `@register("name")`. That decorator populates a global registry the benchmark iterates over. Forty-seven algorithms at the current count, covering the classical baselines, sklearn extras, research methods from the review, improved variants, three synthesized families (aura, meta_clusterer, rapid, each versioned v1 to v3), eight learned routers, and the last few round-14 and round-15 additions. Adding a new algorithm is one file plus one config entry.

**Three: the harness.** For every (task, algorithm, seed) combination, run the algorithm, measure quality (ARI, NMI, silhouette, Davies-Bouldin), measure resources (wall time, memory delta, CPU), write a single row to `results.csv`. Any per-step trajectory the algorithm chose to emit lands in a per-run parquet under `artifacts/`. Errors are trapped per-task and written as sentinel rows, so one algorithm crashing on one dataset does not sink the whole run. This is where the schema pays for itself: every row is validated, every artefact has a stable name, every path is reproducible from the input triple.

**Four: the dashboard.** A build script reads the results, computes per-algorithm ranks, runs a Friedman test, and writes four JSON files to `docs/data/`. A static HTML page in `docs/index.html` reads those JSON files and renders leaderboards, per-dataset heatmaps, complexity plots, and a trajectory viewer that lets you step through any algorithm's optimisation path. GitHub Pages deploys the whole thing on every push to master.

The end result: every commit rebuilds the entire evidence base and re-publishes it. There is no separate paper-writing step. There is no separate results-freezing step. If I add an algorithm today, the leaderboard updates when the next benchmark completes, and any downstream analysis (like the learned routers) picks up the new results automatically.

The most useful property of this pipeline is not any single component. It is that the marginal cost of testing a new idea is close to zero. If I want to know whether a new algorithm beats the current field, I add one file, add one config line, wait fifteen minutes for the benchmark, and read the answer. I have used this to test dozens of algorithm variants I would never have written otherwise, because the cost of being wrong dropped to almost nothing. Most were wrong. A few were surprisingly good. All of them are honestly ranked.

The next post is what happens when you let this pipeline compound over 15 rounds.

Repo: https://github.com/Ashail33/clustbench
Live dashboard: https://ashail33.github.io/clustbench/
