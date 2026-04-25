# Clustbench

A language-agnostic benchmark harness for clustering algorithms, with a
trajectory layer that captures every step of an iterative algorithm as a
`(state, action, cost, delta_cost)` record so the optimization process
itself can be modeled.

- Live dashboard: <https://ashail33.github.io/clustbench/>
- Sample run (96 rows, 48 trajectories) ships in `docs/data/`

## Why this exists

Clustering papers are notoriously hard to compare across. Different
authors report different metrics on different datasets at different
sample sizes with different hyperparameters, and almost nobody reports
runtime or memory in a way that's reproducible. Clustbench is the
boring infrastructure that fixes the lower half of that problem:

1. Generate **synthetic datasets with known ground truth** at sample
   sizes you control.
2. Run a registry of **clustering algorithms** — Python-native or any
   external executable that speaks a tiny JSON protocol — on the same
   data.
3. Capture **internal validity, external validity, and resource use**
   (wall time, memory delta, CPU time, I/O bytes) in a single tidy
   table per run.
4. Aggregate across seeds with **average ranks and a Friedman test**, so
   "is algorithm A actually better than B?" becomes a one-liner.

The non-boring half is what motivates the trajectory layer.

## The trajectory / state-action layer

Most clustering algorithms are presented as a black box that maps
`X → labels`. But the iterative ones — k-means, k-medoids / CLARANS,
mean shift, density-peaks, EM-GMM — are sequential decision processes:
at each step they pick an *action* (reassign points, swap a medoid, move
a centroid) from a *state* (current centroids / medoids / cluster
assignment) and see a *reward* (drop in the objective).

Clustbench instruments these algorithms so every step is a row in a
trajectory table:

| field         | example                              | role                          |
| ------------- | ------------------------------------ | ----------------------------- |
| `step_idx`    | 7                                    | action index                  |
| `state`       | `{"centroids": [[...], [...]]}`      | observation                   |
| `action`      | `{"type": "swap", "out": 17, "in": 42}` | action                     |
| `cost`        | `52414.57`                           | scalar objective              |
| `delta_cost`  | `-101.19`                            | reward signal                 |
| `accepted`    | `true`                               | whether the action was taken  |

Concatenate trajectories across many runs and the result is the raw
material for:

- **Latent state spaces** — autoencode or contrastively embed
  serialized states; trajectories become smooth curves in the latent
  space.
- **Value / policy models over `(state, action)`** — predict
  `delta_cost` from the action, then bias the search toward
  high-improvement actions instead of random sampling.
- **Algorithm meta-search** — once states and actions are in a common
  space, hybrids (run k-means EM until cost stabilizes, then switch to
  CLARANS-style swaps) become learnable.

This is the angle the project is being built around. Currently
**k-means** and **CLARANS** emit trajectories; the schema is generic
enough to plug any iterative algorithm into.

## Paper

**Maharaj, A. (2024). *Review of Big Data Clustering Methods*.** MEng
(Structured, Industrial Engineering) research assignment, Faculty of
Engineering, Stellenbosch University. Supervisor: Prof. A.P. Engelbrecht.

PDF: [`paper/clustering-review-maharaj-2024.pdf`](paper/clustering-review-maharaj-2024.pdf)

> In an era defined by the challenges of processing vast and complex
> datasets, the study delves into the evolving landscape of big data
> clustering. It introduces a novel taxonomy categorizing clustering
> models into four distinct groups, offering a roadmap for understanding
> their scalability and efficiency in the face of increasing data volume
> and complexity. […] Insights from this research highlighted the
> scalability and efficiency of models like parallel k-means and
> mini-batch k-means, both theoretically and empirically, marking them
> as exemplary for large-scale applications. Conversely, it unveiled
> the computational constraints of models like selective sampling-based
> scalable sparse subspace clustering (S⁵C) and purity-weighted
> consensus clustering (PWCC), showing their limitations in scaling to
> big data. […] It lays the foundation for a centralized database for
> clustering research, aiming to fill existing knowledge gaps and
> facilitate optimal model discovery tailored to specific needs and
> infrastructural capabilities.
>
> — Abstract

The paper's stated final contribution — *"a centralized database for
clustering research [...] to facilitate optimal model discovery"* — is
what this codebase makes operational. The trajectory layer extends that
contribution: rather than only comparing final outputs, clustbench now
captures the *process* each algorithm takes to reach those outputs, so
the next generation of meta-algorithms can learn from the search itself.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e '.[test]'

# Tiny smoke run
clustbench --config configs/benchmark.sample.yaml --out runs/demo

# Larger sweep that populates the dashboard
clustbench --config configs/benchmark.dashboard.yaml --out runs/dashboard
python scripts/build_site.py --run runs/dashboard --out docs/data
```

After a run, `runs/demo/` contains:

| path                                              | what it is                                                                 |
| ------------------------------------------------- | -------------------------------------------------------------------------- |
| `manifest.json`                                   | the config, generation timestamp, Python/platform info                     |
| `results.parquet` / `results.csv`                 | one row per (algo, dataset, n, d, k, compactness, seed)                    |
| `summary.json`                                    | average ranks per metric and Friedman χ² across algorithms                 |
| `artifacts/labels_<algo>__<task>.npy`             | predicted labels                                                           |
| `artifacts/metrics_<algo>__<task>.json`           | the full Record (Pydantic) for that run                                    |
| `artifacts/trajectory_<algo>__<task>.parquet`     | step-by-step trajectory (only for instrumented algorithms)                 |

## Configuration

A benchmark is defined by a YAML file. Cartesian product over every
list value gives the task grid.

```yaml
datasets:
  - id: blobs
    n_samples: [500, 2000]
    n_features: [8]
    k_targets: [3, 5]
    compactness: [0.5, 1.0]

algorithms:
  - name: kmeans
    kind: python
    entry: kmeans          # key in ALGO_REGISTRY
    params:
      max_iter: 100
      n_init: 3
  - name: clarans
    kind: python
    entry: clarans
    params:
      numlocal: 2
      maxneigh: 40
  - name: my_external
    kind: external
    entry: ./bin/run_my_algo   # any executable on PATH
    params:
      flavor: "fast"

seeds: [1, 2, 3]
```

`kind: external` invokes the executable via the JSON-over-stdin protocol
documented in `src/clustbench/runners/external_runner.py`.

## Algorithms

| name              | kind         | trajectory     | notes                                                                     |
| ----------------- | ------------ | -------------- | ------------------------------------------------------------------------- |
| `kmeans`          | python       | yes            | manual EM loop captures centroids per iteration                           |
| `minibatch_kmeans`| python (sklearn wrapper) | no | fast baseline                                                             |
| `dbscan`          | python (sklearn wrapper) | no | density-based, ignores `k`                                                |
| `birch_algo`      | python (sklearn wrapper) | no | hierarchical                                                              |
| `clarans`         | python (custom) | yes         | k-medoids with random local search                                        |
| `consensus`       | python (meta)| no             | majority vote over a list of base algorithms                              |
| `parallel_kmeans` | python (custom) | yes         | Zhao 2009 MapReduce-style: per-iteration map/reduce via multiprocessing   |
| `pwcc`            | python (meta) | yes           | Alguliyev purity-weighted consensus; one step per base + final vote       |
| `s5c`             | python (custom) | yes         | Matsushima selective-sampling sparse subspace clustering (OMP + spectral) |
| `gmm`             | python (sklearn wrapper) | no | Gaussian-mixture EM                                                       |
| `agglomerative`   | python (sklearn wrapper) | no | hierarchical, Ward linkage by default                                     |
| `spectral`        | python (sklearn wrapper) | no | spectral clustering on a kNN affinity                                     |
| `meanshift`       | python (sklearn wrapper) | no | non-parametric in `k`; bandwidth auto-estimated                           |
| `optics`          | python (sklearn wrapper) | no | density-based, ignores `k`                                                |

To add one of your own, see `src/clustbench/algorithms/base.py` —
subclass `Algorithm`, decorate with `@register`, return an `AlgoResult`
(optionally with a `trajectory: list[Step]`).

## Datasets

| id       | generator                                           | knobs                                                |
| -------- | --------------------------------------------------- | ---------------------------------------------------- |
| `blobs`  | `sklearn.datasets.make_blobs`                       | n, d, k, compactness                                 |
| `mixed`  | `sklearn.datasets.make_classification`              | n, d, k, compactness                                 |
| `mdcgen` | MDCGen-style Gaussian mixture (Lopez et al., reproduced) | n, d, k, compactness, **outliers**, **noise**, **density** |
| `moons`  | `sklearn.datasets.make_moons` (non-convex) | n, compactness (drives noise) |
| `circles`| `sklearn.datasets.make_circles` (concentric) | n, compactness (drives noise) |
| `anisotropic` | sheared blobs | n, d, k, compactness |

Add more in `src/clustbench/datasets.py` and register them in `DATASETS`.

### Recreating the paper experiments

`configs/benchmark.paper.yaml` is the full grid from Maharaj (2024)
(2 outliers × 3 noise × 3 clusters × 5 sizes × 3 features × 3 density
× 3 seeds × 4 algorithms — the same shape that hit Colab Pro+ limits in
the paper). `configs/benchmark.paper.demo.yaml` is a downsampled version
that fits the same shape into a couple of minutes:

```bash
clustbench --config configs/benchmark.paper.demo.yaml --out runs/paper_demo
python scripts/build_site.py --run runs/paper_demo --out docs/data
```

The four paper algorithms (one per taxonomy category) are now
in-tree: `parallel_kmeans` (parallel/distributed), `minibatch_kmeans`
(incremental), `pwcc` (ensemble), and `s5c` (sampling/partitioning).
All except minibatch emit trajectories, so the dashboard's trajectory
panel turns into a side-by-side view of how each category's search
process actually unfolds.

## Metrics

Computed by `src/clustbench/metrics.py:bundle_scores`:

- **External validity** (need ground truth): ARI, NMI
- **Internal validity**: silhouette, Davies-Bouldin
- **Custom shape metrics**: cluster compactness, cluster separation, Dunn index

Resource metrics from `psutil`: wall time, RSS delta, CPU user/system,
read/write bytes (where the platform supports it).

`results.csv` is flat: every metric is a top-level column, so a one-liner
in pandas / DuckDB / etc. gets you a leaderboard.

## Dashboard

`docs/index.html` is a static, dependency-free dashboard (Chart.js from
CDN) that reads `docs/data/results.json` and `docs/data/trajectories.json`.
GitHub Pages serves it from the `/docs` folder on `master`.

Refresh after a new run:

```bash
python scripts/build_site.py --run runs/dashboard --out docs/data
git add docs/data && git commit -m "Refresh dashboard" && git push
```

Local preview:

```bash
python -m http.server 8000 --directory docs
```

## Project layout

```
clustbench/
├── README.md                       <- this file
├── pyproject.toml
├── configs/
│   ├── benchmark.sample.yaml       <- minimal smoke config
│   └── benchmark.dashboard.yaml    <- 24-task grid that backs the dashboard
├── docs/
│   ├── index.html                  <- static dashboard
│   ├── data/                       <- JSON the dashboard fetches
│   ├── DATA_MODEL.md
│   ├── SCHEMA.md
│   └── README.md
├── scripts/
│   ├── build_site.py               <- run dir -> dashboard JSON
│   └── report_summary.py           <- matplotlib report from results.parquet
├── src/clustbench/
│   ├── __init__.py
│   ├── benchmark.py                <- core run loop + resource measurement
│   ├── cli/run.py                  <- `clustbench` CLI entrypoint
│   ├── datasets.py
│   ├── metrics.py
│   ├── schemas.py                  <- Pydantic models (Record, StepRecord, ...)
│   ├── consensus.py                <- consensus meta-algorithm
│   ├── algorithms/
│   │   ├── base.py                 <- Algorithm ABC, AlgoResult, Step, register()
│   │   ├── kmeans.py               <- instrumented EM loop
│   │   ├── clarans.py              <- instrumented local search
│   │   ├── minibatch_kmeans.py
│   │   ├── dbscan.py
│   │   └── birch.py
│   └── runners/
│       └── external_runner.py      <- JSON-over-stdin protocol
├── tests/
│   └── test_smoke.py               <- end-to-end smoke + registry coverage
└── notebooks/
    └── clustbench_test.ipynb
```

## Tests

```bash
pip install -e '.[test]'
pytest -q
```

Currently five tests: registry coverage, CLI end-to-end on a tiny
config, external-runner round-trip via a shebang-ed Python script, and
a ranks/Friedman sanity check.

## External algorithm protocol

Any executable can plug in by reading one JSON object from stdin:

```json
{
  "data_path":   "/abs/path/X.npy",
  "k":           5,
  "params":      {"flavor": "fast"},
  "artifacts_dir": "/abs/path/artifacts",
  "labels_path": "/abs/path/artifacts/labels_external_<entry>.npy"
}
```

…and writing the predicted integer labels to `labels_path` (NumPy
`.npy`). It may also print a JSON object to stdout:

```json
{"extra": {"inertia": 1234.5, "iters": 17}}
```

That's the entire protocol. See
`src/clustbench/runners/external_runner.py` and the round-trip test in
`tests/test_smoke.py`.

## Roadmap

- More instrumented algorithms (mean shift, GMM-EM, density-peaks)
- DuckDB view that joins `results.parquet` + every `trajectory_*.parquet`
- Latent-space trainer: autoencoder over serialized states + a
  `(state, action) → delta_cost` value model, with a notebook in
  `notebooks/`
- Real datasets adapter (UCI, OpenML)

## License

Add a license file when ready (likely MIT or BSD-3 to match the
scientific-Python ecosystem). Until then, the work is © Ashail Maharaj.

## Author

Ashail Maharaj — <https://github.com/Ashail33>
