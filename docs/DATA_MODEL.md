# Data Model

Clustbench uses a structured data model to persist results.  Each benchmark run writes the following files under the output directory:

| File | Description |
| --- | --- |
| `manifest.json` | Metadata about the run (config, generation time, environment) |
| `results.parquet` / `results.csv` | Tidy table where each row corresponds to a single algorithm × dataset × seed × compactness combination with metrics and resource usage |
| `artifacts/labels_<algo>__<task_suffix>.npy` | NumPy array of predicted labels for each algorithm run |
| `artifacts/metrics_<algo>__<task_suffix>.json` | JSON document containing the flattened record for that run |
| `artifacts/trajectory_<algo>__<task_suffix>.parquet` | Per-step trajectory for iterative algorithms (currently kmeans, clarans) — one row per step with state, action, cost, delta_cost |
| `summary.json` | Summary statistics, including average ranks and Friedman test results |

`<task_suffix>` encodes the dataset identifiers so artifacts from different tasks don't collide: `<dataset_id>_n<n_samples>_d<n_features>_k<k_target>_c<compactness>_s<seed>`.

The `results` table includes metrics (ARI, NMI, silhouette, Davies–Bouldin), resource metrics (wall time, memory delta, CPU time), and — when present — `n_steps` and `trajectory_path` pointing to the trajectory table.

## Trajectory table

Each row in a trajectory parquet file is a :class:`StepRecord` and contains:

- `run_id` — UUID for this specific algorithm run
- task identifiers (`algo`, `dataset_id`, `n_samples`, `n_features`, `k_target`, `compactness`, `seed`)
- `step_idx` — monotonic counter within the run
- `cost` — the current objective (lower is better)
- `delta_cost` — change in cost from the previous accepted state (null on first step / inits)
- `accepted` — whether the proposed action was accepted
- `action` — dict describing the action (e.g., `{"type": "swap", "out": 17, "in": 42}`)
- `state` — numeric state snapshot (e.g., `{"medoids": [...]}` or `{"centroids": [[...]]}`)

This schema is designed to feed downstream latent-space / state-action models: concatenate trajectories across runs, encode `state` via an autoencoder, and learn a policy over `(state, action) → delta_cost`.

See `docs/SCHEMA.md` for details.
