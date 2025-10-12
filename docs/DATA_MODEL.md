# Data Model

Clustbench uses a structured data model to persist results.  Each benchmark run writes the following files under the output directory:

| File | Description |
| --- | --- |
| `manifest.json` | Metadata about the run (config, generation time, environment) |
| `results.parquet` / `results.csv` | Tidy table where each row corresponds to a single algorithm × dataset × seed combination with metrics and resource usage |
| `artifacts/labels_<algo>.npy` | NumPy array of predicted labels for each algorithm |
| `artifacts/metrics_<algo>.json` | JSON document containing the flattened record for that algorithm |
| `summary.json` | Summary statistics, including average ranks and Friedman test results |

The `results` table includes metrics (ARI, NMI, silhouette, Davies–Bouldin) and resource metrics (wall time, memory delta, CPU time). See `docs/SCHEMA.md` for details.
