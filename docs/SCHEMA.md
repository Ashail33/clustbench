# Schema

The core record stored for each algorithm run adheres to the following schema (simplified for readability):

- `algo` (string) – name of the algorithm
- `dataset_id` (string) – dataset identifier
- `n_samples` (int) – number of samples
- `n_features` (int) – number of features
- `k_target` (int|null) – target number of clusters
- `compactness` (float) – cluster compactness factor used for synthetic generation
- `seed` (int) – random seed for reproducibility
- `wall_time_s` (float) – wall‑clock time in seconds to run the algorithm
- `rss_delta_mb` (float) – change in resident memory usage (MB)
- `cpu_user_s`, `cpu_system_s` (float) – CPU user and system time (seconds)
- `read_bytes`, `write_bytes` (int|null) – I/O bytes read/written (may be null on some platforms)
- `n_clusters_found` (int|null) – number of clusters detected (ignoring noise)
- `metrics` (object) – bundle of clustering evaluation metrics:
  - `ari`, `nmi` – external validity metrics (adjusted Rand index and normalized mutual information)
  - `silhouette`, `davies_bouldin` – internal validity metrics
  - `compactness`, `separation`, `dunn` – extra metrics (may be null if not computed)
- `extra` (object) – algorithm‑specific metadata (e.g., inertia, BIC)
- `labels_path` (string) – path to saved labels in the `artifacts` directory

This schema is enforced via Pydantic models in `src/clustbench/schemas.py`.
