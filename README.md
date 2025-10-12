# Clustbench

Clustbench is a language‑agnostic benchmark harness for clustering algorithms.  
It can generate synthetic datasets, run multiple clustering algorithms, and save metrics and resource usage in a structured data model.  
You can plug in new algorithms (written in Python or any language via a simple JSON protocol) and evaluate them on configurable datasets.

## Quick start

Create a virtual environment and install the package locally:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .
```

Run a small benchmark:

```bash
clustbench --config configs/benchmark.sample.yaml --out runs/demo
```

This will generate a synthetic dataset, run the configured algorithms, and save results to `runs/demo/`:

- `results.parquet` / `results.csv` – per‑run metrics and resource usage
- `summary.json` – average ranks and Friedman test per metric
- `artifacts/` – per‑algorithm label arrays and metric JSON

## Extending

- **Datasets:** Add a new generator function in `src/clustbench/datasets.py` and register it in the `DATASETS` dict.
- **Algorithms:** Subclass `Algorithm` in `src/clustbench/algorithms/base.py` and register it with the `register` decorator. Implement a `fit_predict` method that accepts `X` and `k` and returns labels and extra metadata.
- **External algorithms:** Provide an executable that reads a JSON payload on stdin, writes a JSON response to stdout, and writes a `.npy` file with predicted labels. See `src/clustbench/runners/external_runner.py` for details.

The data model and schema are documented in `docs/`.
