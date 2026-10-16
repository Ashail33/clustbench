# MCG results (rounds 20–21)

`mcg` was run on its own on the same tasks and seeds as the existing results.
All generators are deterministic, so the rows pair one-to-one with:

| file | tasks | pairs with |
|---|---|---|
| `results_paper.json` | 140 (in-distribution) | `docs/data/results.json` |
| `results_unseen.json` | 35 (unseen families) | `docs/data_unseen/results.json` |
| `results_fastgraph.json` | 44 (n = 20k / 100k) | `docs/data_fastgraph/results.json` |
| `results_bigdata.json` | 11 (scaling axes) | `docs/data_bigdata/results.json` |

`results_cf_*.json` are the same tasks for `mcg_cf`, the tiered MCG with
CF-triple edge weights (round 21).

See `docs/ALGORITHM_ANALYSIS.md`, rounds 20 and 21.
