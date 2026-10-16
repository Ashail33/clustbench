# MCG results (round 20)

`mcg` was run on its own on the same tasks and seeds as the existing results.
All generators are deterministic, so the rows pair one-to-one with:

| file | tasks | pairs with |
|---|---|---|
| `results_paper.json` | 140 (in-distribution) | `docs/data/results.json` |
| `results_unseen.json` | 35 (unseen families) | `docs/data_unseen/results.json` |
| `results_fastgraph.json` | 44 (n = 20k / 100k) | `docs/data_fastgraph/results.json` |
| `results_bigdata.json` | 11 (scaling axes) | `docs/data_bigdata/results.json` |

See `docs/ALGORITHM_ANALYSIS.md`, round 20.
