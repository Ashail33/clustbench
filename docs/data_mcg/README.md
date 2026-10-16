# MCG results (rounds 20–22)

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

`results_lmm_mcg_*.json` are `lmm_mcg`, LMM's EM on the `mcg_cf` graph basis
(round 22). `failures_lmm_mcg_bigdata.json` holds its one run that did not
finish: a MemoryError at k = 200, which plain LMM also hits.

See `docs/ALGORITHM_ANALYSIS.md`, rounds 20–22.

These files predate round 24's adaptive micro-cluster cap. The canonical
MCG rows in `docs/data`, `docs/data_unseen` and `docs/data_shapes` are the
round-24 re-runs. Tasks where the cap does not bind (at least 30 points per
cluster) give identical results.
