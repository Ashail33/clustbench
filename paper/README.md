# Papers

| paper | source | PDF |
|---|---|---|
| **Review of Big Data Clustering Methods** (MEng research assignment, Stellenbosch, 2024) | — | [`clustering-review-maharaj-2024.pdf`](clustering-review-maharaj-2024.pdf) |
| **Clustbench: A Reproducible, Trajectory-Instrumented, Self-Extending Benchmark for Clustering Algorithms** | [`clustbench/clustbench.tex`](clustbench/clustbench.tex) | [`clustbench/clustbench.pdf`](clustbench/clustbench.pdf) |
| **Synthesising Clustering Algorithms from Benchmark Failure Modes** (FMM/LMM/AMM, TBMM/FLHMM, MRIC, RAPID, AURA, META-CLUSTERER, `louvain_knn`, learned routers, `rl_pipeline`, `lmm_fast`, the MCG micro-cluster graph family) | [`algorithms/algorithms.tex`](algorithms/algorithms.tex) | [`algorithms/algorithms.pdf`](algorithms/algorithms.pdf) |

The two 2026 papers are companions: the first describes the harness and what
it measures, the second the algorithms built with it. They share
[`references.bib`](references.bib) and the figures in [`figures/`](figures/).

## Rebuilding

Every figure and every final-snapshot number is computed from the data
committed under `docs/`: `data/` (in-distribution, 61 algorithms),
`data_unseen/` (out-of-distribution, 62), `data_scaling/` and `data_bigdata/`
(scaling), `data_fastgraph/` (graph clustering at n = 100k) and `data_shapes/`
(every algorithm on every dataset). The manifold and gallery figures
recompute eigenvectors and labels from the generators, so `make figures`
needs the `clustbench` package installed:

```bash
pip install -e .   # clustbench, plus matplotlib
make -C paper figures   # regenerate paper/figures/*.pdf
make -C paper           # pdflatex + bibtex for both papers
```

Tables marked "round N" in the algorithms paper are taken from the round in
which that version was introduced, as recorded in
[`docs/ALGORITHM_ANALYSIS.md`](../docs/ALGORITHM_ANALYSIS.md).
