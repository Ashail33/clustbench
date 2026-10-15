# Prior-art check

A literature search (arXiv via alphaXiv, plus web search) run in October 2026
on the algorithms and ideas in the two papers under `paper/`. It is a
targeted check, not a systematic review. "Not found" means not found in
these searches, not proven new.

| contribution | closest prior art found | verdict |
|---|---|---|
| **LMM** (EM mixture over graph-Laplacian eigenvectors) | Korenblum, *Laplacian mixture modeling for network analysis and unsupervised learning on graphs*, PLOS ONE 2018 ([arXiv:1502.00727](https://arxiv.org/abs/1502.00727)); also spectral embedding + GMM pipelines, e.g. [arXiv:1610.08227](https://arxiv.org/abs/1610.08227) | **The name and the core idea are taken.** Korenblum combines Laplacian eigenspaces with finite mixture modelling, fitted by global optimisation of a linear transform of the eigenvectors. Ours differs in the fitting: per-cluster exponential-family log-densities on the eigenvector basis, EM with self-normalisation over the data, and a learned heat-kernel diffusion time per cluster. Present it as a variant, not a new model class. Consider renaming it, e.g. "heat-kernel exponential-family mixture". |
| **FMM** (mixture of random-Fourier exponential families) | EM for exponential-family mixtures: Banerjee et al., *Clustering with Bregman Divergences*, JMLR 2005. Kernel exponential families: Sriperumbudur et al., JMLR 2017. RFF kernel clustering: Chitta, Jin & Jain, ICDM 2012 | Ingredients are known. The exact combination (RFF exponential-family components, log-partition self-normalised over the sample, Newton-step EM) was not found. Make only a weak novelty claim. |
| **AMM** (autoencoder basis) | Deep clustering (encode, then cluster) is a large literature | A variant of a standard pattern. No novelty claim. |
| **MRIC** (MRI-relaxation clustering) | Only k-means applied *to* MRI relaxation maps | No prior clustering method of this kind found. |
| **Learned routers** | Rice 1976; de Souto et al. 2008; Ferrari & de Castro 2015; ML2DAC (Treder-Tschechlov et al., SIGMOD 2023); CLAMS ([arXiv:2407.11286](https://arxiv.org/abs/2407.11286)); ClustRecNet ([arXiv:2509.25289](https://arxiv.org/abs/2509.25289)) | Established problem with active recent systems. Our contribution is the controlled evaluation (LOFO, unseen split). Direct comparison with ML2DAC/CLAMS/ClustRecNet is missing. |
| **rl_pipeline** (agent composing clustering from primitives, behaviour-cloned) | Searches surfaced RL for grouping/trajectory clustering, not this | Not found. The trajectory dataset plus imitation-trained composer looks distinctive, but the search was shallow. |
| **Clustbench (the harness)** | Gagolewski, *A framework for benchmarking clustering algorithms*, SoftwareX 2022 | **Name collision:** Gagolewski's PyPI package `clustering-benchmarks` is imported as `clustbench`. A rename, or at least a prominent note, is advisable before publishing. |
| MDCGen citation | Iglesias, Zseby, Ferreira & Zimek, *MDCGen: Multidimensional Dataset Generator for Clustering*, J. Classification 36:599–618, 2019 | Confirmed. The thesis/README attribution to López, Luna & Herrera is wrong. |

## Authorship note

Per `git log`, the algorithm implementations (FMM, LMM, AMM, MRIC on
2026-05-11, and every later algorithm) were committed by Claude Code
sessions run in this repository. The original scaffold (October 2025) was
committed by Ashail33. Both papers now carry an AI-assistance
acknowledgement.
