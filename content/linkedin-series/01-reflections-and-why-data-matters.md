# Reflections on my master's, and why data infrastructure matters more than the model

A few years ago I finished a master's on big-data clustering. The formal title was *Review of Big Data Clustering Methods*, submitted in 2024 under Prof. Andries Engelbrecht. I spent months reading, running, and comparing clustering algorithms across the four Vs (volume, velocity, variety, veracity). At the end I had the usual review artefact: a taxonomy, a set of experiments, a comparison table, some conclusions.

And a nagging feeling I hadn't quite finished the job.

The feeling was this. Every paper I read reported different metrics on different datasets at different sample sizes with different hyperparameters. Almost no one reported runtime or memory in a way I could reproduce. So even after writing my own comparison, I couldn't say with confidence to a practitioner "here's the algorithm to use for your data" because there was no shared surface to compare against. The literature was a stack of one-off snapshots. Mine was another one.

The recommendation in my conclusion was that the field needed a centralised clustering database. Not a paper. A living resource where any new algorithm could be dropped in, run against the same battery of datasets, and compared against everything else. If we had that, someone with real-world data could look up the closest matching regime and get an honest recommendation.

I moved on and started working. But the idea kept coming back, and it kept looking bigger the longer I looked at it. Because what I had actually recommended wasn't really about clustering. It was about the thing I now think matters more than any model choice in applied AI: the data infrastructure underneath.

The models get all the attention. The infrastructure decides what questions you can even ask. If your evaluation costs a week to re-run, you evaluate rarely and trust your gut. If your evaluation costs seconds, you evaluate constantly and trust the evidence. If your results live in a hundred forgotten notebooks, no one can build on them. If they live in one schema with one contract, everyone can.

This applies far beyond clustering. Every ML team I have spoken to has some version of the same story: the model was easy, the data pipeline was hard, and the pipeline is what compounded. The intelligence, in the end, accrues in the data, not in the code.

So a few months ago I started building the thing my thesis had recommended. I called it clustbench. It runs 47 clustering algorithms across 19 dataset regimes on every commit, publishes a live dashboard, records every step of every iterative algorithm as a state-action trajectory, and has been running for 15 rounds of iterative discovery. It has already produced results I did not expect, including several honest negatives that changed my mind.

This series is about how I built it, what I learned, and why the boring infrastructure layer turned out to be the interesting one.

Repo: https://github.com/Ashail33/clustbench
Live dashboard: https://ashail33.github.io/clustbench/

Next post: the centralised database I proposed in the thesis, and why nobody had built one.
