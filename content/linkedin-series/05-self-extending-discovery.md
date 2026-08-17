# Fifteen rounds later: a self-extending clustering discovery system

I have run 15 rounds of iteration on clustbench so far. This is the post where I try to be honest about what actually worked, what did not, and why the whole thing kept improving even when specific ideas failed.

The short version: the pipeline compounds because the intelligence lives in the data, not in any single algorithm. Here is what that looks like in practice.

**Round 4** introduced the learned router. Nearest-neighbours over data fingerprints, where the label is "the algorithm that won on similar data before." Established meta-learning idea, fresh benchmark. It went to the top of the leaderboard immediately.

**Round 11** demonstrated the self-extending property. I added a graph-native algorithm (louvain_knn) to the registry to close a gap on the karate-club dataset. I did not touch the router code. On the next benchmark run, the router automatically started dispatching to louvain_knn on graph data. Mean ARI lifted 0.03 with zero router changes. The router improved because the data it trained on improved.

**Round 12** was the humility round. I built a deliberately out-of-distribution benchmark, sample sizes and dimensionalities and outlier extremities the routers had never seen. My top router dropped 13 ARI points on unseen data and lost to a plain Laplacian mixture model. The simpler routers actually gained. This is the bias-variance tradeoff in a clustering costume: sophistication bought training-set accuracy and paid for it in generalisation, monotonically, every step.

**Round 13** built a reinforcement-learning agent that clusters by chaining primitive actions (kpp_init, assign_to_centers, update_centers, eigen_embed, and five others). Behavior-cloning on 419 trajectory rows, all captured by the per-step schema I described in the earlier post. The agent learned, from imitation alone, to fire the spectral embedding move on concentric-ring data and reached ARI 1.0 on circles at 300 points. At 500 points it hit a wall because 419 traces are not enough to teach coordinated multi-step sequences. Honest ceiling. Real proof that the trajectory layer is trainable.

**Round 14** ran a multi-agent workflow. Three discovery agents proposed new algorithms, new datasets, and adversarial datasets in parallel. Eight implementation agents wrote the code. One integration agent wired it in. One benchmark agent ran the sweep. One report agent wrote the first draft of the analysis. Fourteen agents, 17 minutes, three new algorithms and five new datasets. All landed mid-pack. The interesting result was that the adversarial datasets successfully found the router's out-of-distribution corner: routers under-perform non-routers by 0.04 to 0.07 on those, and by roughly zero on the general ones.

**Round 15** designed a new algorithm (trident) from the round-14 winner analysis, and a new router (v8) from the round-14 failure analysis. Both landed. Both regressed on average. The v8 router with an enriched fingerprint and an OOD-triggered ensemble fallback dropped 0.02 versus v7. trident's silhouette gate keeps picking k-means on non-convex data. Two honest negative results in one round. Meanwhile v7's own score moved from 0.871 in round 14 to 0.912 in round 15 without any code change, purely from auto-retraining on the richer dataset.

The pattern is clear. Adding coverage helps. Adding sophistication does not. The routers have saturated on this benchmark. The next real move is either broader coverage (bigger n, more real-world domains) or the RL trajectory work.

Fifteen rounds in, the system finds better algorithms without me writing router code. What I wrote is the pipeline that makes finding them cheap. That is the master's thesis recommendation, actually built.

Repo: https://github.com/Ashail33/clustbench
Live dashboard: https://ashail33.github.io/clustbench/
