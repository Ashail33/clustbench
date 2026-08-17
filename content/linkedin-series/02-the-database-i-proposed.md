# The centralised clustering database I proposed in my master's, and why nobody had built one

In my thesis I recommended building a centralised database for clustering research. Any new algorithm could be dropped in, evaluated against the same datasets under the same metrics, and compared honestly against everything else. The paper closed with that recommendation. I did not build it at the time. Neither, as far as I could find, had anyone else.

That absence surprised me. The value proposition is obvious. So why does it not exist?

The honest answer is that a database is not a paper. Nobody gets promoted for shipping infrastructure. You get promoted for the novel algorithm you built on top of infrastructure someone else provided. So the incentive structure quietly ensures that the underlying resource is always someone else's problem. In clustering specifically, there are benchmark suites (like the University of Eastern Finland's clustering datasets, or the ELKI project's collection), but they are static file archives. Not a running system. Not something that regenerates its numbers on every commit. Not something a new algorithm can be added to in one afternoon.

The second reason is that a good clustering benchmark actually is genuinely hard to design. Real-world clustering has no single ground truth. The Adjusted Rand Index needs known labels. Internal metrics like silhouette can be gamed by any algorithm that produces compact clusters, even when the compact clusters are wrong. Runtime and memory depend on the machine. Sample size, dimensionality, cluster count, outlier fraction, noise level, and cluster shape all interact. You cannot just publish "the best algorithm." You have to publish which algorithm wins on which regime, and that requires a matrix, not a scalar.

The third reason is that the algorithms themselves are moving targets. sklearn's implementation of DBSCAN changes between versions. So does k-means with k-means++ initialisation. So does GMM's regularisation default. A benchmark that ran once in 2019 is quietly wrong by 2024. The only cure is a benchmark that re-runs itself on every commit against a pinned environment.

None of these problems are fatal. They just add up to enough friction that nobody had done it end to end. My master's could argue for the database. It could not build it. Not because I was time-constrained, though I was. Because the moment you commit to building it properly, you commit to the boring infrastructure part first, and the boring part takes months before it produces anything a supervisor can grade.

So the recommendation sat there. I finished, I graduated, I worked. Then, a couple of years later, I decided to just do it.

The idea was not "let me improve the state of the art in clustering." It was "let me build the missing shared surface, so any improvement anyone else makes can be measured against everything." What I did not expect was that once the surface existed, the improvements would follow it naturally, mostly from me, but also from an agent-driven discovery loop I did not know I would end up wiring in.

The next post is the schema. What the database actually stores, why those specific columns, and where the genuinely new idea (a per-step trajectory layer) came from.

Repo: https://github.com/Ashail33/clustbench
Live dashboard: https://ashail33.github.io/clustbench/
