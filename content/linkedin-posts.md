# LinkedIn posts (atomized from the Medium series)

Each block is one post. Lead line is the hook (it's all most people see
before "…more"). Attach a dashboard screenshot where noted. End every
post with the Medium link + the repo.

Hashtags to rotate: #MachineLearning #Clustering #DataScience
#Benchmarking #AutoML #ReinforcementLearning #MLOps

---

## For Part 1: The setup

**Post 1a (launch)**
> Everyone benchmarks clustering on iris and two moons. I ran 43
> algorithms across 15 dataset regimes, and made the whole thing
> re-run itself on every commit.

My master's thesis ended with the usual frozen comparison table: this
method is fast, that one handles outliers. Useful for a day, stale
forever.

So I rebuilt it as clustbench, a benchmark that generates data with
known ground truth, runs every algorithm through the same harness, and
publishes a live dashboard on every push. The numbers are never
hand-curated; they're whatever the last commit produced.

Part 1 of a 7-part series on what I found. Link in comments.
[attach: dashboard screenshot]

**Post 1b (a finding)**
> The algorithm that needs the most tuning looks worst in an untuned
> benchmark.

DBSCAN and OPTICS sit at the bottom of my clustering leaderboard (ARI
0.08 and 0.15). Not because they're bad, but because I used one `eps`
across 15 wildly different dataset geometries, and density methods live
or die by that one parameter.

That's a finding, not a bug. A fair benchmark has to decide: do you
tune per-dataset (rewards fiddly methods) or fix config (rewards robust
ones)? I chose fixed, and I report it loudly.

---

## For Part 2: Where algorithms break

**Post 2a**
> "k-means is bad at non-convex shapes" is folklore. Here's the actual
> reason, and it's one line of the algorithm.

Every centroid method assigns each point to its nearest center. That
makes cluster boundaries straight lines. A ring inside a ring can't be
split by any straight line through the centers, so k-means scores ARI
≈ 0 on concentric circles. Literally no better than random.

It's not a tuning problem. It's baked into the assignment step. The fix
is to change the *space* (spectral embedding) before you cluster, not
the parameters.

Full per-algorithm breakdown in Part 2 👇

**Post 2b**
> Outliers wreck almost every clustering algorithm for the same single
> reason: they all compute a mean somewhere, and a mean has unbounded
> influence per point.

k-means centroids, Ward's variance, BIRCH summaries, mean-shift modes:
all means. One far-away point drags them.

The exception is GMM, and it's instructive: GMM weights its mean update
by posterior responsibility, so an outlier that fits no component gets
near-zero weight. That one design choice is why it's the most
outlier-robust centroid method in my benchmark.

The fix menu (medoids, trimmed means, pre-filtering) in Part 2.

---

## For Part 3: Synthesizing algorithms

**Post 3a**
> If GMM owns outlier-robustness and spectral owns non-convex shapes,
> what happens when you compose the winning mechanisms into new
> algorithms?

I built three (adaptive routing, a stacker, an outlier-first method),
versioned them v1→v3, and ran them against everything.

Result: they tie the best classical methods. They don't beat them.

That's the honest finding: composing winning mechanisms gets you to
the component frontier; it doesn't move it. Which is exactly what
pushed me toward *selection* instead of composition. Part 3 👇

---

## For Part 4: The learned router

**Post 4a**
> Instead of one best clustering algorithm, I trained a model to read
> your data and pick the right one. It went straight to the top of the
> leaderboard.

It's not a magic deep net. It's k-nearest-neighbours over data
fingerprints, where the label is "the algorithm that won on similar
data before." Established idea (meta-learning / landmarking), fresh
benchmark.

Best version beat the best single classical algorithm by +0.11 ARI.

And the best part: when I added a new algorithm later, the router
learned to use it with ZERO code changes, just re-running the
benchmark. A self-extending system. Part 4 👇

---

## For Part 5: THE VIRAL CANDIDATE

**Post 5a (the main one, push this hardest)**
> I spent weeks building 7 versions of a learned model to pick the best
> clustering algorithm. Then I tested it on data it had never seen. It
> lost to a plain mixture model from the 2000s.
>
> It's the most useful result I got. Here's why. 🧵

My #1-ranked router scored 0.879 on the training distribution. On
genuinely out-of-distribution data (new sizes, dims, k, outlier
extremities, far-off seeds) it dropped to 0.753, a 13-point fall.

The pattern was brutal and clean: the MORE sophisticated the router,
the HARDER it fell. The four fanciest models each lost 11–13 ARI
points. The SIMPLEST router actually gained. And the overall winner on
unseen data wasn't a router at all: it was `lmm`, a basic mixture
model.

The sophistication that won on training data was exactly the
sophistication that failed to generalize. Bias-variance tradeoff in a
clustering costume.

Three takeaways that apply to anyone doing AutoML or model selection:
1. Your leaderboard lies, by an amount that GROWS with model
   sophistication.
2. The fix is more data coverage, not a cleverer model.
3. The only honest number comes from data your system has never
   touched, and you have to build that test on purpose.

I built something elaborate and its most valuable output was proof I
didn't need it. Full write-up + reproducible config 👇
[attach: the trained-vs-unseen table as an image]

**Post 5b (shorter reshare a few days later)**
> Reminder that your validation score and your real-world score are
> different numbers, and the gap grows with how clever your model is.
>
> Seven routers. Each scored better than the last on training data.
> Each generalized worse. The chart that taught me this 👇
[attach: same table]

---

## For Part 6: The RL agent

**Post 6a**
> I broke clustering into 8 primitive moves and tried to teach an agent
> to chain them. From imitation alone, it rediscovered the core trick
> of spectral clustering.

The agent learned (with no hand-coded knowledge that spectral methods
solve rings) that when the cost surface won't drop, the right move is
to re-embed via the graph Laplacian. On concentric circles it hit ARI
1.0, choosing `eigen_embed` exactly when k-means would have failed.

A learned policy treating "switch to a spectral embedding" as a
conditional move. Part 6 👇

**Post 6b (the honest follow-up)**
> Yesterday I said my RL clustering agent learned the spectral trick.
> Today, the wall it hit, because the failure is more interesting than
> the win.

The agent aced concentric circles at 300 points. At 500 points it
scored ZERO on the same shape. It still fires the right first move
(`eigen_embed`), but it never learned the multi-step follow-up to
finish the job, so the trajectory dies in no-ops.

That's the behavior-cloning ceiling: 419 demonstration traces teach
individual moves but not coordinated sequences. BC learns the
vocabulary; it can't learn the grammar. The fix is reinforcement
learning that scores the whole episode, not each step: the next stage.

Reporting the ceiling, not just the win. Part 6 👇

---

## For Part 7: The finale

**Post 7a**
> 6 months turning a thesis into a living ML benchmark. The biggest
> lesson wasn't an algorithm: it was that infrastructure changes which
> questions you can even ask.

When re-running a 43-algorithm comparison costs nothing, you stop
asking "is this good?" and start asking "good compared to what, where,
and does it survive a distribution shift?"

Three of my best findings were failures. All more useful than a win
would have been. The results came and went; the infrastructure is what
compounded.

Series finale 👇

**Post 7b (reflective / personal-brand)**
> When I did the literature search, most of what I'd built already
> existed: meta-learning, landmarking, stacking, all decades old. That
> stung for a day. Then it clarified the work.

I wasn't inventing the genre; I was operationalizing it on broad,
reproducible coverage, plus one piece that does look new: capturing
every algorithm's optimization as state-action data and training a
policy on it directly.

Naming your prior art honestly is how you find the one thing that's
actually yours. Finale 👇
