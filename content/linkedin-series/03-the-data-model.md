# The data model behind clustbench: turning "clustering research" into a schema

If the recommendation in my thesis was "build a centralised clustering database," the first real design decision was what that database should store. Everything else in the project follows from that one choice, so it is worth walking through the schema in detail.

Clustbench stores four things per benchmark run.

**A task.** Which dataset, at which size, at which dimensionality, at which target k, at which seed. This is the input side of the experiment.

```python
class Task(BaseModel):
    dataset_id: str
    n_samples: int
    n_features: int
    k_target: Optional[int] = None
    compactness: float
    seed: int
```

**A record.** For every (algorithm, task) pair, one row describing what happened. This includes the quality metrics (ARI, NMI, silhouette, Davies-Bouldin) and the resource footprint (wall time, RSS delta, CPU user and system time). Metrics and resources sit in the same row deliberately, because "which algorithm is best" is a joint question about both.

**A step record.** This is the layer I am proudest of. For every iterative algorithm (k-means, CLARANS, agglomerative, spectral, DBSCAN, plus the newer synthesized ones), clustbench records every step as its own row: the step index, the cost at that step, the delta from the previous step, the action taken, and the state of the clustering at that moment. Not just what the algorithm produced. Every move it made getting there.

```python
class StepRecord(BaseModel):
    run_id: str
    algo: str
    dataset_id: str
    step_idx: int
    cost: float
    delta_cost: float | None = None
    accepted: bool = True
    action: Dict[str, Any] = Field(default_factory=dict)
    state: Dict[str, Any] = Field(default_factory=dict)
```

That single table is the reason for the whole project. When you record every step of every iterative algorithm, the optimisation process itself becomes training data. You can train a model to propose better next steps. You can measure which mechanisms actually reduce cost on which data. You can reconstruct why an algorithm failed, not just that it did. My earlier posts flagged this as the one genuinely novel piece of clustbench compared to existing benchmark efforts. Prior benchmarks store outcomes. This one stores process.

**A dashboard contract.** Four JSON files: `manifest.json` (what ran), `results.json` (per-task metrics), `summary.json` (per-algorithm ranks and Friedman statistics), `trajectories.json` (the per-step records). The GitHub Pages dashboard reads exactly these four files, so anyone with a browser can inspect the current state of the benchmark without cloning anything.

Two design choices worth calling out.

First, schemas are typed. Everything goes through Pydantic models, so a malformed run cannot silently poison the database. If someone adds an algorithm that returns strings where floats are expected, the run fails at write time, not at analysis time three weeks later.

Second, the same schema serves both storage and analysis. The results file is what the benchmark writes, what the dashboard reads, and what the learned routers train on. There is no ETL layer. That is deliberate. Every additional transformation is a place where meaning drifts.

Once the schema was stable, everything else followed: adding an algorithm became a one-line config change, adding a dataset became a one-file addition, and the dashboard automatically absorbed whatever the last commit produced. That is the topic of the next post: the pipeline.

Repo: https://github.com/Ashail33/clustbench
Live dashboard: https://ashail33.github.io/clustbench/
