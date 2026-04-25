"""Split a clustbench run into per-algorithm result files."""

from __future__ import annotations

import argparse
import json
import pathlib

import pandas as pd


def _load_results(run_dir: pathlib.Path) -> pd.DataFrame:
    parquet = run_dir / "results.parquet"
    if parquet.exists():
        return pd.read_parquet(parquet)
    return pd.read_csv(run_dir / "results.csv")


def export(run_dir: pathlib.Path) -> None:
    df = _load_results(run_dir)
    by_algo = run_dir / "by_algo"
    by_algo.mkdir(parents=True, exist_ok=True)

    index = {}
    for algo, sub in df.groupby("algo", sort=True):
        algo_dir = by_algo / str(algo)
        algo_dir.mkdir(parents=True, exist_ok=True)
        sub = sub.copy()
        sub.to_csv(algo_dir / "results.csv", index=False)
        try:
            sub.to_parquet(algo_dir / "results.parquet", index=False)
        except Exception:
            pass

        artifacts = []
        for _, row in sub.iterrows():
            for col in ("labels_path", "trajectory_path"):
                value = row.get(col)
                if isinstance(value, str) and value:
                    artifacts.append(value)
        index[str(algo)] = {
            "rows": int(len(sub)),
            "results_csv": str(algo_dir / "results.csv"),
            "results_parquet": str(algo_dir / "results.parquet"),
            "artifacts": artifacts,
        }

    (by_algo / "index.json").write_text(json.dumps(index, indent=2, default=str))
    print(f"Wrote per-algorithm exports for {len(index)} algorithms to {by_algo}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=pathlib.Path, required=True)
    args = parser.parse_args()
    export(args.run)


if __name__ == "__main__":
    main()
