"""CLI entrypoint for clustbench."""

from __future__ import annotations
import argparse
import pathlib

import yaml

from ..benchmark import run_benchmark


def main() -> None:
    parser = argparse.ArgumentParser(description="Run clustbench benchmark")
    parser.add_argument("--config", type=pathlib.Path, required=True)
    parser.add_argument("--out", type=pathlib.Path, required=True)
    parser.add_argument(
        "--firebase-bucket",
        default=None,
        help="Firebase Storage bucket name (e.g. my-project.appspot.com). "
             "When provided, results are uploaded to Firebase after writing locally.",
    )
    parser.add_argument(
        "--firebase-credentials",
        default=None,
        help="Path to a Firebase service-account JSON key file. "
             "Falls back to GOOGLE_APPLICATION_CREDENTIALS env var or ADC.",
    )
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text())

    df = run_benchmark(
        cfg,
        args.out,
        firebase_bucket=args.firebase_bucket,
        firebase_credentials=args.firebase_credentials,
    )
    print(f"Wrote results to {args.out} ({len(df)} rows)")

