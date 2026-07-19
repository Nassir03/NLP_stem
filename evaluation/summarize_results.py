from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from config import CFG


def summarize_results(results_dir: Path = CFG.results_dir) -> pd.DataFrame:
    """Combine all metric CSV files into one sorted comparison table."""
    metric_files = sorted(results_dir.glob("*_metrics.csv"))
    if not metric_files:
        raise FileNotFoundError(f"No metric files found in {results_dir}")

    frames = []
    for path in metric_files:
        df = pd.read_csv(path)
        df["file"] = path.name
        frames.append(df)

    summary = pd.concat(frames, ignore_index=True, sort=False)
    if "bleu" in summary.columns:
        summary = summary.sort_values("bleu", ascending=False).reset_index(drop=True)

    out_path = results_dir / "all_model_metrics_summary.csv"
    summary.to_csv(out_path, index=False)
    print(summary.to_string(index=False))
    print(f"Saved summary: {out_path}")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.parse_args()
    summarize_results()
