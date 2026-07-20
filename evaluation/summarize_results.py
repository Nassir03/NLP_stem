from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from config import CFG


def summarize_results(results_dir: Path = CFG.results_dir) -> pd.DataFrame:
    """Combine generated prediction metric CSV files into one sorted comparison table."""
    metric_files = [
        path
        for path in sorted(results_dir.glob("*_metrics.csv"))
        if "_stem_" not in path.name
    ]
    if not metric_files:
        print(f"No generated metric files found in {results_dir}; using training histories.")
        return summarize_training_histories(results_dir)

    frames = []
    for path in metric_files:
        df = pd.read_csv(path)
        if not {"bleu", "chrf++", "ter"}.issubset(df.columns):
            print(f"Skipping {path}; not a translation metric file.")
            continue
        df["file"] = path.name
        frames.append(df)
    if not frames:
        raise ValueError(f"No usable translation metric files found in {results_dir}")

    summary = pd.concat(frames, ignore_index=True, sort=False)
    if "bleu" in summary.columns:
        summary = summary.sort_values("bleu", ascending=False).reset_index(drop=True)

    out_path = results_dir / "all_model_metrics_summary.csv"
    summary.to_csv(out_path, index=False)
    print(summary.to_string(index=False))
    print(f"Saved summary: {out_path}")
    summarize_stem_scores(results_dir)
    return summary


def summarize_stem_scores(results_dir: Path = CFG.results_dir) -> pd.DataFrame | None:
    """Combine per-model STEM preservation scores for report Table 3."""
    score_files = sorted(results_dir.glob("*_stem_scores.csv"))
    if not score_files:
        print(f"No STEM score files found in {results_dir}.")
        return None

    frames = []
    for path in score_files:
        df = pd.read_csv(path)
        if {"model", "metric", "score"} - set(df.columns):
            print(f"Skipping {path}; missing model/metric/score columns.")
            continue
        frames.append(df)
    if not frames:
        return None

    scores = pd.concat(frames, ignore_index=True)
    scores = scores.drop_duplicates(subset=["model", "metric"], keep="last")
    summary = (
        scores.pivot(index="model", columns="metric", values="score")
        .reset_index()
        .rename(columns={
            "term_accuracy": "term",
            "number_accuracy": "number",
            "symbol_accuracy": "symbol",
            "unit_accuracy": "unit",
        })
    )
    ordered_columns = ["model", "term", "number", "symbol", "unit"]
    for column in ordered_columns:
        if column not in summary.columns:
            summary[column] = pd.NA
    summary = summary[ordered_columns]
    out_path = results_dir / "all_model_stem_summary.csv"
    summary.to_csv(out_path, index=False)
    print(summary.to_string(index=False))
    print(f"Saved STEM summary: {out_path}")
    return summary


def summarize_training_histories(results_dir: Path = CFG.results_dir) -> pd.DataFrame:
    """Rank trained neural models by their best validation loss without decoding."""
    history_files = [
        path
        for path in sorted(results_dir.glob("*_training_history.csv"))
        if not path.name.startswith("combined_")
    ]
    if not history_files:
        raise FileNotFoundError(f"No training history files found in {results_dir}")

    rows = []
    for path in history_files:
        df = pd.read_csv(path)
        if df.empty:
            continue
        if "valid_loss" not in df.columns and "validation_loss" in df.columns:
            df = df.rename(columns={"validation_loss": "valid_loss"})
        if "valid_ppl" not in df.columns:
            df["valid_ppl"] = float("nan")
        if "train_loss" not in df.columns and "training_loss" in df.columns:
            df = df.rename(columns={"training_loss": "train_loss"})
        missing = {"epoch", "valid_loss", "train_loss"} - set(df.columns)
        if missing:
            print(f"Skipping {path}; missing columns: {sorted(missing)}")
            continue
        best = df.loc[df["valid_loss"].idxmin()]
        model = path.name.removesuffix("_training_history.csv")
        rows.append({
            "model": model,
            "best_epoch": int(best["epoch"]),
            "best_valid_loss": float(best["valid_loss"]),
            "best_valid_ppl": float(best["valid_ppl"]),
            "final_train_loss": float(df.iloc[-1]["train_loss"]),
            "history_file": path.name,
        })

    if not rows:
        raise ValueError(f"No usable training history files found in {results_dir}")

    summary = pd.DataFrame(rows).sort_values("best_valid_loss").reset_index(drop=True)
    out_path = results_dir / "all_model_training_summary.csv"
    summary.to_csv(out_path, index=False)
    print(summary.to_string(index=False))
    print(f"Saved training summary: {out_path}")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--training",
        action="store_true",
        help="Summarize training histories instead of generated prediction metrics.",
    )
    args = parser.parse_args()
    if args.training:
        summarize_training_histories()
    else:
        summarize_results()
