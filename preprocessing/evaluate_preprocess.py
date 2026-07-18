from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from config import CFG


def split_stats(path: Path) -> dict[str, float | int | str]:
    """Summarize a prepared split so data issues are visible before training."""
    df = pd.read_csv(path)
    src_lengths = df.source.astype(str).str.split().str.len()
    tgt_lengths = df.target.astype(str).str.split().str.len()
    duplicate_pairs = int(df.duplicated(subset=["source", "target"]).sum())
    return {
        "split": path.stem,
        "rows": int(len(df)),
        "source_mean_words": float(src_lengths.mean()) if len(df) else 0.0,
        "target_mean_words": float(tgt_lengths.mean()) if len(df) else 0.0,
        "source_max_words": int(src_lengths.max()) if len(df) else 0,
        "target_max_words": int(tgt_lengths.max()) if len(df) else 0,
        "duplicate_pairs": duplicate_pairs,
    }


def evaluate_preprocessing() -> pd.DataFrame:
    """Create a project-level preprocessing report from the prepared CSV files."""
    CFG.results_dir.mkdir(parents=True, exist_ok=True)
    required = [CFG.split_dir / f"{name}.csv" for name in ("train", "validation", "test")]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Prepared split files are missing. Run `python main.py prepare` first. "
            f"Missing: {', '.join(missing)}"
        )

    report = pd.DataFrame(split_stats(path) for path in required)
    report.to_csv(CFG.results_dir / "preprocessing_split_report.csv", index=False)

    total = int(report.rows.sum())
    expected = {
        "train": CFG.train_ratio,
        "validation": CFG.valid_ratio,
        "test": CFG.test_ratio,
    }
    report["actual_ratio"] = report["rows"] / max(total, 1)
    report["expected_ratio"] = report["split"].map(expected)
    print(report.to_string(index=False))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.parse_args()
    evaluate_preprocessing()
