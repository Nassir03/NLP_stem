from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import CFG


def parse_model_file(items: list[str]) -> dict[str, Path]:
    output: dict[str, Path] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(
                f"Invalid history specification {item!r}. Use MODEL=path.csv."
            )
        model, path = item.split("=", 1)
        output[model.strip()] = Path(path)
    return output


def discover_history_files(results_dir: Path = CFG.results_dir) -> dict[str, Path]:
    """Find all training-history CSVs produced by the no-generation runner."""
    histories = [
        path
        for path in sorted(results_dir.glob("*_training_history.csv"))
        if not path.name.startswith("combined_")
    ]
    if not histories:
        raise FileNotFoundError(f"No training history files found in {results_dir}")
    return {
        path.name.removesuffix("_training_history.csv"): path
        for path in histories
    }


def load_histories(model_files: dict[str, Path]) -> pd.DataFrame:
    """Load model history CSVs from either current or older column names."""
    frames = []

    for model, path in model_files.items():
        history = pd.read_csv(path)
        if "valid_loss" in history.columns and "validation_loss" not in history.columns:
            history = history.rename(columns={"valid_loss": "validation_loss"})

        required = {"epoch", "validation_loss"}
        missing = required - set(history.columns)
        if missing:
            raise KeyError(
                f"{path} is missing columns: {sorted(missing)}"
            )

        selected = history[["epoch", "validation_loss"]].copy()
        selected["model"] = model
        frames.append(selected)

    return pd.concat(frames, ignore_index=True)


def plot_histories(histories: pd.DataFrame, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 6))

    for model, group in histories.groupby("model", sort=False):
        group = group.sort_values("epoch")
        ax.plot(
            group["epoch"],
            group["validation_loss"],
            marker="o",
            linewidth=2,
            label=model.replace("_", " + "),
        )

    ax.set_title("Validation cross-entropy by training epoch")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation cross-entropy")
    ax.grid(alpha=0.25)
    ax.legend(
        title="Neural MT model",
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
    )

    # Integer epoch ticks when the total number of epochs is manageable.
    max_epoch = int(histories["epoch"].max())
    if max_epoch <= 30:
        ax.set_xticks(range(1, max_epoch + 1))

    fig.tight_layout()
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--history-files",
        nargs="+",
        metavar="MODEL=CSV",
        help="Optional MODEL=CSV history files. Defaults to all results/*_training_history.csv.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "results/figures/validation_cross_entropy_by_epoch.png"
        ),
    )
    parser.add_argument(
        "--combined-output",
        type=Path,
        default=Path("results/combined_training_history.csv"),
    )
    args = parser.parse_args()

    model_files = (
        parse_model_file(args.history_files)
        if args.history_files
        else discover_history_files()
    )
    histories = load_histories(model_files)
    args.combined_output.parent.mkdir(parents=True, exist_ok=True)
    histories.to_csv(args.combined_output, index=False)
    plot_histories(histories, args.output)

    print(f"Saved combined history: {args.combined_output}")
    print(f"Saved figure: {args.output}")


if __name__ == "__main__":
    main()
