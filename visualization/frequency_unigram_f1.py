from __future__ import annotations

import argparse
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import pandas as pd


# Word-oriented evaluation after detokenization.
# Keeps letters, digits, apostrophes, hyphens, and common STEM expressions.
TOKEN_RE = re.compile(
    r"""
    [A-Za-zÀ-ÖØ-öø-ÿ]+(?:['’-][A-Za-zÀ-ÖØ-öø-ÿ]+)*
    |
    [+-]?\d+(?:[.,]\d+)?(?:%|°[CF])?
    |
    [A-Za-z]+\d+(?:[A-Za-z0-9_^+\-]*)?
    """,
    flags=re.VERBOSE,
)


def tokenize_words(text: str, lowercase: bool = True) -> list[str]:
    """Tokenize already-detokenized English/Swahili text for unigram analysis."""
    text = str(text)
    if lowercase:
        text = text.lower()
    return TOKEN_RE.findall(text)


def parse_model_file(items: list[str]) -> dict[str, Path]:
    """Parse MODEL=path.csv command-line arguments."""
    output: dict[str, Path] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(
                f"Invalid prediction specification {item!r}. Use MODEL=path.csv."
            )
        model, path = item.split("=", 1)
        model = model.strip()
        if not model:
            raise ValueError(f"Missing model name in {item!r}.")
        output[model] = Path(path)
    return output


def frequency_bin(frequency: int) -> str:
    """
    Stable bins for a low-resource corpus.

    OOV means the reference word was absent from the Swahili training targets.
    """
    if frequency == 0:
        return "OOV"
    if frequency == 1:
        return "1"
    if frequency <= 4:
        return "2–4"
    if frequency <= 9:
        return "5–9"
    if frequency <= 99:
        return "10–99"
    if frequency <= 999:
        return "100–999"
    return "1000+"


BIN_ORDER = ["OOV", "1", "2–4", "5–9", "10–99", "100–999", "1000+"]


def build_training_frequency(
    train_csv: Path,
    target_column: str = "target",
) -> Counter[str]:
    train = pd.read_csv(train_csv)
    if target_column not in train.columns:
        raise KeyError(
            f"{train_csv} does not contain target column {target_column!r}."
        )

    counts: Counter[str] = Counter()
    for sentence in train[target_column].fillna("").astype(str):
        counts.update(tokenize_words(sentence))
    return counts


def clipped_matches(
    reference_counts: Counter[str],
    hypothesis_counts: Counter[str],
) -> Counter[str]:
    """Modified-unigram matches, analogous to clipped BLEU counts."""
    return Counter(
        {
            word: min(count, hypothesis_counts[word])
            for word, count in reference_counts.items()
            if min(count, hypothesis_counts[word]) > 0
        }
    )


def corpus_frequency_f1(
    references: Iterable[str],
    hypotheses: Iterable[str],
    training_frequency: Counter[str],
) -> pd.DataFrame:
    """
    Calculate corpus-level unigram precision, recall, and F1 in each frequency bin.

    Reference occurrences are assigned to bins using target-word frequency in
    the Swahili training corpus. Hypothesis occurrences of known words are
    assigned using the same lookup. Hypothesis words unseen in training enter
    the OOV bin.

    Matching uses clipped corpus counts within each sentence and bin.
    """
    stats = {
        label: {"match": 0, "reference": 0, "hypothesis": 0}
        for label in BIN_ORDER
    }

    for reference, hypothesis in zip(references, hypotheses):
        ref_counts = Counter(tokenize_words(reference))
        hyp_counts = Counter(tokenize_words(hypothesis))
        matches = clipped_matches(ref_counts, hyp_counts)

        for word, count in ref_counts.items():
            label = frequency_bin(training_frequency[word])
            stats[label]["reference"] += count

        for word, count in hyp_counts.items():
            label = frequency_bin(training_frequency[word])
            stats[label]["hypothesis"] += count

        for word, count in matches.items():
            label = frequency_bin(training_frequency[word])
            stats[label]["match"] += count

    rows = []
    for label in BIN_ORDER:
        match = stats[label]["match"]
        ref_total = stats[label]["reference"]
        hyp_total = stats[label]["hypothesis"]

        precision = match / hyp_total if hyp_total else 0.0
        recall = match / ref_total if ref_total else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
        rows.append(
            {
                "frequency_bin": label,
                "matches": match,
                "reference_unigrams": ref_total,
                "hypothesis_unigrams": hyp_total,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        )

    return pd.DataFrame(rows)


def evaluate_models(
    train_csv: Path,
    model_files: dict[str, Path],
    reference_column: str = "target",
    prediction_column: str = "prediction",
) -> pd.DataFrame:
    """Evaluate prediction files, aligning shared test rows when needed."""
    training_frequency = build_training_frequency(train_csv)
    loaded: dict[str, pd.DataFrame] = {}
    key_columns: list[str] | None = None
    shared_keys: set[tuple[str, ...]] | None = None
    expected_references: list[str] | None = None

    for model, prediction_file in model_files.items():
        data = pd.read_csv(prediction_file)

        missing = {
            reference_column,
            prediction_column,
        } - set(data.columns)
        if missing:
            raise KeyError(
                f"{prediction_file} is missing columns: {sorted(missing)}"
            )

        current_key_columns = [
            column
            for column in ("corpus", "line_no", "source", reference_column)
            if column in data.columns
        ]
        if current_key_columns:
            data = data.drop_duplicates(subset=current_key_columns, keep="first")
            data = data.sort_values(current_key_columns).reset_index(drop=True)
            current_keys = set(map(tuple, data[current_key_columns].astype(str).to_numpy()))
            if key_columns is None:
                key_columns = current_key_columns
                shared_keys = current_keys
            elif current_key_columns == key_columns:
                shared_keys = shared_keys & current_keys if shared_keys is not None else current_keys
            else:
                key_columns = []
                shared_keys = None

        references = data[reference_column].fillna("").astype(str).tolist()
        if expected_references is None:
            expected_references = references
        loaded[model] = data

    if len(loaded) > 1 and any(
        data[reference_column].fillna("").astype(str).tolist() != expected_references
        for data in loaded.values()
    ):
        if not key_columns or shared_keys is None or not shared_keys:
            raise ValueError(
                "Prediction files use different test rows and could not be aligned. "
                "Generate predictions with the same --limit and test split."
            )
        for model, data in loaded.items():
            loaded[model] = data[
                data[key_columns].astype(str).apply(tuple, axis=1).isin(shared_keys)
            ].sort_values(key_columns).reset_index(drop=True)
        print(f"Aligned all prediction files to {len(shared_keys)} shared keyed rows.")

    frames = []
    expected_references = None
    for model, data in loaded.items():
        references = data[reference_column].fillna("").astype(str).tolist()
        hypotheses = data[prediction_column].fillna("").astype(str).tolist()
        if expected_references is None:
            expected_references = references
        elif references != expected_references:
            raise ValueError("Aligned prediction files still have different references.")
        scores = corpus_frequency_f1(
            references,
            hypotheses,
            training_frequency,
        )
        scores.insert(0, "model", model)
        frames.append(scores)

    return pd.concat(frames, ignore_index=True)


def plot_f1(scores: pd.DataFrame, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 6))

    for model, group in scores.groupby("model", sort=False):
        group = (
            group.set_index("frequency_bin")
            .reindex(BIN_ORDER)
            .reset_index()
        )
        ax.plot(
            group["frequency_bin"],
            group["f1"] * 100,
            marker="o",
            linewidth=2,
            label=model.replace("_", " + "),
        )

    ax.set_title(
        "English→Swahili unigram F1 by target-word training frequency"
    )
    ax.set_xlabel("Swahili target-word frequency in the training corpus")
    ax.set_ylabel("Unigram F1 (%)")
    ax.set_ylim(bottom=0)
    ax.grid(alpha=0.25)
    ax.legend(
        title="Translation system",
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
    )

    fig.tight_layout()
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument(
        "--prediction-files",
        nargs="+",
        required=True,
        metavar="MODEL=CSV",
    )
    parser.add_argument(
        "--reference-column",
        default="target",
    )
    parser.add_argument(
        "--prediction-column",
        default="prediction",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/figures/unigram_f1_by_frequency.png"),
    )
    parser.add_argument(
        "--scores-output",
        type=Path,
        default=Path("results/unigram_f1_by_frequency.csv"),
    )
    args = parser.parse_args()

    model_files = parse_model_file(args.prediction_files)
    scores = evaluate_models(
        train_csv=args.train,
        model_files=model_files,
        reference_column=args.reference_column,
        prediction_column=args.prediction_column,
    )

    args.scores_output.parent.mkdir(parents=True, exist_ok=True)
    scores.to_csv(args.scores_output, index=False)
    plot_f1(scores, args.output)

    print(f"Saved scores: {args.scores_output}")
    print(f"Saved figure: {args.output}")
    print(
        scores.pivot(
            index="frequency_bin",
            columns="model",
            values="f1",
        )
        .reindex(BIN_ORDER)
        .mul(100)
        .round(2)
        .to_string()
    )


if __name__ == "__main__":
    main()
