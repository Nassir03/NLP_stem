from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path


DATA_DIR = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = DATA_DIR / "parallel_corpus_clean.csv"
DEFAULT_OUTPUT = DATA_DIR / "reports" / "frequent_words.csv"

# Important note for this translation project:
# We do not remove stopwords from the training pairs because words such as
# "of", "to", "kwa", "ya" and "wa" carry grammar needed by NMT models.
# This file only reports frequent words for inspection.


def count_words(input_path: Path, limit: int) -> list[tuple[str, str, int]]:
    counts = {"english": Counter(), "swahili": Counter()}
    with input_path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            counts["english"].update(word.lower().strip(".,;:!?()[]{}\"'") for word in row["english"].split())
            counts["swahili"].update(word.lower().strip(".,;:!?()[]{}\"'") for word in row["swahili"].split())

    results: list[tuple[str, str, int]] = []
    for language, counter in counts.items():
        for word, count in counter.most_common(limit):
            if word:
                results.append((language, word, count))
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Report frequent words without removing stopwords.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--limit", type=int, default=100)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rows = count_words(args.input, args.limit)

    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["language", "word", "count"])
        writer.writerows(rows)

    print(f"Frequent-word report written to {args.output.resolve()}")


if __name__ == "__main__":
    main()
