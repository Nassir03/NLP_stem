from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


DATA_DIR = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = DATA_DIR / "parallel_corpus_clean.csv"
DEFAULT_OUTPUT = DATA_DIR / "reports" / "token_length_report.json"


def percentile(values: list[int], point: float) -> int:
    """Calculate a simple percentile for sequence-length decisions."""
    if not values:
        return 0
    values = sorted(values)
    return values[round((len(values) - 1) * point)]


def analyze_lengths(input_path: Path) -> dict[str, object]:
    english_lengths: list[int] = []
    swahili_lengths: list[int] = []

    with input_path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            english_lengths.append(len(row["english"].split()))
            swahili_lengths.append(len(row["swahili"].split()))

    return {
        "note": (
            "These are whitespace-token estimates for inspection only. "
            "During model training, use each pretrained model's own tokenizer."
        ),
        "recommended_start_max_source_length": 256,
        "recommended_start_max_target_length": 256,
        "english_word_percentiles": {
            "p50": percentile(english_lengths, 0.50),
            "p90": percentile(english_lengths, 0.90),
            "p95": percentile(english_lengths, 0.95),
            "p99": percentile(english_lengths, 0.99),
        },
        "swahili_word_percentiles": {
            "p50": percentile(swahili_lengths, 0.50),
            "p90": percentile(swahili_lengths, 0.90),
            "p95": percentile(swahili_lengths, 0.95),
            "p99": percentile(swahili_lengths, 0.99),
        },
        "model_tokenizers_to_use": {
            "marian": "Helsinki-NLP/opus-mt-en-sw tokenizer",
            "nllb": "facebook/nllb-200-distilled-600M tokenizer with eng_Latn -> swh_Latn",
            "mt5": "google/mt5-small tokenizer with a translate English to Swahili prefix",
            "byt5": "google/byt5-small byte tokenizer",
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect token lengths before model training.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = analyze_lengths(args.input)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Token length report written to {args.output.resolve()}")


if __name__ == "__main__":
    main()
