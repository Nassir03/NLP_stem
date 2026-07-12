from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path


DATA_DIR = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = DATA_DIR / "parallel_corpus_clean.csv"
DEFAULT_OUTPUT = DATA_DIR / "processed" / "segmented_pairs.csv"

# Common abbreviations that should not end a sentence during segmentation.
ABBREVIATIONS = {
    "dr.",
    "mr.",
    "mrs.",
    "ms.",
    "prof.",
    "fig.",
    "eq.",
    "e.g.",
    "i.e.",
    "etc.",
}


def should_split(text: str, index: int) -> bool:
    """Return True when punctuation at index is likely to end a sentence."""
    char = text[index]
    if char not in ".!?":
        return False

    # Avoid splitting decimal numbers such as 3.14.
    before = text[index - 1] if index > 0 else ""
    after = text[index + 1] if index + 1 < len(text) else ""
    if before.isdigit() and after.isdigit():
        return False

    # Avoid splitting known abbreviations such as Fig. or Eq.
    prefix = text[max(0, index - 8) : index + 1].lower().split()[-1]
    if prefix in ABBREVIATIONS:
        return False

    # Split only when followed by whitespace, a closing mark, or the end.
    if index + 1 == len(text):
        return True
    closing_marks = set("\"')]}")
    return after.isspace() or after in closing_marks


def segment_text(text: str) -> list[str]:
    """Split one paragraph into sentences while preserving STEM expressions."""
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []

    segments: list[str] = []
    start = 0
    for index, _ in enumerate(text):
        if should_split(text, index):
            segment = text[start : index + 1].strip()
            if segment:
                segments.append(segment)
            start = index + 1

    tail = text[start:].strip()
    if tail:
        segments.append(tail)
    return segments


def segment_parallel_row(row: dict[str, str]) -> list[dict[str, str]]:
    """Segment a pair only when both sides produce the same number of sentences."""
    english_segments = segment_text(row["english"])
    swahili_segments = segment_text(row["swahili"])

    if len(english_segments) != len(swahili_segments) or len(english_segments) <= 1:
        return [row]

    segmented_rows = []
    for sentence_index, (english, swahili) in enumerate(zip(english_segments, swahili_segments), start=1):
        segmented_rows.append(
            {
                "source": row["source"],
                "row_number": f"{row['row_number']}.{sentence_index}",
                "english": english,
                "swahili": swahili,
            }
        )
    return segmented_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Segment aligned paragraph pairs into sentence pairs.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    total_input = 0
    total_output = 0
    with args.input.open("r", encoding="utf-8", newline="") as input_handle, args.output.open(
        "w", encoding="utf-8", newline=""
    ) as output_handle:
        reader = csv.DictReader(input_handle)
        writer = csv.DictWriter(output_handle, fieldnames=["source", "row_number", "english", "swahili"])
        writer.writeheader()

        for row in reader:
            total_input += 1
            for segmented_row in segment_parallel_row(row):
                writer.writerow(segmented_row)
                total_output += 1

    print(f"Read {total_input} aligned rows.")
    print(f"Wrote {total_output} segmented rows to {args.output.resolve()}")


if __name__ == "__main__":
    main()
