from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path


DATA_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT = DATA_DIR / "parallel_corpus_clean.csv"
WHITESPACE_RE = re.compile(r"\s+")


def clean_text(text: str) -> str:
    """Normalize a corpus line without changing its meaning."""
    return WHITESPACE_RE.sub(" ", text.replace("\ufeff", "")).strip()


def find_parallel_pairs(data_dir: Path) -> list[tuple[str, Path, Path]]:
    pairs: list[tuple[str, Path, Path]] = []

    for english_file in sorted(data_dir.glob("*.en")):
        stem = english_file.name[: -len(".en")]
        swahili_file = data_dir / f"{stem}.sw"

        if swahili_file.exists():
            pairs.append((stem, english_file, swahili_file))

    return pairs


def iter_clean_rows(
    pairs: list[tuple[str, Path, Path]],
    *,
    skip_empty: bool = True,
) -> tuple[list[dict[str, str]], dict[str, int]]:
    rows: list[dict[str, str]] = []
    stats = {
        "pairs": len(pairs),
        "written": 0,
        "skipped_empty": 0,
        "line_mismatches": 0,
    }

    for source, english_file, swahili_file in pairs:
        with english_file.open("r", encoding="utf-8-sig", errors="replace") as en_handle:
            english_lines = en_handle.readlines()

        with swahili_file.open("r", encoding="utf-8-sig", errors="replace") as sw_handle:
            swahili_lines = sw_handle.readlines()

        if len(english_lines) != len(swahili_lines):
            stats["line_mismatches"] += 1
            print(
                f"Warning: {source} has {len(english_lines)} English lines and "
                f"{len(swahili_lines)} Swahili lines. Using the shortest length."
            )

        for row_number, (english, swahili) in enumerate(
            zip(english_lines, swahili_lines),
            start=1,
        ):
            english_text = clean_text(english)
            swahili_text = clean_text(swahili)

            if skip_empty and (not english_text or not swahili_text):
                stats["skipped_empty"] += 1
                continue

            rows.append(
                {
                    "source": source,
                    "row_number": str(row_number),
                    "english": english_text,
                    "swahili": swahili_text,
                }
            )
            stats["written"] += 1

    return rows, stats


def write_csv(rows: list[dict[str, str]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8", newline="") as csv_handle:
        writer = csv.DictWriter(
            csv_handle,
            fieldnames=["source", "row_number", "english", "swahili"],
        )
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert paired English/Swahili corpus files into a clean CSV."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DATA_DIR,
        help="Directory containing paired files ending in .en and .sw.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="CSV file to create.",
    )
    parser.add_argument(
        "--keep-empty",
        action="store_true",
        help="Keep rows where either side of the translation pair is empty.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir.resolve()
    output_path = args.output.resolve()

    pairs = find_parallel_pairs(data_dir)
    if not pairs:
        raise SystemExit(f"No .en/.sw file pairs found in {data_dir}")

    rows, stats = iter_clean_rows(pairs, skip_empty=not args.keep_empty)
    write_csv(rows, output_path)

    print(f"Found {stats['pairs']} paired corpora.")
    print(f"Wrote {stats['written']} rows to {output_path}")
    print(f"Skipped {stats['skipped_empty']} empty rows.")
    if stats["line_mismatches"]:
        print(f"Warning: {stats['line_mismatches']} file pair(s) had mismatched lines.")


if __name__ == "__main__":
    main()
