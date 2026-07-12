from __future__ import annotations

import argparse
import csv
import random
from collections import defaultdict
from pathlib import Path


DATA_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = DATA_DIR / "parallel_corpus_clean.csv"
DEFAULT_OUTPUT_DIR = DATA_DIR / "splits"
FIELDNAMES = ["source", "row_number", "english", "swahili"]


def positive_float(value: str) -> float:
    number = float(value)
    if number < 0:
        raise argparse.ArgumentTypeError("Split ratios must be zero or greater.")
    return number


def read_rows(input_path: Path) -> list[dict[str, str]]:
    with input_path.open("r", encoding="utf-8", newline="") as csv_handle:
        reader = csv.DictReader(csv_handle)
        missing_columns = [column for column in FIELDNAMES if column not in reader.fieldnames]

        if missing_columns:
            raise SystemExit(
                f"{input_path} is missing required column(s): {', '.join(missing_columns)}"
            )

        rows = []
        for row in reader:
            english = row["english"].strip()
            swahili = row["swahili"].strip()

            if not english or not swahili:
                continue

            rows.append(
                {
                    "source": row["source"].strip(),
                    "row_number": row["row_number"].strip(),
                    "english": english,
                    "swahili": swahili,
                }
            )

    if not rows:
        raise SystemExit(f"No usable rows found in {input_path}")

    return rows


def split_source_rows(
    rows: list[dict[str, str]],
    *,
    train_ratio: float,
    validation_ratio: float,
) -> tuple[list[dict[str, str]], list[dict[str, str]], list[dict[str, str]]]:
    train_end = round(len(rows) * train_ratio)
    validation_end = train_end + round(len(rows) * validation_ratio)

    train_rows = rows[:train_end]
    validation_rows = rows[train_end:validation_end]
    test_rows = rows[validation_end:]

    return train_rows, validation_rows, test_rows


def split_rows(
    rows: list[dict[str, str]],
    *,
    train_ratio: float,
    validation_ratio: float,
    seed: int,
) -> tuple[list[dict[str, str]], list[dict[str, str]], list[dict[str, str]]]:
    grouped_rows: dict[str, list[dict[str, str]]] = defaultdict(list)
    rng = random.Random(seed)

    for row in rows:
        grouped_rows[row["source"]].append(row)

    train_rows: list[dict[str, str]] = []
    validation_rows: list[dict[str, str]] = []
    test_rows: list[dict[str, str]] = []

    for source_rows in grouped_rows.values():
        rng.shuffle(source_rows)
        source_train, source_validation, source_test = split_source_rows(
            source_rows,
            train_ratio=train_ratio,
            validation_ratio=validation_ratio,
        )
        train_rows.extend(source_train)
        validation_rows.extend(source_validation)
        test_rows.extend(source_test)

    rng.shuffle(train_rows)
    rng.shuffle(validation_rows)
    rng.shuffle(test_rows)

    return train_rows, validation_rows, test_rows


def write_csv(rows: list[dict[str, str]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8", newline="") as csv_handle:
        writer = csv.DictWriter(csv_handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def validate_ratios(train: float, validation: float, test: float) -> None:
    total = train + validation + test
    if total <= 0:
        raise SystemExit("At least one split ratio must be greater than zero.")

    if abs(total - 1.0) > 0.000001:
        raise SystemExit(
            f"Split ratios must add up to 1.0. Current total is {total:.4f}."
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create reproducible train, validation, and test CSV splits."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="Clean CSV file created by convert_to_csv.py.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where train.csv, validation.csv, and test.csv are written.",
    )
    parser.add_argument("--train", type=positive_float, default=0.8)
    parser.add_argument("--validation", type=positive_float, default=0.1)
    parser.add_argument("--test", type=positive_float, default=0.1)
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible splits.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = args.input.resolve()
    output_dir = args.output_dir.resolve()

    validate_ratios(args.train, args.validation, args.test)

    rows = read_rows(input_path)
    train_rows, validation_rows, test_rows = split_rows(
        rows,
        train_ratio=args.train,
        validation_ratio=args.validation,
        seed=args.seed,
    )

    write_csv(train_rows, output_dir / "train.csv")
    write_csv(validation_rows, output_dir / "validation.csv")
    write_csv(test_rows, output_dir / "test.csv")

    print(f"Read {len(rows)} clean rows from {input_path}")
    print(f"Wrote {len(train_rows)} rows to {output_dir / 'train.csv'}")
    print(f"Wrote {len(validation_rows)} rows to {output_dir / 'validation.csv'}")
    print(f"Wrote {len(test_rows)} rows to {output_dir / 'test.csv'}")


if __name__ == "__main__":
    main()
