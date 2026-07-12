from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path


DATA_DIR = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = DATA_DIR / "parallel_corpus_clean.csv"
DEFAULT_OUTPUT = DATA_DIR / "processed" / "protected_pairs.csv"

FORMULA_RE = re.compile(r"([A-Za-z]\s*=\s*[^,.;]+|[A-Za-z0-9]+\s*[+\-*/^]\s*[A-Za-z0-9]+)")
CHEMICAL_RE = re.compile(r"\b(?:H2O|CO2|NaCl|H2SO4|O2|N2|CO|CH4|NH3)\b")
NUMBER_UNIT_RE = re.compile(
    r"\b\d+(?:[.,]\d+)?\s*(?:%|mm|cm|km|kg|mg|g|mol|m/s(?:2|²)?|m|s|N|J|W|Pa|V|A|Hz|°C|Celsius)?\b"
)


def protect_text(text: str) -> tuple[str, dict[str, str]]:
    """Replace STEM expressions with placeholders for safer model training."""
    placeholders: dict[str, str] = {}

    def replace(match: re.Match[str]) -> str:
        placeholder = f"<STEM_{len(placeholders)}>"
        placeholders[placeholder] = match.group(0)
        return placeholder

    protected = CHEMICAL_RE.sub(replace, text)
    protected = FORMULA_RE.sub(replace, protected)
    protected = NUMBER_UNIT_RE.sub(replace, protected)
    return protected, placeholders


def restore_text(text: str, placeholders: dict[str, str]) -> str:
    """Put protected STEM expressions back after translation."""
    for placeholder, original in placeholders.items():
        text = text.replace(placeholder, original)
    return text


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a protected STEM-expression dataset variant.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    written = 0

    with args.input.open("r", encoding="utf-8", newline="") as input_handle, args.output.open(
        "w", encoding="utf-8", newline=""
    ) as output_handle:
        reader = csv.DictReader(input_handle)
        fieldnames = ["source", "row_number", "english", "swahili", "english_placeholders", "swahili_placeholders"]
        writer = csv.DictWriter(output_handle, fieldnames=fieldnames)
        writer.writeheader()

        for row in reader:
            english, english_placeholders = protect_text(row["english"])
            swahili, swahili_placeholders = protect_text(row["swahili"])
            writer.writerow(
                {
                    "source": row["source"],
                    "row_number": row["row_number"],
                    "english": english,
                    "swahili": swahili,
                    "english_placeholders": json.dumps(english_placeholders, ensure_ascii=False),
                    "swahili_placeholders": json.dumps(swahili_placeholders, ensure_ascii=False),
                }
            )
            written += 1

    print(f"Wrote {written} protected rows to {args.output.resolve()}")


if __name__ == "__main__":
    main()
