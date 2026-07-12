from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PREDICTIONS = ROOT / "outputs" / "predictions" / "predictions.csv"
DEFAULT_OUTPUT = ROOT / "outputs" / "metrics" / "stem_metrics.json"

NUMBER_RE = re.compile(r"\b\d+(?:[.,]\d+)?%?\b")
UNIT_RE = re.compile(r"(?<![A-Za-z])(?:mm|cm|km|kg|mg|g|mol|m/s(?:2|²)?|m|s|N|J|W|Pa|V|A|Hz|°C)\b")
FORMULA_RE = re.compile(r"([A-Za-z]\s*=\s*[^,.;]+|[A-Za-z0-9]+\s*[+\-*/^]\s*[A-Za-z0-9]+)")


def extract(pattern: re.Pattern[str], text: str) -> set[str]:
    return {match.group(0).replace(" ", "") for match in pattern.finditer(text)}


def preservation_score(source_items: set[str], prediction_items: set[str]) -> float:
    """Return 1.0 when there is nothing to preserve, otherwise matched ratio."""
    if not source_items:
        return 1.0
    return len(source_items & prediction_items) / len(source_items)


def compute_metrics(rows: list[dict[str, str]]) -> dict[str, float]:
    number_scores = []
    unit_scores = []
    formula_scores = []

    for row in rows:
        source = row["english"]
        prediction = row["prediction"]
        number_scores.append(preservation_score(extract(NUMBER_RE, source), extract(NUMBER_RE, prediction)))
        unit_scores.append(preservation_score(extract(UNIT_RE, source), extract(UNIT_RE, prediction)))
        formula_scores.append(preservation_score(extract(FORMULA_RE, source), extract(FORMULA_RE, prediction)))

    count = max(len(rows), 1)
    return {
        "number_accuracy": round(sum(number_scores) / count, 4),
        "unit_accuracy": round(sum(unit_scores) / count, 4),
        "formula_exact_match": round(sum(formula_scores) / count, 4),
        "rows_evaluated": len(rows),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate STEM preservation metrics.")
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.predictions.exists():
        raise SystemExit(
            f"Prediction file not found: {args.predictions}. "
            "Expected columns: english, swahili, prediction."
        )

    with args.predictions.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    metrics = compute_metrics(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
