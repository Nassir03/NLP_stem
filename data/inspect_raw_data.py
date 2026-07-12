from __future__ import annotations

import argparse
import csv
import html
import json
import re
import statistics
from collections import Counter
from pathlib import Path


DATA_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = DATA_DIR / "reports"

# These keywords are only used to label the existing corpus roughly.
# They do not add new data; they help us inspect which rows look STEM related.
SUBJECT_KEYWORDS = {
    "mathematics": {
        "calculate",
        "equation",
        "triangle",
        "circle",
        "area",
        "volume",
        "angle",
        "algebra",
        "ratio",
        "graph",
        "function",
        "probability",
        "geometry",
        "mlinganyo",
        "pembetatu",
        "duara",
        "eneo",
    },
    "biology": {
        "cell",
        "cells",
        "plant",
        "plants",
        "animal",
        "organism",
        "photosynthesis",
        "blood",
        "virus",
        "bacteria",
        "dna",
        "gene",
        "seli",
        "mimea",
        "wanyama",
        "usanisinuru",
        "virusi",
    },
    "physics": {
        "force",
        "energy",
        "velocity",
        "acceleration",
        "mass",
        "motion",
        "gravity",
        "temperature",
        "pressure",
        "electric",
        "current",
        "voltage",
        "nguvu",
        "nishati",
        "kasi",
        "mchapuko",
        "joto",
        "shinikizo",
    },
    "chemistry": {
        "atom",
        "molecule",
        "acid",
        "base",
        "chemical",
        "reaction",
        "oxygen",
        "hydrogen",
        "carbon",
        "solution",
        "compound",
        "atomi",
        "molekuli",
        "kemikali",
        "oksijeni",
        "hidrojeni",
    },
    "technology_computing": {
        "computer",
        "software",
        "hardware",
        "algorithm",
        "data",
        "internet",
        "network",
        "program",
        "technology",
        "digital",
        "kompyuta",
        "programu",
        "mtandao",
        "teknolojia",
    },
}

EN_HINTS = {"the", "and", "is", "are", "of", "to", "in", "for", "with", "that"}
SW_HINTS = {"na", "ya", "wa", "ni", "kwa", "katika", "hii", "hivyo", "kuwa", "kama"}
FORMULA_RE = re.compile(r"([A-Za-z]\s*=\s*[^,.;]+|[A-Za-z0-9]+\s*[+\-*/^]\s*[A-Za-z0-9]+)")
NUMBER_RE = re.compile(r"\b\d+(?:[.,]\d+)?%?\b")
UNIT_RE = re.compile(
    r"(?<![A-Za-z])(?:mm|cm|km|kg|mg|g|mol|m/s(?:2|²)?|m|s|N|J|W|Pa|V|A|Hz|°C|Celsius|newton|joule|watt|volt|ampere)s?\b"
)


def clean_for_counting(text: str) -> str:
    """Use light cleaning only for statistics, not for changing the dataset."""
    return re.sub(r"\s+", " ", text.replace("\ufeff", "")).strip()


def find_parallel_pairs(data_dir: Path) -> list[tuple[str, Path, Path]]:
    """Find all English/Swahili files that share the same corpus name."""
    pairs = []
    for english_file in sorted(data_dir.glob("*.en")):
        source = english_file.name[: -len(".en")]
        swahili_file = data_dir / f"{source}.sw"
        if swahili_file.exists():
            pairs.append((source, english_file, swahili_file))
    return pairs


def iter_raw_rows(data_dir: Path) -> list[dict[str, str]]:
    """Read the raw aligned files before advanced filtering."""
    rows: list[dict[str, str]] = []
    for source, english_file, swahili_file in find_parallel_pairs(data_dir):
        english_lines = english_file.read_text(encoding="utf-8-sig", errors="replace").splitlines()
        swahili_lines = swahili_file.read_text(encoding="utf-8-sig", errors="replace").splitlines()
        for row_number, (english, swahili) in enumerate(zip(english_lines, swahili_lines), start=1):
            rows.append(
                {
                    "source": source,
                    "row_number": str(row_number),
                    "english": clean_for_counting(english),
                    "swahili": clean_for_counting(swahili),
                }
            )
    return rows


def detect_language(text: str) -> str:
    """Small heuristic language check so the project works without extra packages."""
    words = set(re.findall(r"[A-Za-zÀ-ÿ']+", text.lower()))
    if not words:
        return "unknown"
    en_score = len(words & EN_HINTS)
    sw_score = len(words & SW_HINTS)
    if en_score > sw_score:
        return "english_like"
    if sw_score > en_score:
        return "swahili_like"
    return "uncertain"


def detect_subject(english: str, swahili: str) -> str:
    """Assign one broad subject label using the first matching STEM keyword group."""
    text = f"{english} {swahili}".lower()
    scores = {
        subject: sum(1 for keyword in keywords if re.search(rf"\b{re.escape(keyword)}\b", text))
        for subject, keywords in SUBJECT_KEYWORDS.items()
    }
    best_subject, best_score = max(scores.items(), key=lambda item: item[1])
    return best_subject if best_score else "general_or_unknown"


def safe_mean(values: list[int]) -> float:
    return statistics.mean(values) if values else 0.0


def percentile(values: list[int], point: float) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    index = round((len(ordered) - 1) * point)
    return ordered[index]


def row_has_formula_number_or_unit(row: dict[str, str]) -> bool:
    text = f"{row['english']} {row['swahili']}"
    return bool(FORMULA_RE.search(text) or NUMBER_RE.search(text) or UNIT_RE.search(text))


def compute_statistics(rows: list[dict[str, str]]) -> dict[str, object]:
    valid_rows = [row for row in rows if row["english"] and row["swahili"]]
    pair_keys = [(row["english"].lower(), row["swahili"].lower()) for row in valid_rows]
    duplicate_pairs = len(pair_keys) - len(set(pair_keys))

    english_word_lengths = [len(row["english"].split()) for row in valid_rows]
    swahili_word_lengths = [len(row["swahili"].split()) for row in valid_rows]
    english_char_lengths = [len(row["english"]) for row in valid_rows]
    swahili_char_lengths = [len(row["swahili"]) for row in valid_rows]

    subject_distribution = Counter(detect_subject(row["english"], row["swahili"]) for row in valid_rows)
    source_distribution = Counter(row["source"] for row in valid_rows)
    english_language = Counter(detect_language(row["english"]) for row in valid_rows)
    swahili_language = Counter(detect_language(row["swahili"]) for row in valid_rows)

    formula_number_unit_rows = sum(1 for row in valid_rows if row_has_formula_number_or_unit(row))
    stem_rows = sum(count for subject, count in subject_distribution.items() if subject != "general_or_unknown")

    return {
        "total_sentence_pairs": len(rows),
        "valid_non_empty_pairs": len(valid_rows),
        "missing_english": sum(1 for row in rows if not row["english"]),
        "missing_swahili": sum(1 for row in rows if not row["swahili"]),
        "duplicate_pairs": duplicate_pairs,
        "unique_pairs": len(set(pair_keys)),
        "stem_specific_pairs": stem_rows,
        "formula_number_unit_pairs": formula_number_unit_rows,
        "formula_number_unit_percentage": round((formula_number_unit_rows / len(valid_rows)) * 100, 2)
        if valid_rows
        else 0.0,
        "average_english_words": round(safe_mean(english_word_lengths), 2),
        "average_swahili_words": round(safe_mean(swahili_word_lengths), 2),
        "maximum_english_words": max(english_word_lengths, default=0),
        "maximum_swahili_words": max(swahili_word_lengths, default=0),
        "maximum_english_characters": max(english_char_lengths, default=0),
        "maximum_swahili_characters": max(swahili_char_lengths, default=0),
        "english_word_percentiles": {
            "p50": percentile(english_word_lengths, 0.50),
            "p90": percentile(english_word_lengths, 0.90),
            "p95": percentile(english_word_lengths, 0.95),
            "p99": percentile(english_word_lengths, 0.99),
        },
        "swahili_word_percentiles": {
            "p50": percentile(swahili_word_lengths, 0.50),
            "p90": percentile(swahili_word_lengths, 0.90),
            "p95": percentile(swahili_word_lengths, 0.95),
            "p99": percentile(swahili_word_lengths, 0.99),
        },
        "language_distribution": {
            "english_column": dict(english_language),
            "swahili_column": dict(swahili_language),
        },
        "subject_distribution": dict(subject_distribution),
        "source_distribution": dict(source_distribution),
    }


def write_distribution_csv(counter: dict[str, int], output_path: Path) -> None:
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["label", "count"])
        for label, count in sorted(counter.items(), key=lambda item: item[1], reverse=True):
            writer.writerow([label, count])


def write_bar_svg(counter: dict[str, int], title: str, output_path: Path) -> None:
    """Create a simple SVG bar chart without requiring matplotlib."""
    items = sorted(counter.items(), key=lambda item: item[1], reverse=True)
    width = 900
    row_height = 34
    left = 220
    bar_width = 560
    height = 80 + row_height * max(len(items), 1)
    max_value = max(counter.values(), default=1)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="24" y="34" font-family="Arial" font-size="22" font-weight="700">{html.escape(title)}</text>',
    ]
    for index, (label, count) in enumerate(items):
        y = 66 + index * row_height
        current_width = int((count / max_value) * bar_width)
        parts.append(f'<text x="24" y="{y + 18}" font-family="Arial" font-size="14">{html.escape(label)}</text>')
        parts.append(f'<rect x="{left}" y="{y}" width="{current_width}" height="22" fill="#2f80ed"/>')
        parts.append(f'<text x="{left + current_width + 8}" y="{y + 17}" font-family="Arial" font-size="13">{count}</text>')
    parts.append("</svg>")
    output_path.write_text("\n".join(parts), encoding="utf-8")


def write_report(stats: dict[str, object], output_path: Path) -> None:
    subject_counts = stats["subject_distribution"]
    lines = [
        "Raw Data Inspection Report",
        "==========================",
        "",
        f"Total pairs: {stats['total_sentence_pairs']}",
        f"Valid non-empty pairs: {stats['valid_non_empty_pairs']}",
        f"Unique pairs: {stats['unique_pairs']}",
        f"Duplicate pairs: {stats['duplicate_pairs']}",
        f"STEM-specific pairs: {stats['stem_specific_pairs']}",
        f"Missing English values: {stats['missing_english']}",
        f"Missing Swahili values: {stats['missing_swahili']}",
        f"Average English length: {stats['average_english_words']} words",
        f"Average Swahili length: {stats['average_swahili_words']} words",
        f"Maximum English length: {stats['maximum_english_words']} words",
        f"Maximum Swahili length: {stats['maximum_swahili_words']} words",
        f"Rows containing formulas, numbers, or units: {stats['formula_number_unit_pairs']} "
        f"({stats['formula_number_unit_percentage']}%)",
        "",
        "Subject distribution:",
    ]
    for subject, count in sorted(subject_counts.items(), key=lambda item: item[1], reverse=True):
        lines.append(f"- {subject}: {count}")

    lines.extend(["", "Language distribution:", json.dumps(stats["language_distribution"], indent=2)])
    lines.extend(["", "Source distribution:"])
    for source, count in sorted(stats["source_distribution"].items(), key=lambda item: item[1], reverse=True):
        lines.append(f"- {source}: {count}")

    output_path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect raw English-Swahili corpus files.")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = iter_raw_rows(args.data_dir.resolve())
    stats = compute_statistics(rows)

    (output_dir / "raw_data_statistics.json").write_text(
        json.dumps(stats, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    write_report(stats, output_dir / "raw_data_report.txt")
    write_distribution_csv(stats["source_distribution"], output_dir / "source_distribution.csv")
    write_distribution_csv(stats["subject_distribution"], output_dir / "subject_distribution.csv")
    write_bar_svg(stats["source_distribution"], "Source Distribution", output_dir / "source_distribution.svg")
    write_bar_svg(stats["subject_distribution"], "Subject Distribution", output_dir / "subject_distribution.svg")

    print(f"Total pairs: {stats['total_sentence_pairs']}")
    print(f"Valid non-empty pairs: {stats['valid_non_empty_pairs']}")
    print(f"Unique pairs: {stats['unique_pairs']}")
    print(f"STEM-specific pairs: {stats['stem_specific_pairs']}")
    print(f"Reports written to {output_dir}")


if __name__ == "__main__":
    main()
