from __future__ import annotations
import argparse
import re
from collections import Counter
from pathlib import Path
import pandas as pd
from config import CFG

NUMBER_RE = re.compile(r"(?<!\w)[+-]?\d+(?:[.,]\d+)?(?:%|°[CF])?")
SYMBOL_RE = re.compile(r"(?:[=+×÷<>≤≥±√∑∫]|[A-Za-z]+\d+|[A-Za-z]\^\d+)")
UNIT_RE = re.compile(r"\b(?:kg|g|mg|km|m|cm|mm|s|ms|Hz|kHz|V|A|W|J|N|Pa|mol|m/s|m/s²)\b")

def multiset_recall(source, prediction, regex):
    """Measure whether protected STEM tokens survive in a generated translation."""
    src, pred = Counter(regex.findall(str(source))), Counter(regex.findall(str(prediction)))
    total = sum(src.values())
    correct = sum(min(n, pred[k]) for k, n in src.items())
    return correct, total

def terminology_accuracy(source, prediction, glossary):
    """Check glossary terms only when their English source term is present."""
    correct = total = 0
    s, p = str(source).lower(), str(prediction).lower()
    for en, sw in glossary:
        if re.search(rf"\b{re.escape(en.lower())}\b", s):
            total += 1
            if re.search(rf"\b{re.escape(sw.lower())}\b", p):
                correct += 1
    return correct, total

def load_glossary():
    """Load the optional manually reviewed bilingual STEM glossary."""
    glossary_path = CFG.root / "data" / "stem_glossary.csv"
    if not glossary_path.exists():
        return []
    glossary = pd.read_csv(glossary_path)
    return list(zip(glossary.english.astype(str), glossary.swahili.astype(str)))

def evaluate_dataframe(df, glossary=None):
    """Run STEM-focused checks against an in-memory prediction table."""
    required = {"source", "prediction"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Prediction data is missing columns: {', '.join(sorted(missing))}")
    glossary = glossary if glossary is not None else load_glossary()
    totals = {"symbol": [0,0], "number": [0,0], "unit": [0,0], "term": [0,0]}
    error_rows = []
    for _, r in df.iterrows():
        row_errors = []
        for name, regex in [("symbol", SYMBOL_RE), ("number", NUMBER_RE), ("unit", UNIT_RE)]:
            c, n = multiset_recall(r.source, r.prediction, regex)
            totals[name][0] += c; totals[name][1] += n
            if c < n: row_errors.append(name)
        c, n = terminology_accuracy(r.source, r.prediction, glossary)
        totals["term"][0] += c; totals["term"][1] += n
        if c < n: row_errors.append("terminology")
        if row_errors:
            error_rows.append({**r.to_dict(), "error_types": ",".join(row_errors)})

    scores = {
        f"{name}_accuracy": 100 * c / max(n, 1)
        for name, (c, n) in totals.items()
    }
    return scores, totals, pd.DataFrame(error_rows)

def evaluate_file(prediction_csv, model_name=None):
    """Run STEM-focused checks against an existing prediction file."""
    df = pd.read_csv(prediction_csv)
    scores, totals, error_rows = evaluate_dataframe(df)
    label = model_name or Path(prediction_csv).stem.removesuffix("_predictions")
    CFG.results_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(error_rows).to_csv(CFG.results_dir / "error_analysis.csv", index=False)
    rows = []
    for name, (correct, total) in totals.items():
        score = 100 * correct / total if total else None
        rows.append({
            "model": label,
            "metric": f"{name}_accuracy",
            "correct": correct,
            "total": total,
            "score": score,
        })
    pd.DataFrame(rows).to_csv(CFG.results_dir / f"{label}_stem_scores.csv", index=False)
    print(scores)
    return scores

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("prediction_csv")
    p.add_argument("--model")
    args = p.parse_args()
    evaluate_file(args.prediction_csv, args.model)
