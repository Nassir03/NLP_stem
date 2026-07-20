from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
from sacrebleu.metrics import BLEU, CHRF, TER
from config import CFG
from kaggle_utils import sync_readonly_artifacts
from preprocessing.tokenizer import load_tokenizers
from inference.decode import decode_sentence, load_model
from evaluation.stem_metrics import evaluate_dataframe

def corpus_metrics(references, hypotheses, include_signatures=False):
    """Compute corpus-level MT metrics from references and existing hypotheses."""
    refs = [list(map(str, references))]
    hyps = list(map(str, hypotheses))
    bleu_metric = BLEU(tokenize="13a")
    chrf_metric = CHRF(word_order=2)
    ter_metric = TER()
    bleu = bleu_metric.corpus_score(hyps, refs)
    chrf = chrf_metric.corpus_score(hyps, refs)
    ter = ter_metric.corpus_score(hyps, refs)
    metrics = {
        "bleu": bleu.score,
        "chrf++": chrf.score,
        "ter": ter.score,
    }
    if include_signatures:
        metrics.update({
            "bleu_signature": str(bleu_metric.get_signature()),
            "chrf++_signature": str(chrf_metric.get_signature()),
            "ter_signature": str(ter_metric.get_signature()),
        })
    return metrics

def evaluate_predictions(prediction_csv, model_name=None):
    """Score an existing CSV; this path never calls the decoder or writes predictions."""
    sync_readonly_artifacts()
    path = Path(prediction_csv)
    if not path.exists():
        raise FileNotFoundError(
            f"Prediction file not found: {path}. "
            "Run evaluation with an existing CSV, or explicitly use --generate."
        )
    df = pd.read_csv(path)
    required = {"target", "prediction"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {', '.join(sorted(missing))}")
    metrics = corpus_metrics(df.target, df.prediction, include_signatures=True)
    CFG.results_dir.mkdir(parents=True, exist_ok=True)
    label = model_name or path.stem
    pd.DataFrame([{"model": label, **metrics}]).to_csv(
        CFG.results_dir / f"{label}_metrics.csv", index=False
    )
    write_stem_scores(df, label)
    print(metrics)
    return metrics

def write_stem_scores(df, model_name):
    """Save STEM preservation metrics for an evaluated prediction table."""
    _, totals, error_rows = evaluate_dataframe(df)
    rows = []
    for name, (correct, total) in totals.items():
        rows.append({
            "model": model_name,
            "metric": f"{name}_accuracy",
            "correct": correct,
            "total": total,
            "score": 100 * correct / total if total else None,
        })
    pd.DataFrame(rows).to_csv(CFG.results_dir / f"{model_name}_stem_scores.csv", index=False)
    error_rows.to_csv(CFG.results_dir / f"{model_name}_error_analysis.csv", index=False)

def generate_and_evaluate(
    model_name,
    limit=None,
    decode_method="beam",
    beam_size=None,
    length_penalty=0.6,
):
    """Explicit generation path for experiments that need fresh model predictions."""
    sync_readonly_artifacts()
    CFG.results_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(CFG.split_dir / "test.csv")
    if limit: df = df.head(limit)
    src_tok, tgt_tok = load_tokenizers()
    model = load_model(model_name, src_tok, tgt_tok)
    preds = [
        decode_sentence(
            model,
            model_name,
            source,
            src_tok,
            tgt_tok,
            method=decode_method,
            beam_size=beam_size,
            length_penalty=length_penalty,
        )
        for source in df.source.astype(str)
    ]
    out = df.copy()
    out["prediction"] = preds
    out.to_csv(CFG.results_dir / f"{model_name}_predictions.csv", index=False)
    metrics = corpus_metrics(out.target, out.prediction, include_signatures=True)
    pd.DataFrame([{"model": model_name, **metrics}]).to_csv(
        CFG.results_dir / f"{model_name}_metrics.csv", index=False
    )
    write_stem_scores(out, model_name)
    print(metrics)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--limit", type=int)
    p.add_argument("--decode-method", choices=["beam", "greedy"], default="beam")
    p.add_argument("--beam-size", type=int, default=CFG.beam_size)
    p.add_argument("--length-penalty", type=float, default=0.6)
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--generate", action="store_true", help="Generate predictions before scoring.")
    mode.add_argument(
        "--no-generate",
        action="store_true",
        help="Score an existing prediction CSV only.",
    )
    p.add_argument(
        "--predictions",
        help="Existing prediction CSV. Defaults to results/<model>_predictions.csv.",
    )
    args = p.parse_args()
    if args.generate:
        generate_and_evaluate(
            args.model,
            args.limit,
            decode_method=args.decode_method,
            beam_size=args.beam_size,
            length_penalty=args.length_penalty,
        )
    else:
        prediction_path = args.predictions or CFG.results_dir / f"{args.model}_predictions.csv"
        evaluate_predictions(prediction_path, args.model)
