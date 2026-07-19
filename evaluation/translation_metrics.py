from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
import sacrebleu
from config import CFG
from kaggle_utils import sync_readonly_artifacts
from preprocessing.tokenizer import load_tokenizers
from inference.decode import load_model, greedy_decode

def corpus_metrics(references, hypotheses):
    """Compute corpus-level MT metrics from references and existing hypotheses."""
    refs = [list(map(str, references))]
    hyps = list(map(str, hypotheses))
    return {
        "bleu": sacrebleu.corpus_bleu(hyps, refs, tokenize="13a").score,
        "chrf++": sacrebleu.corpus_chrf(hyps, refs, word_order=2).score,
        "ter": sacrebleu.corpus_ter(hyps, refs).score,
    }

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
    metrics = corpus_metrics(df.target, df.prediction)
    CFG.results_dir.mkdir(parents=True, exist_ok=True)
    label = model_name or path.stem
    pd.DataFrame([{"model": label, **metrics}]).to_csv(
        CFG.results_dir / f"{label}_metrics.csv", index=False
    )
    print(metrics)
    return metrics

def generate_and_evaluate(model_name, limit=None):
    """Explicit generation path for experiments that need fresh model predictions."""
    sync_readonly_artifacts()
    CFG.results_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(CFG.split_dir / "test.csv")
    if limit: df = df.head(limit)
    src_tok, tgt_tok = load_tokenizers()
    model = load_model(model_name, src_tok, tgt_tok)
    preds = [greedy_decode(model, model_name, s, src_tok, tgt_tok) for s in df.source.astype(str)]
    out = df.copy()
    out["prediction"] = preds
    out.to_csv(CFG.results_dir / f"{model_name}_predictions.csv", index=False)
    metrics = corpus_metrics(out.target, out.prediction)
    pd.DataFrame([{"model": model_name, **metrics}]).to_csv(
        CFG.results_dir / f"{model_name}_metrics.csv", index=False
    )
    print(metrics)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--limit", type=int)
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
        generate_and_evaluate(args.model, args.limit)
    else:
        prediction_path = args.predictions or CFG.results_dir / f"{args.model}_predictions.csv"
        evaluate_predictions(prediction_path, args.model)
