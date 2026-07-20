from __future__ import annotations
import argparse
import subprocess
import sys

import torch

from config import CFG, cuda_status

PROJECT_VERSION = "2026-07-21-50ep-beam-stem-eval"

NEURAL_MODELS = [
    "rnn_seq2seq",
    "gru_seq2seq",
    "lstm_seq2seq",
    "gru_attention",
    "lstm_attention",
    "transformer",
]

def run(*args):
    """Run project stages as modules so imports resolve from the repository root."""
    print("$", sys.executable, *args)
    subprocess.run([sys.executable, *args], check=True)

def main():
    print(f"[NLP_stem] version={PROJECT_VERSION}")
    p = argparse.ArgumentParser(
        description="Run the English-Swahili machine translation project stages."
    )
    p.add_argument(
        "stage",
        choices=[
            "check",
            "device",
            "prepare",
            "tokenize",
            "preprocess_eval",
            "train",
            "train_all",
            "evaluate",
            "generate_eval",
            "generate_all",
            "smt_train",
            "smt_generate",
            "smt_eval",
            "summarize",
            "summarize_training",
        ],
        help=(
            "`evaluate` scores an existing prediction CSV and does not generate "
            "translations. Use `generate_eval` only when you explicitly want model output."
        ),
    )
    p.add_argument("--model", default="gru_attention")
    p.add_argument("--limit", type=int)
    p.add_argument("--epochs", type=int, help="Override epochs for neural training.")
    p.add_argument("--patience", type=int, help="Override early-stopping patience.")
    p.add_argument("--batch-size", type=int, help="Override batch size for neural training.")
    p.add_argument("--beam-size", type=int, default=CFG.beam_size, help="Beam width for generated evaluation.")
    p.add_argument("--length-penalty", type=float, default=0.6, help="Beam-search length penalty.")
    p.add_argument(
        "--decode-method",
        choices=["beam", "greedy"],
        default="beam",
        help="Decoder used only by generate_eval/generate_all.",
    )
    p.add_argument("--train-limit", type=int, help="Use only the first N train rows.")
    p.add_argument("--valid-limit", type=int, help="Use only the first N validation rows.")
    p.add_argument("--test-limit", type=int, help="Use only the first N test rows.")
    p.add_argument("--skip-existing", action="store_true", help="Skip neural models with checkpoints.")
    p.add_argument("--iterations", type=int, default=5, help="SMT EM iterations.")
    p.add_argument(
        "--predictions",
        help="CSV containing source, target, and prediction columns for non-generating evaluation.",
    )
    args = p.parse_args()

    if args.stage == "device":
        cuda_ok, cuda_message = cuda_status()
        if cuda_ok:
            print(f"device=cuda name={torch.cuda.get_device_name(0)}")
        else:
            print(f"device=cpu cuda_usable=False reason={cuda_message}")
    elif args.stage == "check":
        run("-m", "project_check")
    elif args.stage == "prepare":
        run("-m", "preprocessing.prepare_data")
    elif args.stage == "tokenize":
        run("-m", "preprocessing.tokenizer")
    elif args.stage == "preprocess_eval":
        run("-m", "preprocessing.evaluate_preprocess")
    elif args.stage == "train":
        cmd = ["-m", "training.train", "--model", args.model]
        if args.epochs:
            cmd += ["--epochs", str(args.epochs)]
        if args.patience:
            cmd += ["--patience", str(args.patience)]
        if args.batch_size:
            cmd += ["--batch-size", str(args.batch_size)]
        if args.train_limit:
            cmd += ["--train-limit", str(args.train_limit)]
        if args.valid_limit:
            cmd += ["--valid-limit", str(args.valid_limit)]
        if args.test_limit:
            cmd += ["--test-limit", str(args.test_limit)]
        if args.skip_existing:
            cmd += ["--skip-existing"]
        run(*cmd)
    elif args.stage == "train_all":
        for model in NEURAL_MODELS:
            cmd = ["-m", "training.train", "--model", model]
            if args.epochs:
                cmd += ["--epochs", str(args.epochs)]
            if args.patience:
                cmd += ["--patience", str(args.patience)]
            if args.batch_size:
                cmd += ["--batch-size", str(args.batch_size)]
            if args.train_limit:
                cmd += ["--train-limit", str(args.train_limit)]
            if args.valid_limit:
                cmd += ["--valid-limit", str(args.valid_limit)]
            if args.test_limit:
                cmd += ["--test-limit", str(args.test_limit)]
            if args.skip_existing:
                cmd += ["--skip-existing"]
            run(*cmd)
    elif args.stage == "evaluate":
        cmd = ["-m", "evaluation.translation_metrics", "--model", args.model, "--no-generate"]
        if args.predictions:
            cmd += ["--predictions", args.predictions]
        run(*cmd)
    elif args.stage == "generate_eval":
        cmd = [
            "-m",
            "evaluation.translation_metrics",
            "--model",
            args.model,
            "--generate",
            "--decode-method",
            args.decode_method,
            "--beam-size",
            str(args.beam_size),
            "--length-penalty",
            str(args.length_penalty),
        ]
        if args.limit: cmd += ["--limit", str(args.limit)]
        run(*cmd)
    elif args.stage == "generate_all":
        for model in NEURAL_MODELS:
            cmd = [
                "-m",
                "evaluation.translation_metrics",
                "--model",
                model,
                "--generate",
                "--decode-method",
                args.decode_method,
                "--beam-size",
                str(args.beam_size),
                "--length-penalty",
                str(args.length_penalty),
            ]
            if args.limit:
                cmd += ["--limit", str(args.limit)]
            run(*cmd)
    elif args.stage == "smt_train":
        cmd = ["-m", "smt.em_alignment", "--iterations", str(args.iterations)]
        if args.limit:
            cmd += ["--limit", str(args.limit)]
        run(*cmd)
    elif args.stage == "smt_generate":
        cmd = ["-m", "smt.decoder"]
        if args.limit:
            cmd += ["--limit", str(args.limit)]
        run(*cmd)
    elif args.stage == "smt_eval":
        run(
            "-m",
            "evaluation.translation_metrics",
            "--model",
            "smt",
            "--no-generate",
            "--predictions",
            str(CFG.results_dir / "smt_predictions.csv"),
        )
    elif args.stage == "summarize":
        run("-m", "evaluation.summarize_results")
    else:
        run("-m", "evaluation.summarize_results", "--training")

if __name__ == "__main__":
    main()
