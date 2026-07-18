from __future__ import annotations
import argparse
import subprocess
import sys

def run(*args):
    """Run project stages as modules so imports resolve from the repository root."""
    print("$", sys.executable, *args)
    subprocess.run([sys.executable, *args], check=True)

def main():
    p = argparse.ArgumentParser(
        description="Run the English-Swahili machine translation project stages."
    )
    p.add_argument(
        "stage",
        choices=[
            "check",
            "prepare",
            "tokenize",
            "preprocess_eval",
            "train",
            "evaluate",
            "generate_eval",
        ],
        help=(
            "`evaluate` scores an existing prediction CSV and does not generate "
            "translations. Use `generate_eval` only when you explicitly want model output."
        ),
    )
    p.add_argument("--model", default="gru_attention")
    p.add_argument("--limit", type=int)
    p.add_argument(
        "--predictions",
        help="CSV containing source, target, and prediction columns for non-generating evaluation.",
    )
    args = p.parse_args()

    if args.stage == "check":
        run("-m", "project_check")
    elif args.stage == "prepare":
        run("-m", "preprocessing.prepare_data")
    elif args.stage == "tokenize":
        run("-m", "preprocessing.tokenizer")
    elif args.stage == "preprocess_eval":
        run("-m", "preprocessing.evaluate_preprocess")
    elif args.stage == "train":
        run("-m", "training.train", "--model", args.model)
    elif args.stage == "evaluate":
        cmd = ["-m", "evaluation.translation_metrics", "--model", args.model, "--no-generate"]
        if args.predictions:
            cmd += ["--predictions", args.predictions]
        run(*cmd)
    else:
        cmd = ["-m", "evaluation.translation_metrics", "--model", args.model, "--generate"]
        if args.limit: cmd += ["--limit", str(args.limit)]
        run(*cmd)

if __name__ == "__main__":
    main()
