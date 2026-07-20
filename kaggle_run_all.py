from __future__ import annotations

import argparse
import subprocess
import sys

import torch

from main import NEURAL_MODELS

RUNNER_VERSION = "2026-07-20-gpu-short-best-no-generation"


def run(*args: str) -> None:
    """Run one project command and stop immediately if it fails."""
    print("$", sys.executable, *args, flush=True)
    subprocess.run([sys.executable, *args], check=True)


def run_pipeline(args: argparse.Namespace) -> None:
    """Execute the complete Kaggle workflow in the correct project order."""
    print(f"[kaggle_run_all] version={RUNNER_VERSION}", flush=True)
    if args.require_gpu and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA GPU is not available. In Kaggle, enable it from "
            "Notebook settings -> Accelerator -> GPU, then restart and rerun."
        )
    device = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print(f"[kaggle_run_all] torch={torch.__version__} device={device}", flush=True)
    if not args.skip_prepare:
        run("main.py", "prepare")
    if not args.skip_tokenize:
        run("main.py", "tokenize")

    run("main.py", "preprocess_eval")
    run("main.py", "check")

    if args.run_smt:
        smt_train = ["main.py", "smt_train", "--iterations", str(args.smt_iterations)]
        if args.smt_limit:
            smt_train += ["--limit", str(args.smt_limit)]
        run(*smt_train)

        if args.generate_predictions:
            smt_generate = ["main.py", "smt_generate"]
            if args.eval_limit:
                smt_generate += ["--limit", str(args.eval_limit)]
            run(*smt_generate)
            run("main.py", "smt_eval")

    models = args.models or NEURAL_MODELS
    for model in models:
        train_cmd = ["main.py", "train", "--model", model]
        if args.epochs:
            train_cmd += ["--epochs", str(args.epochs)]
        if args.batch_size:
            train_cmd += ["--batch-size", str(args.batch_size)]
        if args.train_limit:
            train_cmd += ["--train-limit", str(args.train_limit)]
        if args.valid_limit:
            train_cmd += ["--valid-limit", str(args.valid_limit)]
        if args.test_limit:
            train_cmd += ["--test-limit", str(args.test_limit)]
        if args.skip_existing:
            train_cmd += ["--skip-existing"]
        run(*train_cmd)

        if args.generate_predictions:
            eval_cmd = ["main.py", "generate_eval", "--model", model]
            if args.eval_limit:
                eval_cmd += ["--limit", str(args.eval_limit)]
            run(*eval_cmd)

    if args.generate_predictions:
        run("-m", "evaluation.summarize_results")
    else:
        run("-m", "evaluation.summarize_results", "--training")
        run("visualization/validation_cross_entropy.py")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Kaggle-friendly full runner for SMT and all neural MT models."
    )
    parser.add_argument("--skip-prepare", action="store_true", help="Reuse existing split CSV files.")
    parser.add_argument("--skip-tokenize", action="store_true", help="Reuse existing tokenizer models.")
    parser.add_argument("--no-smt", dest="run_smt", action="store_false", help="Skip SMT baseline.")
    parser.add_argument(
        "--generate-predictions",
        action="store_true",
        help="Generate translation prediction CSVs and BLEU/chrF++/TER metrics.",
    )
    parser.add_argument("--smt-iterations", type=int, default=5)
    parser.add_argument("--smt-limit", type=int, default=50000)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--train-limit", type=int, default=None)
    parser.add_argument("--valid-limit", type=int, default=None)
    parser.add_argument("--test-limit", type=int, default=None)
    parser.add_argument("--eval-limit", type=int, default=None)
    parser.add_argument("--skip-existing", action="store_true", help="Reuse existing neural checkpoints.")
    parser.add_argument(
        "--allow-cpu",
        dest="require_gpu",
        action="store_false",
        help="Allow CPU training. By default this runner requires GPU.",
    )
    parser.add_argument("--models", nargs="+", choices=NEURAL_MODELS)
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Fast Kaggle smoke test: tiny data limits and one epoch.",
    )
    parser.add_argument(
        "--best-quality",
        action="store_true",
        help="Longer no-generation training preset for stronger validation results.",
    )
    parser.add_argument(
        "--full-data",
        action="store_true",
        help="Use full train/validation data with config/default epochs.",
    )
    args = parser.parse_args()
    args.require_gpu = True if args.require_gpu is None else args.require_gpu

    if args.quick:
        args.epochs = args.epochs or 1
        args.batch_size = args.batch_size or 16
        args.train_limit = args.train_limit or 128
        args.valid_limit = args.valid_limit or 64
        args.test_limit = args.test_limit or 64
        args.eval_limit = args.eval_limit or 32
        args.smt_limit = min(args.smt_limit, 1000)
    elif args.best_quality:
        args.epochs = args.epochs or 25
        args.batch_size = args.batch_size or 64
    elif not args.full_data:
        # Default Kaggle run: short enough to finish, large enough to compare models.
        args.epochs = args.epochs or 6
        args.batch_size = args.batch_size or 64
        args.train_limit = args.train_limit or 20000
        args.valid_limit = args.valid_limit or 3000
        args.smt_limit = min(args.smt_limit, 20000)

    return args


if __name__ == "__main__":
    run_pipeline(parse_args())
