from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_DIR = ROOT / "outputs" / "checkpoints" / "nllb"
DEFAULT_TEST_FILE = ROOT / "data" / "splits" / "test.csv"
DEFAULT_OUTPUT = ROOT / "outputs" / "predictions" / "predictions.csv"
sys.path.append(str(ROOT / "models"))

from hf_utils import auth_kwargs
from translation_utils import chunk_text, translate_chunks


def require_transformers() -> None:
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401
    except ImportError as exc:
        raise SystemExit("Install torch, transformers and sentencepiece before generating predictions.") from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate test-set translations for evaluation.")
    parser.add_argument("--test-file", type=Path, default=DEFAULT_TEST_FILE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--model-type", choices=["nllb", "marian", "mt5", "byt5"], default="nllb")
    parser.add_argument("--max-rows", type=int, default=None, help="Limit rows for a quick smoke test.")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-beams", type=int, default=5)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--max-source-tokens", type=int, default=240)
    parser.add_argument("--length-penalty", type=float, default=1.0)
    parser.add_argument("--no-repeat-ngram-size", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    require_transformers()

    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    args = parse_args()
    tokenizer_kwargs = {"src_lang": "eng_Latn"} if args.model_type == "nllb" else {}
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, **tokenizer_kwargs, **auth_kwargs())
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_dir, **auth_kwargs())

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    with args.test_file.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if args.max_rows:
        rows = rows[: args.max_rows]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["english", "swahili", "prediction"])
        writer.writeheader()

        for row in rows:
            chunks = chunk_text(
                row["english"],
                tokenizer,
                args.model_type,
                max_source_tokens=args.max_source_tokens,
            )
            predictions = translate_chunks(
                chunks=chunks,
                tokenizer=tokenizer,
                model=model,
                model_type=args.model_type,
                device=device,
                batch_size=args.batch_size,
                num_beams=args.num_beams,
                max_new_tokens=args.max_new_tokens,
                length_penalty=args.length_penalty,
                no_repeat_ngram_size=args.no_repeat_ngram_size,
            )

            writer.writerow(
                {
                    "english": row["english"],
                    "swahili": row["swahili"],
                    "prediction": " ".join(predictions),
                }
            )

    print(f"Wrote {len(rows)} predictions to {args.output.resolve()}")


if __name__ == "__main__":
    main()
