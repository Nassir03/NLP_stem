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
    return parser.parse_args()


def source_texts(texts: list[str], model_type: str) -> list[str]:
    if model_type in {"mt5", "byt5"}:
        return [f"translate English to Swahili: {text}" for text in texts]
    return texts


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
    generate_kwargs = {
        "num_beams": args.num_beams,
        "max_new_tokens": args.max_new_tokens,
        "early_stopping": True,
        "no_repeat_ngram_size": 3,
    }
    if args.model_type == "nllb":
        generate_kwargs["forced_bos_token_id"] = tokenizer.convert_tokens_to_ids("swh_Latn")

    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["english", "swahili", "prediction"])
        writer.writeheader()

        for start in range(0, len(rows), args.batch_size):
            batch = rows[start : start + args.batch_size]
            texts = source_texts([row["english"] for row in batch], args.model_type)
            inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=256).to(device)

            with torch.no_grad():
                output_ids = model.generate(**inputs, **generate_kwargs)
            predictions = tokenizer.batch_decode(output_ids, skip_special_tokens=True)

            for row, prediction in zip(batch, predictions):
                writer.writerow(
                    {
                        "english": row["english"],
                        "swahili": row["swahili"],
                        "prediction": prediction,
                    }
                )

    print(f"Wrote {len(rows)} predictions to {args.output.resolve()}")


if __name__ == "__main__":
    main()
