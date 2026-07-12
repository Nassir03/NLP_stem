from __future__ import annotations

import argparse
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_DIR = ROOT / "outputs" / "checkpoints" / "nllb"


def require_transformers() -> None:
    try:
        import transformers  # noqa: F401
    except ImportError as exc:
        raise SystemExit("Install transformers and sentencepiece before running inference.") from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Translate English STEM text to Swahili.")
    parser.add_argument("text", help="English text to translate.")
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--num-beams", type=int, default=5)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    return parser.parse_args()


def main() -> None:
    require_transformers()

    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, src_lang="eng_Latn")
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_dir)
    inputs = tokenizer(args.text, return_tensors="pt", truncation=True, max_length=256)
    forced_bos_token_id = tokenizer.convert_tokens_to_ids("swh_Latn")
    output_ids = model.generate(
        **inputs,
        forced_bos_token_id=forced_bos_token_id,
        num_beams=args.num_beams,
        max_new_tokens=args.max_new_tokens,
        early_stopping=True,
        no_repeat_ngram_size=3,
    )
    print(tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0])


if __name__ == "__main__":
    main()
