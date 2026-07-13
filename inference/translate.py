from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_DIR = ROOT / "outputs" / "checkpoints" / "nllb"
sys.path.append(str(ROOT / "models"))

from hf_utils import auth_kwargs
from translation_utils import chunk_text, translate_chunks


def require_transformers() -> None:
    try:
        import transformers  # noqa: F401
    except ImportError as exc:
        raise SystemExit("Install transformers and sentencepiece before running inference.") from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Translate English STEM text to Swahili.")
    parser.add_argument("text", help="English text to translate.")
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument(
        "--model-type",
        choices=["nllb", "marian", "mt5", "byt5"],
        default="nllb",
        help="Checkpoint family. NLLB needs forced Swahili language token; T5 models need a task prefix.",
    )
    parser.add_argument("--num-beams", type=int, default=5)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--max-source-tokens", type=int, default=240)
    parser.add_argument("--batch-size", type=int, default=4)
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

    chunks = chunk_text(
        args.text,
        tokenizer,
        args.model_type,
        max_source_tokens=args.max_source_tokens,
    )
    translations = translate_chunks(
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
    print("\n\n".join(translations))


if __name__ == "__main__":
    main()
