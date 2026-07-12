from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_DIR = ROOT / "outputs" / "checkpoints" / "nllb"
sys.path.append(str(ROOT / "models"))

from hf_utils import auth_kwargs


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

    source_text = args.text
    if args.model_type in {"mt5", "byt5"}:
        source_text = f"translate English to Swahili: {source_text}"

    inputs = tokenizer(source_text, return_tensors="pt", truncation=True, max_length=256).to(device)
    generate_kwargs = {
        "num_beams": args.num_beams,
        "max_new_tokens": args.max_new_tokens,
        "early_stopping": True,
        "no_repeat_ngram_size": 3,
    }
    if args.model_type == "nllb":
        generate_kwargs["forced_bos_token_id"] = tokenizer.convert_tokens_to_ids("swh_Latn")

    with torch.no_grad():
        output_ids = model.generate(**inputs, **generate_kwargs)
    print(tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0])


if __name__ == "__main__":
    main()
