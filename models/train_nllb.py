from __future__ import annotations

import argparse
import csv
import inspect
import os
from pathlib import Path

from hf_utils import auth_kwargs, restrict_to_single_gpu


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "splits"
OUTPUT_DIR = ROOT / "outputs" / "checkpoints" / "nllb"
REQUIRED_COLUMNS = {"english", "swahili"}

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def require_libraries() -> None:
    try:
        import datasets  # noqa: F401
        import sacrebleu  # noqa: F401
        import sentencepiece  # noqa: F401
        import transformers  # noqa: F401
    except ImportError as exc:
        raise SystemExit(
            "Install transformers, datasets, sentencepiece and sacrebleu before running NLLB training."
        ) from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune facebook/nllb-200-distilled-600M.")
    parser.add_argument("--train-file", type=Path, default=DATA_DIR / "train.csv")
    parser.add_argument("--validation-file", type=Path, default=DATA_DIR / "validation.csv")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--model-name", default="facebook/nllb-200-distilled-600M")
    parser.add_argument("--max-source-length", type=int, default=256)
    parser.add_argument("--max-target-length", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--epochs", type=float, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--num-beams", type=int, default=5)
    parser.add_argument("--min-chars", type=int, default=2)
    parser.add_argument("--max-length-ratio", type=float, default=4.0)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-validation-samples", type=int, default=None)
    parser.add_argument(
        "--allow-multi-gpu",
        action="store_true",
        help=(
            "Allow Transformers Trainer to use every visible GPU. Disabled by default "
            "because Kaggle dual-GPU DataParallel can crash NLLB training."
        ),
    )
    return parser.parse_args()


def validate_csv(path: Path) -> None:
    if not path.exists():
        raise SystemExit(f"Required split file not found: {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = set(reader.fieldnames or [])
    missing = sorted(REQUIRED_COLUMNS - fieldnames)
    if missing:
        raise SystemExit(f"{path} is missing required column(s): {', '.join(missing)}")


def strategy_kwargs(training_args_class) -> dict[str, str]:
    """Support both old and new Transformers argument names."""
    parameters = inspect.signature(training_args_class.__init__).parameters
    if "eval_strategy" in parameters:
        return {"eval_strategy": "epoch", "save_strategy": "epoch"}
    return {"evaluation_strategy": "epoch", "save_strategy": "epoch"}


def trainer_tokenizer_kwargs(trainer_class, tokenizer) -> dict[str, object]:
    """Support Transformers versions before and after tokenizer was renamed."""
    parameters = inspect.signature(trainer_class.__init__).parameters
    if "processing_class" in parameters:
        return {"processing_class": tokenizer}
    return {"tokenizer": tokenizer}


def main() -> None:
    args = parse_args()
    if not args.allow_multi_gpu:
        restrict_to_single_gpu()

    require_libraries()

    from datasets import load_dataset
    import numpy as np
    import sacrebleu
    import torch
    from transformers import (
        AutoModelForSeq2SeqLM,
        AutoTokenizer,
        DataCollatorForSeq2Seq,
        Seq2SeqTrainer,
        Seq2SeqTrainingArguments,
    )

    validate_csv(args.train_file)
    validate_csv(args.validation_file)

    dataset = load_dataset(
        "csv",
        data_files={
            "train": str(args.train_file),
            "validation": str(args.validation_file),
        },
    )

    def keep_clean_pair(row):
        english = str(row["english"]).strip()
        swahili = str(row["swahili"]).strip()
        if len(english) < args.min_chars or len(swahili) < args.min_chars:
            return False
        if english.casefold() == swahili.casefold():
            return False
        shorter = max(min(len(english), len(swahili)), 1)
        longer = max(len(english), len(swahili))
        return longer / shorter <= args.max_length_ratio

    dataset = dataset.filter(keep_clean_pair)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        src_lang="eng_Latn",
        tgt_lang="swh_Latn",
        use_fast=False,
        **auth_kwargs(),
    )
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name, **auth_kwargs())
    forced_bos_token_id = tokenizer.convert_tokens_to_ids("swh_Latn")
    model.config.forced_bos_token_id = forced_bos_token_id
    model.generation_config.forced_bos_token_id = forced_bos_token_id

    def tokenize_batch(batch):
        # NLLB uses language IDs, so Swahili is forced during generation/evaluation.
        inputs = tokenizer(
            batch["english"],
            max_length=args.max_source_length,
            truncation=True,
        )
        labels = tokenizer(
            text_target=batch["swahili"],
            max_length=args.max_target_length,
            truncation=True,
        )
        inputs["labels"] = labels["input_ids"]
        return inputs

    tokenized = dataset.map(tokenize_batch, batched=True, remove_columns=dataset["train"].column_names)
    if args.max_train_samples:
        tokenized["train"] = tokenized["train"].select(range(min(args.max_train_samples, len(tokenized["train"]))))
    if args.max_validation_samples:
        tokenized["validation"] = tokenized["validation"].select(
            range(min(args.max_validation_samples, len(tokenized["validation"])))
        )

    use_fp16 = torch.cuda.is_available()
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False

    def compute_metrics(eval_preds):
        predictions, labels = eval_preds
        if isinstance(predictions, tuple):
            predictions = predictions[0]
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_predictions = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        decoded_predictions = [text.strip() for text in decoded_predictions]
        decoded_labels = [text.strip() for text in decoded_labels]
        bleu = sacrebleu.corpus_bleu(decoded_predictions, [decoded_labels]).score
        return {"bleu": round(bleu, 4)}

    training_args = Seq2SeqTrainingArguments(
        output_dir=str(args.output_dir),
        learning_rate=args.learning_rate,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        weight_decay=0.01,
        warmup_steps=args.warmup_steps,
        label_smoothing_factor=0.1,
        predict_with_generate=True,
        generation_num_beams=args.num_beams,
        generation_max_length=args.max_target_length,
        save_total_limit=2,
        logging_steps=100,
        fp16=use_fp16,
        gradient_checkpointing=True,
        load_best_model_at_end=True,
        metric_for_best_model="bleu",
        greater_is_better=True,
        seed=42,
        data_seed=42,
        **strategy_kwargs(Seq2SeqTrainingArguments),
    )
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        data_collator=DataCollatorForSeq2Seq(tokenizer),
        compute_metrics=compute_metrics,
        **trainer_tokenizer_kwargs(Seq2SeqTrainer, tokenizer),
    )
    trainer.train()
    trainer.save_model(str(args.output_dir))
    tokenizer.save_pretrained(str(args.output_dir))


if __name__ == "__main__":
    main()
