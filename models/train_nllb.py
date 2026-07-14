from __future__ import annotations

import argparse
import csv
import math
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
        import torch  # noqa: F401
        import transformers  # noqa: F401
    except ImportError as exc:
        raise SystemExit(
            "Install torch, transformers, datasets, sentencepiece and sacrebleu before running NLLB training."
        ) from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune facebook/nllb-200-distilled-600M safely on Kaggle.")
    parser.add_argument("--train-file", type=Path, default=DATA_DIR / "train.csv")
    parser.add_argument("--validation-file", type=Path, default=DATA_DIR / "validation.csv")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--model-name", default="facebook/nllb-200-distilled-600M")
    parser.add_argument("--max-source-length", type=int, default=256)
    parser.add_argument("--max-target-length", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--num-beams", type=int, default=5)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-validation-samples", type=int, default=None)
    parser.add_argument("--min-chars", type=int, default=2)
    parser.add_argument("--max-length-ratio", type=float, default=4.0)
    parser.add_argument("--logging-steps", type=int, default=100)
    parser.add_argument("--bleu-samples", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fp16", action="store_true", help="Use CUDA mixed precision. Keep off if Kaggle is unstable.")
    parser.add_argument(
        "--allow-multi-gpu",
        action="store_true",
        help="Disabled by default because Kaggle dual-GPU setups have repeatedly crashed NLLB training.",
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


def main() -> None:
    args = parse_args()
    if not args.allow_multi_gpu:
        restrict_to_single_gpu()

    require_libraries()

    from datasets import load_dataset
    import sacrebleu
    import torch
    from torch.nn.utils import clip_grad_norm_
    from torch.utils.data import DataLoader
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, get_linear_schedule_with_warmup

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    validate_csv(args.train_file)
    validate_csv(args.validation_file)

    dataset = load_dataset(
        "csv",
        data_files={"train": str(args.train_file), "validation": str(args.validation_file)},
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
    if args.max_train_samples:
        dataset["train"] = dataset["train"].select(range(min(args.max_train_samples, len(dataset["train"]))))
    if args.max_validation_samples:
        dataset["validation"] = dataset["validation"].select(
            range(min(args.max_validation_samples, len(dataset["validation"])))
        )

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
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False

    def tokenize_batch(batch):
        inputs = tokenizer(batch["english"], max_length=args.max_source_length, truncation=True)
        labels = tokenizer(text_target=batch["swahili"], max_length=args.max_target_length, truncation=True)
        inputs["labels"] = labels["input_ids"]
        return inputs

    tokenized = dataset.map(tokenize_batch, batched=True, remove_columns=dataset["train"].column_names)

    def collate(features):
        model_inputs = tokenizer.pad(
            {
                "input_ids": [feature["input_ids"] for feature in features],
                "attention_mask": [feature["attention_mask"] for feature in features],
            },
            return_tensors="pt",
        )
        labels = tokenizer.pad(
            {"input_ids": [feature["labels"] for feature in features]},
            return_tensors="pt",
        )["input_ids"]
        labels[labels == tokenizer.pad_token_id] = -100
        model_inputs["labels"] = labels
        return model_inputs

    generator = torch.Generator()
    generator.manual_seed(args.seed)
    train_loader = DataLoader(
        tokenized["train"],
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate,
        generator=generator,
    )
    validation_loader = DataLoader(
        tokenized["validation"],
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)
    optimizer_steps_per_epoch = math.ceil(len(train_loader) / args.gradient_accumulation_steps)
    total_optimizer_steps = max(optimizer_steps_per_epoch * args.epochs, 1)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=min(args.warmup_steps, max(total_optimizer_steps - 1, 0)),
        num_training_steps=total_optimizer_steps,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=args.fp16 and device.type == "cuda")

    def evaluate_loss() -> float:
        model.eval()
        total_loss = 0.0
        total_batches = 0
        with torch.no_grad():
            for batch in validation_loader:
                batch = {key: value.to(device) for key, value in batch.items()}
                outputs = model(**batch)
                total_loss += float(outputs.loss.detach().cpu())
                total_batches += 1
        return total_loss / max(total_batches, 1)

    def evaluate_bleu() -> float:
        if args.bleu_samples <= 0:
            return 0.0
        model.eval()
        predictions: list[str] = []
        references: list[str] = []
        sample_count = min(args.bleu_samples, len(dataset["validation"]))
        for start in range(0, sample_count, args.batch_size):
            rows = dataset["validation"][start : min(start + args.batch_size, sample_count)]
            source_texts = rows["english"]
            target_texts = rows["swahili"]
            inputs = tokenizer(
                source_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=args.max_source_length,
            ).to(device)
            with torch.no_grad():
                output_ids = model.generate(
                    **inputs,
                    forced_bos_token_id=forced_bos_token_id,
                    num_beams=args.num_beams,
                    max_new_tokens=args.max_target_length,
                    no_repeat_ngram_size=3,
                    early_stopping=True,
                )
            predictions.extend(text.strip() for text in tokenizer.batch_decode(output_ids, skip_special_tokens=True))
            references.extend(text.strip() for text in target_texts)
        return round(sacrebleu.corpus_bleu(predictions, [references]).score, 4)

    best_bleu = -1.0
    global_step = 0
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        running_loss = 0.0

        for step, batch in enumerate(train_loader, start=1):
            batch = {key: value.to(device) for key, value in batch.items()}
            with torch.cuda.amp.autocast(enabled=args.fp16 and device.type == "cuda"):
                outputs = model(**batch)
                loss = outputs.loss / args.gradient_accumulation_steps

            scaler.scale(loss).backward()
            running_loss += float(loss.detach().cpu()) * args.gradient_accumulation_steps

            if step % args.gradient_accumulation_steps == 0 or step == len(train_loader):
                scaler.unscale_(optimizer)
                clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

                if global_step % args.logging_steps == 0:
                    average_loss = running_loss / max(args.logging_steps * args.gradient_accumulation_steps, 1)
                    lr = scheduler.get_last_lr()[0]
                    print(f"step={global_step} epoch={epoch} loss={average_loss:.4f} lr={lr:.3e}", flush=True)
                    running_loss = 0.0

        eval_loss = evaluate_loss()
        bleu = evaluate_bleu()
        print(f"epoch={epoch} eval_loss={eval_loss:.4f} bleu={bleu:.4f}", flush=True)

        if bleu >= best_bleu:
            best_bleu = bleu
            model.save_pretrained(args.output_dir)
            tokenizer.save_pretrained(args.output_dir)
            print(f"Saved best checkpoint to {args.output_dir} with bleu={best_bleu:.4f}", flush=True)

    print(f"Training complete. Best BLEU={best_bleu:.4f}. Checkpoint: {args.output_dir.resolve()}", flush=True)


if __name__ == "__main__":
    main()
