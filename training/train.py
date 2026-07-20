from __future__ import annotations

import argparse
import csv
import importlib
import math
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from tqdm import tqdm

from config import CFG, ensure_runtime_defaults
from preprocessing.dataset import get_loaders

# Every model wrapper in models/ exposes the same build_model function.  Keeping
# this registry here lets training, inference, and evaluation agree on names.
MODEL_MODULES = {
    "rnn": "models.rnn_seq2seq",
    "rnn_seq2seq": "models.rnn_seq2seq",
    "gru": "models.gru_seq2seq",
    "gru_seq2seq": "models.gru_seq2seq",
    "lstm": "models.lstm_seq2seq",
    "lstm_seq2seq": "models.lstm_seq2seq",
    "gru_attention": "models.gru_attention",
    "lstm_attention": "models.lstm_attention",
    "transformer": "models.transformer",
}

TRAINABLE_MODELS = [
    "rnn_seq2seq",
    "gru_seq2seq",
    "lstm_seq2seq",
    "gru_attention",
    "lstm_attention",
    "transformer",
]


def set_seed(seed: int) -> None:
    """Make training runs repeatable across Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = True


def amp_enabled() -> bool:
    """Use mixed precision on CUDA for faster Kaggle training."""
    return bool(getattr(CFG, "use_amp", True) and CFG.device == "cuda")


def make_grad_scaler():
    """Create a GradScaler across PyTorch versions."""
    if not amp_enabled():
        return None
    try:
        return torch.amp.GradScaler("cuda")
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler()


def autocast_context():
    """Create an autocast context across PyTorch versions."""
    if not amp_enabled():
        return torch.amp.autocast("cpu", enabled=False)
    try:
        return torch.amp.autocast("cuda")
    except (AttributeError, TypeError):
        return torch.cuda.amp.autocast()


def build_model(model_name: str, src_vocab: int, tgt_vocab: int) -> nn.Module:
    if model_name not in MODEL_MODULES:
        valid = ", ".join(sorted(MODEL_MODULES))
        raise ValueError(f"Unknown model '{model_name}'. Choose one of: {valid}")
    module = importlib.import_module(MODEL_MODULES[model_name])
    return module.build_model(
        src_vocab,
        tgt_vocab,
        CFG.embedding_dim,
        CFG.hidden_dim,
        CFG.num_layers,
        CFG.dropout,
    )


def batch_loss(model: nn.Module, model_name: str, src: torch.Tensor, tgt: torch.Tensor,
               criterion: nn.Module, teacher_forcing: float) -> torch.Tensor:
    """Compute next-token loss for both Transformer and recurrent models."""
    if model_name == "transformer":
        logits = model(src, tgt[:, :-1])
        gold = tgt[:, 1:]
    else:
        logits = model(src, tgt, teacher_forcing)
        gold = tgt[:, 1:]
    return criterion(logits.reshape(-1, logits.size(-1)), gold.reshape(-1))


def run_epoch(model: nn.Module, model_name: str, loader, criterion: nn.Module,
              optimizer: torch.optim.Optimizer | None = None,
              scaler=None) -> float:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_tokens = 0

    for src, tgt in tqdm(loader, leave=False, desc="train" if training else "valid"):
        src = src.to(CFG.device)
        tgt = tgt.to(CFG.device)
        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training), autocast_context():
            loss = batch_loss(
                model,
                model_name,
                src,
                tgt,
                criterion,
                CFG.teacher_forcing if training else 0.0,
            )
        if training:
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

        tokens = int(tgt[:, 1:].ne(criterion.ignore_index).sum().item())
        total_loss += float(loss.item()) * max(tokens, 1)
        total_tokens += max(tokens, 1)

    return total_loss / max(total_tokens, 1)


def write_history(path: Path, rows: list[dict[str, float | int]]) -> None:
    """Persist a small CSV that can be plotted without loading checkpoints."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "valid_loss", "valid_ppl"])
        writer.writeheader()
        writer.writerows(rows)


def checkpoint_config() -> dict[str, str | int | float | None]:
    """Store a portable config snapshot without OS-specific Path objects."""
    return {
        key: str(value) if isinstance(value, Path) else value
        for key, value in CFG.__dict__.items()
    }


def train(model_name: str, skip_existing: bool = False) -> Path:
    """Train one model and save the best validation checkpoint."""
    ensure_runtime_defaults()
    set_seed(CFG.seed)
    CFG.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    CFG.results_dir.mkdir(parents=True, exist_ok=True)

    loaders, src_tok, tgt_tok = get_loaders()
    model = build_model(model_name, src_tok.vocab_size, tgt_tok.vocab_size).to(CFG.device)
    criterion = nn.CrossEntropyLoss(ignore_index=tgt_tok.pad_id)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=CFG.learning_rate,
        weight_decay=getattr(CFG, "weight_decay", 1e-4),
    )
    scaler = make_grad_scaler()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=getattr(CFG, "lr_factor", 0.5),
        patience=getattr(CFG, "lr_patience", 2),
    )

    best_loss = math.inf
    best_path = CFG.checkpoint_dir / f"{model_name}_best.pt"
    if skip_existing and best_path.exists():
        print(f"Skipping {model_name}; checkpoint already exists: {best_path}")
        return best_path
    stale_epochs = 0
    history: list[dict[str, float | int]] = []

    for epoch in range(1, CFG.epochs + 1):
        train_loss = run_epoch(model, model_name, loaders["train"], criterion, optimizer, scaler)
        valid_loss = run_epoch(model, model_name, loaders["validation"], criterion)
        valid_ppl = math.exp(min(valid_loss, 20.0))
        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "valid_loss": valid_loss,
            "valid_ppl": valid_ppl,
        })
        print(
            f"epoch={epoch} train_loss={train_loss:.4f} "
            f"valid_loss={valid_loss:.4f} valid_ppl={valid_ppl:.2f} "
            f"lr={optimizer.param_groups[0]['lr']:.2e}"
        )
        scheduler.step(valid_loss)

        if valid_loss < best_loss:
            best_loss = valid_loss
            stale_epochs = 0
            torch.save(
                {
                    "model": model.state_dict(),
                    "model_name": model_name,
                    "src_vocab": src_tok.vocab_size,
                    "tgt_vocab": tgt_tok.vocab_size,
                    "config": checkpoint_config(),
                    "valid_loss": valid_loss,
                },
                best_path,
            )
        else:
            stale_epochs += 1
            if stale_epochs >= CFG.patience:
                print(f"Early stopping after {CFG.patience} epochs without validation improvement.")
                break

    write_history(CFG.results_dir / f"{model_name}_training_history.csv", history)
    print(f"Best checkpoint: {best_path}")
    return best_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gru_attention", choices=sorted(MODEL_MODULES))
    parser.add_argument("--epochs", type=int, help="Override config epochs for this run.")
    parser.add_argument("--patience", type=int, help="Override early-stopping patience for this run.")
    parser.add_argument("--batch-size", type=int, help="Override config batch size for this run.")
    parser.add_argument("--train-limit", type=int, help="Use only the first N training rows.")
    parser.add_argument("--valid-limit", type=int, help="Use only the first N validation rows.")
    parser.add_argument("--test-limit", type=int, help="Use only the first N test rows.")
    parser.add_argument("--skip-existing", action="store_true", help="Do not retrain if checkpoint exists.")
    args = parser.parse_args()
    if args.epochs:
        CFG.epochs = args.epochs
    if args.patience:
        CFG.patience = args.patience
    if args.batch_size:
        CFG.batch_size = args.batch_size
    if args.train_limit:
        CFG.train_limit = args.train_limit
    if args.valid_limit:
        CFG.valid_limit = args.valid_limit
    if args.test_limit:
        CFG.test_limit = args.test_limit
    train(args.model, skip_existing=args.skip_existing)
