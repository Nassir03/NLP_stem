from __future__ import annotations

import importlib
from pathlib import Path

import pandas as pd
import torch

from config import CFG, ensure_runtime_defaults
from kaggle_utils import sync_readonly_artifacts
from preprocessing.tokenizer import load_tokenizers
from models.attention import BahdanauAttention
from training.train import MODEL_MODULES, TRAINABLE_MODELS, batch_loss, build_model


REQUIRED_SPLIT_COLUMNS = {"source", "target"}


def require_file(path: Path, label: str) -> None:
    """Fail early with a readable message when a required project artifact is absent."""
    if not path.exists():
        raise FileNotFoundError(f"Missing {label}: {path}")


def check_splits() -> None:
    """Validate prepared train/validation/test CSVs without reading the full corpus."""
    for split in ("train", "validation", "test"):
        path = CFG.split_dir / f"{split}.csv"
        require_file(path, f"{split} split")
        sample = pd.read_csv(path, nrows=20)
        missing = REQUIRED_SPLIT_COLUMNS - set(sample.columns)
        if missing:
            raise ValueError(f"{path} is missing columns: {', '.join(sorted(missing))}")
        if sample.empty:
            raise ValueError(f"{path} is empty")
        print(f"[OK] {split} split: {path}")


def check_tokenizers() -> tuple[object, object]:
    """Load both tokenizers and run a tiny encode/decode smoke test."""
    src_tok, tgt_tok = load_tokenizers()
    for name, tok in (("source", src_tok), ("target", tgt_tok)):
        ids = tok.encode("small tokenizer test")
        if not ids or ids[0] != tok.bos_id or ids[-1] != tok.eos_id:
            raise ValueError(f"{name} tokenizer did not add BOS/EOS correctly")
        _ = tok.decode(ids)
        print(f"[OK] {name} tokenizer: vocab={tok.vocab_size}")
    return src_tok, tgt_tok


def check_model_registry(src_vocab: int, tgt_vocab: int) -> None:
    """Instantiate each registered model once so broken imports are caught centrally."""
    seen_modules = set()
    for model_name, module_name in sorted(MODEL_MODULES.items()):
        importlib.import_module(module_name)
        model = build_model(model_name, src_vocab, tgt_vocab)
        total_params = sum(p.numel() for p in model.parameters())
        if total_params <= 0:
            raise ValueError(f"{model_name} has no trainable parameters")
        duplicate = " alias" if module_name in seen_modules else ""
        seen_modules.add(module_name)
        print(f"[OK] {model_name}{duplicate}: {total_params:,} parameters")


def check_model_forward(model_name: str, src_tok, tgt_tok) -> None:
    """Run one tiny loss pass through a model without training or decoding."""
    model = build_model(model_name, src_tok.vocab_size, tgt_tok.vocab_size).to(CFG.device)
    src = torch.tensor([src_tok.encode("science is useful")[:8]], device=CFG.device)
    tgt = torch.tensor([tgt_tok.encode("sayansi ni muhimu")[:8]], device=CFG.device)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=tgt_tok.pad_id)
    with torch.no_grad():
        loss = batch_loss(model, model_name, src, tgt, criterion, teacher_forcing=0.0)
    if not torch.isfinite(loss):
        raise ValueError(f"{model_name} produced a non-finite smoke-test loss")
    print(f"[OK] {model_name} forward pass on {CFG.device}: loss={float(loss):.4f}")


def check_trainable_forwards(src_tok, tgt_tok) -> None:
    """Smoke-test every trainable architecture used by the Kaggle full run."""
    for model_name in TRAINABLE_MODELS:
        check_model_forward(model_name, src_tok, tgt_tok)


def check_attention_mask_dtype() -> None:
    """Catch AMP-style mask overflow in attention before Kaggle training starts."""
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    device = CFG.device
    attn = BahdanauAttention(hidden=4).to(device=device, dtype=dtype)
    decoder_hidden = torch.randn(2, 4, device=device, dtype=dtype)
    encoder_outputs = torch.randn(2, 3, 4, device=device, dtype=dtype)
    src_mask = torch.tensor([[True, True, False], [True, False, False]], device=device)
    with torch.no_grad():
        context, weights = attn(decoder_hidden, encoder_outputs, src_mask)
    if not torch.isfinite(context).all() or not torch.isfinite(weights).all():
        raise ValueError("Attention mask produced non-finite values")
    print(f"[OK] attention mask dtype check: dtype={dtype}")


def check_default_forward(src_tok, tgt_tok) -> None:
    """Keep the legacy shape check for the default attention model."""
    model = build_model("gru_attention", src_tok.vocab_size, tgt_tok.vocab_size).to(CFG.device)
    src = torch.tensor([src_tok.encode("science is useful")[:8]], device=CFG.device)
    tgt = torch.tensor([tgt_tok.encode("sayansi ni muhimu")[:8]], device=CFG.device)
    with torch.no_grad():
        logits = model(src, tgt, teacher_forcing=0.0)
    if logits.shape[:2] != (1, max(tgt.size(1) - 1, 0)):
        raise ValueError(f"Unexpected model output shape: {tuple(logits.shape)}")
    print(f"[OK] default forward pass on {CFG.device}: shape={tuple(logits.shape)}")


def check_project() -> None:
    """Project-level smoke test that does not train models or generate translations."""
    ensure_runtime_defaults()
    sync_readonly_artifacts()
    check_splits()
    src_tok, tgt_tok = check_tokenizers()
    check_model_registry(src_tok.vocab_size, tgt_tok.vocab_size)
    check_attention_mask_dtype()
    check_trainable_forwards(src_tok, tgt_tok)
    check_default_forward(src_tok, tgt_tok)
    print("Project check passed without generating translation answers.")


if __name__ == "__main__":
    check_project()
