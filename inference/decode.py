from __future__ import annotations
import importlib
import torch
from config import CFG
from training.train import MODEL_MODULES

def load_model(model_name, src_tok, tgt_tok):
    """Load a trained checkpoint for the named model registry entry."""
    ckpt_path = CFG.checkpoint_dir / f"{model_name}_best.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {ckpt_path}. Train first with "
            f"`python main.py train --model {model_name}`."
        )
    module = importlib.import_module(MODEL_MODULES[model_name])
    model = module.build_model(
        src_tok.vocab_size, tgt_tok.vocab_size, CFG.embedding_dim,
        CFG.hidden_dim, CFG.num_layers, CFG.dropout
    ).to(CFG.device)
    ckpt = torch.load(ckpt_path, map_location=CFG.device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model

@torch.no_grad()
def greedy_decode(model, model_name, text, src_tok, tgt_tok, max_len=None):
    """Translate one sentence with greedy decoding; used only by explicit generation."""
    max_len = max_len or CFG.max_seq_len
    src = torch.tensor([src_tok.encode(text)[:max_len]], device=CFG.device)
    ys = torch.tensor([[tgt_tok.bos_id]], device=CFG.device)

    if model_name == "transformer":
        for _ in range(max_len - 1):
            logits = model(src, ys)
            nxt = logits[:, -1].argmax(-1, keepdim=True)
            ys = torch.cat([ys, nxt], dim=1)
            if nxt.item() == tgt_tok.eos_id: break
        return tgt_tok.decode(ys[0].tolist())

    # Recurrent models are decoded through their regular forward function with a growing
    # placeholder target. This keeps one shared interface, although beam search is preferable.
    for _ in range(max_len - 1):
        placeholder = torch.cat([ys, torch.full((1, 1), tgt_tok.eos_id, device=CFG.device)], 1)
        logits = model(src, placeholder, 0.0)
        nxt = logits[:, -1].argmax(-1, keepdim=True)
        ys = torch.cat([ys, nxt], dim=1)
        if nxt.item() == tgt_tok.eos_id: break
    return tgt_tok.decode(ys[0].tolist())
