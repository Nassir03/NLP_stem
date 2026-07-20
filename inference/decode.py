from __future__ import annotations
import importlib
import pickle
import torch
import torch.nn.functional as F
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
    try:
        # Future checkpoints are saved with primitive metadata, so the safer
        # tensor-focused loader is enough on Kaggle and local machines.
        ckpt = torch.load(ckpt_path, map_location=CFG.device, weights_only=True)
    except TypeError:
        ckpt = torch.load(ckpt_path, map_location=CFG.device)
    except pickle.UnpicklingError:
        # Older local checkpoints may contain Path objects in their metadata.
        # The project owns these files, so this fallback keeps them usable.
        ckpt = torch.load(ckpt_path, map_location=CFG.device, weights_only=False)
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


@torch.no_grad()
def beam_decode(
    model,
    model_name,
    text,
    src_tok,
    tgt_tok,
    max_len=None,
    beam_size=None,
    length_penalty=0.6,
):
    """Translate one sentence with beam search and normalized sequence scores."""
    max_len = max_len or CFG.max_seq_len
    beam_size = beam_size or CFG.beam_size
    src = torch.tensor([src_tok.encode(text)[:max_len]], device=CFG.device)
    beams = [([tgt_tok.bos_id], 0.0, False)]

    for _ in range(max_len - 1):
        candidates = []
        for tokens, score, finished in beams:
            if finished:
                candidates.append((tokens, score, finished))
                continue

            ys = torch.tensor([tokens], device=CFG.device)
            if model_name == "transformer":
                logits = model(src, ys)[:, -1]
            else:
                placeholder = torch.cat(
                    [ys, torch.full((1, 1), tgt_tok.eos_id, device=CFG.device)],
                    dim=1,
                )
                logits = model(src, placeholder, 0.0)[:, -1]

            log_probs = F.log_softmax(logits, dim=-1)
            top_scores, top_ids = log_probs.topk(beam_size, dim=-1)
            for token_score, token_id in zip(top_scores[0], top_ids[0]):
                next_id = int(token_id.item())
                candidates.append(
                    (
                        tokens + [next_id],
                        score + float(token_score.item()),
                        next_id == tgt_tok.eos_id,
                    )
                )

        def normalized(item):
            tokens, score, _ = item
            length = max(len(tokens) - 1, 1)
            penalty = ((5.0 + length) / 6.0) ** length_penalty
            return score / penalty

        beams = sorted(candidates, key=normalized, reverse=True)[:beam_size]
        if all(finished for _, _, finished in beams):
            break

    best_tokens = max(beams, key=lambda item: item[1] / (((5.0 + max(len(item[0]) - 1, 1)) / 6.0) ** length_penalty))[0]
    return tgt_tok.decode(best_tokens)


def decode_sentence(
    model,
    model_name,
    text,
    src_tok,
    tgt_tok,
    method="beam",
    max_len=None,
    beam_size=None,
    length_penalty=0.6,
):
    """Shared decoding entry point used by evaluation scripts."""
    if method == "greedy":
        return greedy_decode(model, model_name, text, src_tok, tgt_tok, max_len=max_len)
    if method == "beam":
        return beam_decode(
            model,
            model_name,
            text,
            src_tok,
            tgt_tok,
            max_len=max_len,
            beam_size=beam_size,
            length_penalty=length_penalty,
        )
    raise ValueError(f"Unknown decoding method: {method}")
