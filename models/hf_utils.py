from __future__ import annotations

import os


def hf_token() -> str | None:
    """Return a Hugging Face token from common notebook environment variables."""
    return os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")


def auth_kwargs() -> dict[str, str]:
    token = hf_token()
    return {"token": token} if token else {}


def restrict_to_single_gpu() -> None:
    """Avoid Trainer DataParallel issues on Kaggle dual-GPU notebooks."""
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
