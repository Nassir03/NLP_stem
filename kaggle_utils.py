from __future__ import annotations

import shutil
from pathlib import Path

from config import CFG


ARTIFACT_DIRS = [
    ("processed data", CFG.root / "data" / "processed", CFG.processed_dir),
    ("split data", CFG.root / "data" / "split", CFG.split_dir),
    ("tokenizers", CFG.root / "tokenizers", CFG.tokenizer_dir),
    ("checkpoints", CFG.root / "checkpoints", CFG.checkpoint_dir),
]


def copy_missing_tree(label: str, source: Path, destination: Path) -> None:
    """Copy a read-only Kaggle input artifact once into the writable artifact root."""
    if source == destination or destination.exists() or not source.exists():
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, destination)
    print(f"[Kaggle] copied {label}: {source} -> {destination}")


def sync_readonly_artifacts() -> None:
    """Make prebuilt data/tokenizer/checkpoint folders available from writable paths.

    Kaggle mounts uploaded datasets under /kaggle/input as read-only.  The project
    writes generated artifacts under /kaggle/working when it detects that layout,
    so this helper copies already prepared inputs only when the writable copy is
    missing.  Local runs are unchanged because source and destination are equal.
    """
    for label, source, destination in ARTIFACT_DIRS:
        copy_missing_tree(label, source, destination)
