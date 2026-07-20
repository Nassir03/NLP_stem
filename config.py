from dataclasses import dataclass, field
import os
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parent


def _env_int(name: str, default: int | None) -> int | None:
    """Read optional integer overrides from Kaggle/notebook environment variables."""
    value = os.environ.get(name)
    return int(value) if value not in (None, "") else default


def _env_float(name: str, default: float) -> float:
    """Read optional float overrides while keeping local defaults unchanged."""
    value = os.environ.get(name)
    return float(value) if value not in (None, "") else default


def _env_bool(name: str, default: bool) -> bool:
    """Read boolean environment flags such as 1/true/yes/on."""
    value = os.environ.get(name)
    if value in (None, ""):
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _artifact_root() -> Path:
    if os.environ.get("KAGGLE_KERNEL_RUN_TYPE") and str(ROOT).startswith("/kaggle/input"):
        return Path("/kaggle/working") / ROOT.name
    return ROOT

@dataclass
class Config:
    root: Path = ROOT
    original_dir: Path = ROOT / "data" / "original"
    artifact_root: Path = field(default_factory=_artifact_root)
    processed_dir: Path = field(init=False)
    split_dir: Path = field(init=False)
    tokenizer_dir: Path = field(init=False)
    checkpoint_dir: Path = field(init=False)
    results_dir: Path = field(init=False)

    source_lang: str = "en"
    target_lang: str = "sw"
    seed: int = 42

    # These names exactly match the uploaded parallel corpora.
    corpora: list[tuple[str, str, str]] = field(default_factory=lambda: [
        ("wikimedia", "wikimedia.en-sw.en", "wikimedia.en-sw.sw"),
        ("health", "ELRC-wikipedia_health.en-sw.en", "ELRC-wikipedia_health.en-sw.sw"),
        ("ted", "TED2020.en-sw.en", "TED2020.en-sw.sw"),
        ("globalvoices", "GlobalVoices.en-sw.en", "GlobalVoices.en-sw.sw"),
        ("wikimatrix", "WikiMatrix.en-sw.en", "WikiMatrix.en-sw.sw"),
    ])

    min_words: int = 2
    max_words: int = 100
    max_length_ratio: float = 3.0
    train_ratio: float = 0.80
    valid_ratio: float = 0.10
    test_ratio: float = 0.10

    tokenizer_type: str = "sentencepiece"  # word | sentencepiece
    vocab_size: int = field(default_factory=lambda: _env_int("MT_VOCAB_SIZE", 12000) or 12000)
    max_seq_len: int = field(default_factory=lambda: _env_int("MT_MAX_SEQ_LEN", 100) or 100)

    embedding_dim: int = field(default_factory=lambda: _env_int("MT_EMBEDDING_DIM", 256) or 256)
    hidden_dim: int = field(default_factory=lambda: _env_int("MT_HIDDEN_DIM", 384) or 384)
    num_layers: int = field(default_factory=lambda: _env_int("MT_NUM_LAYERS", 2) or 2)
    dropout: float = field(default_factory=lambda: _env_float("MT_DROPOUT", 0.2))
    batch_size: int = field(default_factory=lambda: _env_int("MT_BATCH_SIZE", 128) or 128)
    epochs: int = field(default_factory=lambda: _env_int("MT_EPOCHS", 15) or 15)
    learning_rate: float = field(default_factory=lambda: _env_float("MT_LEARNING_RATE", 3e-4))
    teacher_forcing: float = field(default_factory=lambda: _env_float("MT_TEACHER_FORCING", 0.5))
    beam_size: int = 4
    patience: int = field(default_factory=lambda: _env_int("MT_PATIENCE", 4) or 4)
    lr_patience: int = field(default_factory=lambda: _env_int("MT_LR_PATIENCE", 2) or 2)
    lr_factor: float = field(default_factory=lambda: _env_float("MT_LR_FACTOR", 0.5))
    num_workers: int = field(
        default_factory=lambda: _env_int("MT_NUM_WORKERS", 0 if os.name == "nt" else 2) or 0
    )
    use_amp: bool = field(default_factory=lambda: _env_bool("MT_USE_AMP", True))
    train_limit: int | None = field(default_factory=lambda: _env_int("MT_TRAIN_LIMIT", None))
    valid_limit: int | None = field(default_factory=lambda: _env_int("MT_VALID_LIMIT", None))
    test_limit: int | None = field(default_factory=lambda: _env_int("MT_TEST_LIMIT", None))

    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    def __post_init__(self) -> None:
        self.processed_dir = self.artifact_root / "data" / "processed"
        self.split_dir = self.artifact_root / "data" / "split"
        self.tokenizer_dir = self.artifact_root / "tokenizers"
        self.checkpoint_dir = self.artifact_root / "checkpoints"
        self.results_dir = self.artifact_root / "results"

    def __getattr__(self, name: str):
        """Return safe defaults for optional runtime fields used by older files.

        This protects Kaggle sessions where one file was refreshed but another
        stale module still tries to access fields like CFG.test_limit directly.
        """
        if name in {"train_limit", "valid_limit", "test_limit"}:
            return None
        raise AttributeError(f"{type(self).__name__!s} object has no attribute {name!r}")

CFG = Config()


def ensure_runtime_defaults() -> Config:
    """Backfill optional runtime fields for older notebooks or stale Kaggle copies.

    Kaggle notebooks often keep a working directory between cells.  If a user
    updates some files but not all generated state, this helper prevents newer
    modules from failing on optional Config attributes that older code did not
    define yet.
    """
    defaults = {
        "train_limit": _env_int("MT_TRAIN_LIMIT", None),
        "valid_limit": _env_int("MT_VALID_LIMIT", None),
        "test_limit": _env_int("MT_TEST_LIMIT", None),
        "artifact_root": _artifact_root(),
    }
    for name, value in defaults.items():
        if not hasattr(CFG, name):
            setattr(CFG, name, value)

    path_defaults = {
        "processed_dir": CFG.artifact_root / "data" / "processed",
        "split_dir": CFG.artifact_root / "data" / "split",
        "tokenizer_dir": CFG.artifact_root / "tokenizers",
        "checkpoint_dir": CFG.artifact_root / "checkpoints",
        "results_dir": CFG.artifact_root / "results",
    }
    for name, value in path_defaults.items():
        if not hasattr(CFG, name):
            setattr(CFG, name, value)
    return CFG
