from dataclasses import dataclass, field
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parent

@dataclass
class Config:
    root: Path = ROOT
    original_dir: Path = ROOT / "data" / "original"
    processed_dir: Path = ROOT / "data" / "processed"
    split_dir: Path = ROOT / "data" / "split"
    tokenizer_dir: Path = ROOT / "tokenizers"
    checkpoint_dir: Path = ROOT / "checkpoints"
    results_dir: Path = ROOT / "results"

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
    vocab_size: int = 12000
    max_seq_len: int = 100

    embedding_dim: int = 256
    hidden_dim: int = 384
    num_layers: int = 2
    dropout: float = 0.2
    batch_size: int = 64
    epochs: int = 15
    learning_rate: float = 3e-4
    teacher_forcing: float = 0.5
    beam_size: int = 4
    patience: int = 4

    device: str = "cuda" if torch.cuda.is_available() else "cpu"

CFG = Config()
