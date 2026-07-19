from __future__ import annotations
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from config import CFG
from kaggle_utils import sync_readonly_artifacts
from preprocessing.tokenizer import load_tokenizers

class TranslationDataset(Dataset):
    def __init__(self, csv_path, src_tok, tgt_tok, limit=None):
        """Load one split, optionally shortened for Kaggle smoke tests."""
        self.df = pd.read_csv(csv_path)
        if limit:
            self.df = self.df.head(limit).reset_index(drop=True)
        self.src_tok, self.tgt_tok = src_tok, tgt_tok

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        src = self.src_tok.encode(str(row.source))[:CFG.max_seq_len]
        tgt = self.tgt_tok.encode(str(row.target))[:CFG.max_seq_len]
        if src[-1] != self.src_tok.eos_id: src[-1] = self.src_tok.eos_id
        if tgt[-1] != self.tgt_tok.eos_id: tgt[-1] = self.tgt_tok.eos_id
        return torch.tensor(src), torch.tensor(tgt)

def make_collate(src_pad, tgt_pad):
    def collate(batch):
        src, tgt = zip(*batch)
        src = pad_sequence(src, batch_first=True, padding_value=src_pad)
        tgt = pad_sequence(tgt, batch_first=True, padding_value=tgt_pad)
        return src, tgt
    return collate

def get_loaders():
    """Build train/validation/test loaders from prepared split CSV files."""
    sync_readonly_artifacts()
    src_tok, tgt_tok = load_tokenizers()
    collate = make_collate(src_tok.pad_id, tgt_tok.pad_id)
    loaders = {}
    # getattr keeps Kaggle runs compatible with older Config objects that were
    # copied before test_limit existed.
    limits = {
        "train": getattr(CFG, "train_limit", None),
        "validation": getattr(CFG, "valid_limit", None),
        "test": getattr(CFG, "test_limit", None),
    }
    for name, shuffle in [("train", True), ("validation", False), ("test", False)]:
        ds = TranslationDataset(CFG.split_dir / f"{name}.csv", src_tok, tgt_tok, limits[name])
        loaders[name] = DataLoader(
            ds, batch_size=CFG.batch_size, shuffle=shuffle,
            collate_fn=collate, num_workers=0, pin_memory=torch.cuda.is_available()
        )
    return loaders, src_tok, tgt_tok
