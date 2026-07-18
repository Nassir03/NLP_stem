from __future__ import annotations
import json
import re
from collections import Counter
from pathlib import Path
import pandas as pd
import sentencepiece as spm
from config import CFG

SPECIALS = ["<pad>", "<unk>", "<bos>", "<eos>"]
TOKEN_RE = re.compile(r"\w+|[^\w\s]", flags=re.UNICODE)

class WordTokenizer:
    def __init__(self, token_to_id=None):
        self.token_to_id = token_to_id or {t: i for i, t in enumerate(SPECIALS)}
        self.id_to_token = {i: t for t, i in self.token_to_id.items()}

    def train(self, texts, vocab_size=12000):
        counts = Counter(tok.lower() for text in texts for tok in TOKEN_RE.findall(text))
        self.token_to_id = {t: i for i, t in enumerate(SPECIALS)}
        for token, _ in counts.most_common(max(0, vocab_size - len(SPECIALS))):
            self.token_to_id.setdefault(token, len(self.token_to_id))
        self.id_to_token = {i: t for t, i in self.token_to_id.items()}

    @property
    def pad_id(self): return self.token_to_id["<pad>"]
    @property
    def unk_id(self): return self.token_to_id["<unk>"]
    @property
    def bos_id(self): return self.token_to_id["<bos>"]
    @property
    def eos_id(self): return self.token_to_id["<eos>"]
    @property
    def vocab_size(self): return len(self.token_to_id)

    def encode(self, text, add_special=True):
        ids = [self.token_to_id.get(t.lower(), self.unk_id) for t in TOKEN_RE.findall(text)]
        return ([self.bos_id] + ids + [self.eos_id]) if add_special else ids

    def decode(self, ids):
        toks = [self.id_to_token.get(int(i), "<unk>") for i in ids]
        toks = [t for t in toks if t not in SPECIALS]
        text = " ".join(toks)
        return re.sub(r"\s+([.,!?;:%)])", r"\1", text)

    def save(self, path):
        Path(path).write_text(json.dumps(self.token_to_id, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path):
        return cls(json.loads(Path(path).read_text(encoding="utf-8")))

class SentencePieceTokenizer:
    def __init__(self, model_path):
        self.model_path = str(model_path)
        self.sp = spm.SentencePieceProcessor(model_file=self.model_path)

    @property
    def pad_id(self): return self.sp.pad_id()
    @property
    def unk_id(self): return self.sp.unk_id()
    @property
    def bos_id(self): return self.sp.bos_id()
    @property
    def eos_id(self): return self.sp.eos_id()
    @property
    def vocab_size(self): return self.sp.vocab_size()

    def encode(self, text, add_special=True):
        ids = self.sp.encode(text, out_type=int)
        return ([self.bos_id] + ids + [self.eos_id]) if add_special else ids

    def decode(self, ids):
        clean = [int(i) for i in ids if int(i) not in {self.pad_id, self.bos_id, self.eos_id}]
        return self.sp.decode(clean)

def train_tokenizers():
    df = pd.read_csv(CFG.split_dir / "train.csv")
    CFG.tokenizer_dir.mkdir(parents=True, exist_ok=True)

    if CFG.tokenizer_type == "word":
        src, tgt = WordTokenizer(), WordTokenizer()
        src.train(df.source.astype(str), CFG.vocab_size)
        tgt.train(df.target.astype(str), CFG.vocab_size)
        src.save(CFG.tokenizer_dir / "source_word.json")
        tgt.save(CFG.tokenizer_dir / "target_word.json")
    else:
        for col, name in [("source", "source_sp"), ("target", "target_sp")]:
            txt = CFG.tokenizer_dir / f"{name}.txt"
            txt.write_text("\n".join(df[col].astype(str)), encoding="utf-8")
            spm.SentencePieceTrainer.train(
                input=str(txt),
                model_prefix=str(CFG.tokenizer_dir / name),
                vocab_size=CFG.vocab_size,
                model_type="unigram",
                character_coverage=1.0,
                pad_id=0, unk_id=1, bos_id=2, eos_id=3,
                hard_vocab_limit=False,
            )
            txt.unlink(missing_ok=True)
    print("Tokenizers trained.")

def load_tokenizers():
    if CFG.tokenizer_type == "word":
        return (
            WordTokenizer.load(CFG.tokenizer_dir / "source_word.json"),
            WordTokenizer.load(CFG.tokenizer_dir / "target_word.json"),
        )
    return (
        SentencePieceTokenizer(CFG.tokenizer_dir / "source_sp.model"),
        SentencePieceTokenizer(CFG.tokenizer_dir / "target_sp.model"),
    )

if __name__ == "__main__":
    train_tokenizers()
