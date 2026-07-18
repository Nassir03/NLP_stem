"""Educational IBM Model 1 word aligner trained with EM."""
from __future__ import annotations
import argparse, json
from collections import defaultdict
import pandas as pd
from tqdm import trange
from config import CFG

def tokenize(s): return str(s).lower().split()

class IBM1:
    def __init__(self):
        self.t = defaultdict(dict)

    def train(self, pairs, iterations=5):
        vocab = defaultdict(set)
        for src, tgt in pairs:
            for e in ["<null>"] + tokenize(src):
                vocab[e].update(tokenize(tgt))
        for e, fs in vocab.items():
            p = 1.0 / max(len(fs), 1)
            self.t[e] = {f: p for f in fs}

        for _ in trange(iterations, desc="IBM1 EM"):
            count, total = defaultdict(lambda: defaultdict(float)), defaultdict(float)
            for src, tgt in pairs:
                es, fs = ["<null>"] + tokenize(src), tokenize(tgt)
                for f in fs:
                    z = sum(self.t[e].get(f, 1e-12) for e in es)
                    for e in es:
                        c = self.t[e].get(f, 1e-12) / z
                        count[e][f] += c
                        total[e] += c
            for e in count:
                self.t[e] = {f: c / total[e] for f, c in count[e].items()}

    def align(self, src, tgt):
        es, fs = ["<null>"] + tokenize(src), tokenize(tgt)
        links = []
        for j, f in enumerate(fs):
            i = max(range(len(es)), key=lambda k: self.t[es[k]].get(f, 0.0))
            if i > 0: links.append((i - 1, j))
        return links

    def save(self, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(dict(self.t), f, ensure_ascii=False)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--iterations", type=int, default=5)
    p.add_argument("--limit", type=int, default=50000)
    args = p.parse_args()
    df = pd.read_csv(CFG.split_dir / "train.csv").head(args.limit)
    pairs = list(zip(df.source, df.target))
    model = IBM1(); model.train(pairs, args.iterations)
    model.save(CFG.checkpoint_dir / "ibm1_translation_table.json")
