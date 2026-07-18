"""Simple lexical SMT baseline. It is educational, not a replacement for Moses."""
from __future__ import annotations
import json, argparse
import pandas as pd
from config import CFG
from smt.language_model import NGramLM

class LexicalDecoder:
    def __init__(self, table, lm=None):
        self.table, self.lm = table, lm

    def candidates(self, word, k=3):
        row = self.table.get(word.lower(), {})
        return sorted(row.items(), key=lambda x: x[1], reverse=True)[:k] or [(word, 1e-8)]

    def translate(self, sentence):
        # Monotonic Viterbi approximation. HMM alignments are used during analysis/training;
        # a production phrase decoder should be substituted for publication-scale SMT.
        words = str(sentence).split()
        return " ".join(self.candidates(w, 1)[0][0] for w in words)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--limit", type=int)
    args = p.parse_args()
    table = json.loads((CFG.checkpoint_dir / "ibm1_translation_table.json").read_text(encoding="utf-8"))
    dec = LexicalDecoder(table)
    df = pd.read_csv(CFG.split_dir / "test.csv")
    if args.limit: df = df.head(args.limit)
    df["prediction"] = [dec.translate(x) for x in df.source]
    df.to_csv(CFG.results_dir / "smt_predictions.csv", index=False)
