"""Compact HMM/Viterbi aligner using IBM1 lexical probabilities and jump probabilities."""
from __future__ import annotations
import math
from collections import Counter, defaultdict
from smt.em_alignment import tokenize, IBM1

class HMMAligner:
    def __init__(self, lexical: IBM1, jump_smoothing=0.1):
        self.lexical = lexical
        self.jump = Counter()
        self.total = 0
        self.jump_smoothing = jump_smoothing

    def fit_jumps(self, pairs):
        for src, tgt in pairs:
            links = sorted(self.lexical.align(src, tgt), key=lambda x: x[1])
            for (i1, _), (i2, _) in zip(links, links[1:]):
                self.jump[i2 - i1] += 1
                self.total += 1

    def jump_logp(self, d):
        return math.log((self.jump[d] + self.jump_smoothing) /
                        (self.total + self.jump_smoothing * (2 * 100 + 1)))

    def viterbi(self, src, tgt):
        es, fs = tokenize(src), tokenize(tgt)
        if not es or not fs: return []
        dp = [{i: (math.log(self.lexical.t[es[i]].get(fs[0], 1e-12)), [i])
               for i in range(len(es))}]
        for f in fs[1:]:
            cur = {}
            for i in range(len(es)):
                emit = math.log(self.lexical.t[es[i]].get(f, 1e-12))
                score, path = max(
                    (prev_score + self.jump_logp(i-j) + emit, prev_path + [i])
                    for j, (prev_score, prev_path) in dp[-1].items()
                )
                cur[i] = (score, path)
            dp.append(cur)
        path = max(dp[-1].values(), key=lambda x: x[0])[1]
        return [(i, j) for j, i in enumerate(path)]
