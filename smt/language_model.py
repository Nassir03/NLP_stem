from __future__ import annotations
import math, pickle
from collections import Counter

class NGramLM:
    def __init__(self, n=3, alpha=0.1):
        self.n, self.alpha = n, alpha
        self.ngrams, self.contexts = Counter(), Counter()
        self.vocab = set()

    def fit(self, sentences):
        for s in sentences:
            words = ["<s>"] * (self.n-1) + str(s).lower().split() + ["</s>"]
            self.vocab.update(words)
            for i in range(self.n-1, len(words)):
                ng = tuple(words[i-self.n+1:i+1])
                self.ngrams[ng] += 1
                self.contexts[ng[:-1]] += 1

    def logprob(self, sentence):
        words = ["<s>"] * (self.n-1) + str(sentence).lower().split() + ["</s>"]
        score = 0.0
        V = max(len(self.vocab), 1)
        for i in range(self.n-1, len(words)):
            ng = tuple(words[i-self.n+1:i+1])
            p = (self.ngrams[ng] + self.alpha) / (self.contexts[ng[:-1]] + self.alpha*V)
            score += math.log(p)
        return score

    def perplexity(self, sentences):
        total_log, total_words = 0.0, 0
        for s in sentences:
            total_log += self.logprob(s)
            total_words += len(str(s).split()) + 1
        return math.exp(-total_log/max(total_words,1))
