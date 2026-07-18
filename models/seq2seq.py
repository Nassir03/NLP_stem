from __future__ import annotations
import random
import torch
from torch import nn

class Encoder(nn.Module):
    def __init__(self, vocab_size, emb, hidden, layers=1, cell="gru", dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb, padding_idx=0)
        cls = {"rnn": nn.RNN, "lstm": nn.LSTM, "gru": nn.GRU}[cell]
        self.rnn = cls(
            emb, hidden, layers, batch_first=True,
            dropout=dropout if layers > 1 else 0
        )
    def forward(self, src):
        return self.rnn(self.embedding(src))

class Decoder(nn.Module):
    def __init__(self, vocab_size, emb, hidden, layers=1, cell="gru", dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb, padding_idx=0)
        cls = {"rnn": nn.RNN, "lstm": nn.LSTM, "gru": nn.GRU}[cell]
        self.rnn = cls(
            emb, hidden, layers, batch_first=True,
            dropout=dropout if layers > 1 else 0
        )
        self.fc = nn.Linear(hidden, vocab_size)
    def forward(self, token, state):
        out, state = self.rnn(self.embedding(token.unsqueeze(1)), state)
        return self.fc(out.squeeze(1)), state

class Seq2Seq(nn.Module):
    def __init__(self, src_vocab, tgt_vocab, emb, hidden, layers=1, cell="gru", dropout=0.1):
        super().__init__()
        self.cell = cell
        self.encoder = Encoder(src_vocab, emb, hidden, layers, cell, dropout)
        self.decoder = Decoder(tgt_vocab, emb, hidden, layers, cell, dropout)

    def forward(self, src, tgt, teacher_forcing=0.5):
        _, state = self.encoder(src)
        outputs = []
        token = tgt[:, 0]
        for t in range(1, tgt.size(1)):
            logits, state = self.decoder(token, state)
            outputs.append(logits)
            token = tgt[:, t] if random.random() < teacher_forcing else logits.argmax(-1)
        return torch.stack(outputs, dim=1)
