from __future__ import annotations
import random
import torch
from torch import nn
from models.attention import BahdanauAttention

class AttentionSeq2Seq(nn.Module):
    def __init__(self, src_vocab, tgt_vocab, emb, hidden, cell="gru", dropout=0.1, src_pad_id=0):
        super().__init__()
        self.cell, self.src_pad_id = cell, src_pad_id
        rnn = nn.LSTM if cell == "lstm" else nn.GRU
        self.src_emb = nn.Embedding(src_vocab, emb, padding_idx=src_pad_id)
        self.tgt_emb = nn.Embedding(tgt_vocab, emb, padding_idx=0)
        self.encoder = rnn(emb, hidden, batch_first=True)
        self.attn = BahdanauAttention(hidden)
        self.decoder = rnn(emb + hidden, hidden, batch_first=True)
        self.fc = nn.Linear(hidden * 2, tgt_vocab)
        self.dropout = nn.Dropout(dropout)

    def _hidden(self, state):
        return state[0][-1] if self.cell == "lstm" else state[-1]

    def forward(self, src, tgt, teacher_forcing=0.5):
        enc, state = self.encoder(self.dropout(self.src_emb(src)))
        mask = src.ne(self.src_pad_id)
        token, outputs = tgt[:, 0], []
        for t in range(1, tgt.size(1)):
            context, _ = self.attn(self._hidden(state), enc, mask)
            x = torch.cat([self.dropout(self.tgt_emb(token)), context], dim=-1).unsqueeze(1)
            dec, state = self.decoder(x, state)
            logits = self.fc(torch.cat([dec.squeeze(1), context], dim=-1))
            outputs.append(logits)
            token = tgt[:, t] if random.random() < teacher_forcing else logits.argmax(-1)
        return torch.stack(outputs, dim=1)
