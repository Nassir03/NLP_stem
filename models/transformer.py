from __future__ import annotations
import math
import torch
from torch import nn

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe[:, 0::2], pe[:, 1::2] = torch.sin(pos * div), torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class TransformerMT(nn.Module):
    def __init__(self, src_vocab, tgt_vocab, d_model=256, nhead=8, layers=4,
                 ff=1024, dropout=0.1, src_pad=0, tgt_pad=0):
        super().__init__()
        if hasattr(torch.backends, "mha") and hasattr(torch.backends.mha, "set_fastpath_enabled"):
            torch.backends.mha.set_fastpath_enabled(False)
        self.d_model, self.src_pad, self.tgt_pad = d_model, src_pad, tgt_pad
        self.src_emb = nn.Embedding(src_vocab, d_model, padding_idx=src_pad)
        self.tgt_emb = nn.Embedding(tgt_vocab, d_model, padding_idx=tgt_pad)
        self.pos = PositionalEncoding(d_model)
        self.net = nn.Transformer(
            d_model=d_model, nhead=nhead, num_encoder_layers=layers,
            num_decoder_layers=layers, dim_feedforward=ff,
            dropout=dropout, batch_first=True
        )
        self.fc = nn.Linear(d_model, tgt_vocab)

    def forward(self, src, tgt_in):
        src_pad_mask = src.eq(self.src_pad)
        tgt_pad_mask = tgt_in.eq(self.tgt_pad)
        # Boolean causal masks match the padding-mask dtype and avoid PyTorch
        # deprecation warnings on newer Kaggle runtimes.
        causal = torch.triu(
            torch.ones(tgt_in.size(1), tgt_in.size(1), device=tgt_in.device, dtype=torch.bool),
            diagonal=1,
        )
        src_e = self.pos(self.src_emb(src) * math.sqrt(self.d_model))
        tgt_e = self.pos(self.tgt_emb(tgt_in) * math.sqrt(self.d_model))
        out = self.net(
            src_e, tgt_e, tgt_mask=causal,
            src_key_padding_mask=src_pad_mask,
            tgt_key_padding_mask=tgt_pad_mask,
            memory_key_padding_mask=src_pad_mask
        )
        return self.fc(out)

def build_model(src_vocab, tgt_vocab, emb, hidden, layers, dropout):
    return TransformerMT(src_vocab, tgt_vocab, d_model=emb, layers=layers, dropout=dropout)
