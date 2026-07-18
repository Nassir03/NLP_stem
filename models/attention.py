import torch
from torch import nn

class BahdanauAttention(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.energy = nn.Linear(hidden * 2, hidden)
        self.score = nn.Linear(hidden, 1, bias=False)

    def forward(self, decoder_hidden, encoder_outputs, src_mask):
        # decoder_hidden: [B,H], encoder_outputs: [B,S,H]
        h = decoder_hidden.unsqueeze(1).expand(-1, encoder_outputs.size(1), -1)
        e = self.score(torch.tanh(self.energy(torch.cat([h, encoder_outputs], dim=-1)))).squeeze(-1)
        e = e.masked_fill(~src_mask, -1e9)
        weights = torch.softmax(e, dim=-1)
        context = torch.bmm(weights.unsqueeze(1), encoder_outputs).squeeze(1)
        return context, weights
