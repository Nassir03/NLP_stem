from models.seq2seq import Seq2Seq

def build_model(src_vocab, tgt_vocab, emb, hidden, layers, dropout):
    # Plain LSTM encoder-decoder; attention variants live in lstm_attention.py.
    return Seq2Seq(src_vocab, tgt_vocab, emb, hidden, layers, "lstm", dropout)
