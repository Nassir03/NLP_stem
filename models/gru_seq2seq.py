from models.seq2seq import Seq2Seq
def build_model(src_vocab, tgt_vocab, emb, hidden, layers, dropout):
    return Seq2Seq(src_vocab, tgt_vocab, emb, hidden, layers, "gru", dropout)
