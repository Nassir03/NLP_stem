from models.attention_seq2seq import AttentionSeq2Seq
def build_model(src_vocab, tgt_vocab, emb, hidden, layers, dropout):
    return AttentionSeq2Seq(src_vocab, tgt_vocab, emb, hidden, "gru", dropout)
