def alignment_prf(predicted, gold):
    pred, gold = set(map(tuple, predicted)), set(map(tuple, gold))
    tp = len(pred & gold)
    precision = tp / max(len(pred), 1)
    recall = tp / max(len(gold), 1)
    f1 = 2*precision*recall / max(precision+recall, 1e-12)
    return {"precision": precision, "recall": recall, "f1": f1}

def alignment_error_rate(predicted, sure, possible=None):
    A, S = set(map(tuple, predicted)), set(map(tuple, sure))
    P = set(map(tuple, possible or sure))
    return 1 - (len(A & S) + len(A & P)) / max(len(A) + len(S), 1)
