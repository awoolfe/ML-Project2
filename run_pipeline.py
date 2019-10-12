import pandas as pd
import numpy as np
from collections import Counter
from itertools import chain

from src.pipeline_utils import stratified_k_folds, ngram_tf
from src.model.BernoulliNaiveBayes import BernoulliNaiveBayes

if __name__ == '__main__':
    K = 5
    VOCAB_SIZE = 5_000
    OOV_TOKEN = "--UNK--"
    NGRAM = 1
    Y_COL = "subreddits"
    X_COL = "preprocessed_comments"

    # TODO: handle multiple ngrams at once

    df = pd.read_csv("data/reddit_spacy_train.csv")
    df.dropna(inplace=True)
    df[X_COL] = df[X_COL].apply(lambda x: x.split())

    ngrams = [[" ".join(x[i:i+NGRAM]) for i in range(len(x) + NGRAM - 1)]for x in df[X_COL]]
    vocab = Counter(list(chain.from_iterable(ngrams)))
    vocab_itos = [OOV_TOKEN] + [word for word, count in vocab.most_common(VOCAB_SIZE - 1)]
    vocab_stoi = {k:i for i,k in enumerate(vocab_itos)}

    train_folds, valid_folds = stratified_k_folds(df, Y_COL, K)
    for i, (train, valid) in enumerate(zip(train_folds, valid_folds)):
        X_train = [ngram_tf(x, vocab_stoi, NGRAM) for x in train[X_COL].to_list()]
        X_valid = [ngram_tf(x, vocab_stoi, NGRAM) for x in valid[X_COL].to_list()]
        Y_train = train[Y_COL].to_list()
        Y_valid = valid[Y_COL].to_list()

        model = BernoulliNaiveBayes()
        model.fit(np.array(X_train), np.array(Y_train))

        valid_pred = model.predict(np.array(X_valid))
        print(valid_pred)
        print(np.array(Y_valid))
        print("-"*80)





